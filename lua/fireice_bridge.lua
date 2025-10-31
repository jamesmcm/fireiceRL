-- Fire 'n Ice Lua bridge for FCEUX <-> Python PPO training.
-- Requires lua-zmq (ZeroMQ bindings), dkjson (JSON encoder), and luasocket (for mime/base64 helpers).
local function augment_package_paths()
  local home = os.getenv("HOME")
  if not home then
    return
  end

  local share = home .. "/.luarocks/share/lua/5.1/?.lua"
  local share_init = home .. "/.luarocks/share/lua/5.1/?/init.lua"
  local lib = home .. "/.luarocks/lib/lua/5.1/?.so"

  if not package.path:find(share, 1, true) then
    package.path = share .. ";" .. package.path
  end
  if not package.path:find(share_init, 1, true) then
    package.path = share_init .. ";" .. package.path
  end

  if not package.cpath:find(lib, 1, true) then
    package.cpath = lib .. ";" .. package.cpath
  end
end

augment_package_paths()

local function normalize_path(path)
  return (path or ""):gsub("\\", "/")
end

local SCRIPT_SOURCE = debug.getinfo(1, "S").source or ""
local SCRIPT_DIR = normalize_path(SCRIPT_SOURCE):match("@(.*/)") or "./"
if SCRIPT_DIR:sub(-1) ~= "/" then
  SCRIPT_DIR = SCRIPT_DIR .. "/"
end

local PROJECT_ROOT = SCRIPT_DIR:match("^(.*)/lua/?$") or SCRIPT_DIR
if PROJECT_ROOT:sub(-1) ~= "/" then
  PROJECT_ROOT = PROJECT_ROOT .. "/"
end

local function is_absolute_path(path)
  path = normalize_path(path)
  return path:sub(1, 1) == "/" or path:match("^%a:/")
end

local function resolve_project_path(path)
  if not path or path == "" then
    return nil
  end
  path = normalize_path(path)
  if is_absolute_path(path) then
    return path
  end
  return PROJECT_ROOT .. path
end

local function require_zmq()
  local candidates = { "zmq", "zmq.core", "lzmq" }
  for _, name in ipairs(candidates) do
    local ok, module_or_err = pcall(require, name)
    if ok then
      local module = module_or_err
      if module then
        return module
      end
    else
      print(string.format("[fireice] failed to require '%s': %s", name, tostring(module_or_err)))
    end
  end
  error("ZeroMQ Lua binding not found. Install lua-zmq or lzmq and ensure it is on package.path/cpath.")
end

local zmq = require_zmq()
local json = require("dkjson")
local luasocket = require("socket")
local mime = require("mime")

local function create_context()
  if zmq.context then
    return zmq.context()
  elseif zmq.init then
    return zmq.init(1)
  else
    error("Loaded ZeroMQ binding does not expose context/init constructor.")
  end
end

local context, ctx_err = create_context()
assert(context, ctx_err)

local function create_socket(ctx, pattern)
  local attempts = {
    function()
      if type(ctx.socket) == "function" then
        return ctx:socket(pattern)
      end
    end,
    function()
      if type(ctx.socket) == "function" then
        return ctx:socket("REP")
      end
    end,
    function()
      if zmq.socket then
        return zmq.socket(ctx, pattern)
      end
    end,
    function()
      if zmq.socket then
        return zmq.socket(ctx, "REP")
      end
    end,
  }

  for _, attempt in ipairs(attempts) do
    local creator = attempt
    if creator then
      local ok, sock, err = pcall(creator)
      if ok and sock then
        return sock, err
      elseif ok and sock == nil and err then
        return sock, err
      end
    end
  end

  error("ZeroMQ binding does not support socket creation via known APIs.")
end

local ZMQ_REP = zmq.REP or zmq.REP_SOCKET or 4

local responder, sock_err = create_socket(context, ZMQ_REP)
assert(responder, sock_err)

local function pick_port_candidates()
  local env_port = tonumber(os.getenv("FIREICE_PORT") or "")
  local base = env_port or 5555
  local attempts = tonumber(os.getenv("FIREICE_PORT_ATTEMPTS") or "") or 20
  local step = tonumber(os.getenv("FIREICE_PORT_STEP") or "") or 1
  if attempts < 1 then
    attempts = 1
  end
  local ports = {}
  for i = 0, attempts - 1 do
    table.insert(ports, base + i * step)
  end
  if not env_port then
    table.insert(ports, 0)
  end
  return ports
end

local ACTIVE_PORT = nil

local function try_bind_port(port)
  local endpoint
  if port and port > 0 then
    endpoint = string.format("tcp://*:%d", port)
  else
    endpoint = "tcp://*:*"
  end

  local ok, err
  if type(responder.bind) == "function" then
    ok, err = responder:bind(endpoint)
  elseif zmq.bind then
    ok, err = zmq.bind(responder, endpoint)
  else
    error("ZeroMQ binding cannot bind sockets via known APIs.")
  end

  if ok then
    if endpoint:sub(-2) == ":*" and responder.getsockopt and zmq.LAST_ENDPOINT then
      local ok_last, last_or_err = pcall(function()
        return responder:getsockopt(zmq.LAST_ENDPOINT)
      end)
      if ok_last and last_or_err then
        local extracted = tostring(last_or_err):match(":%d+$")
        if extracted then
          port = tonumber(extracted:sub(2))
        end
      elseif not ok_last then
        print(string.format("[fireice] warning: failed to read dynamic bind endpoint: %s", tostring(last_or_err)))
      end
    end
    return true, port
  end

  return false, err
end

local bind_success = false
for _, port in ipairs(pick_port_candidates()) do
  local ok, result = try_bind_port(port)
  if ok then
    ACTIVE_PORT = result
    bind_success = true
    break
  else
    print(string.format("[fireice] failed to bind on port candidate %s: %s", tostring(port), tostring(result)))
  end
end

assert(bind_success, "ZeroMQ bridge failed to bind to any requested port.")
print(string.format("[fireice] bound to tcp://*:%d", ACTIVE_PORT or -1))

local FRAME_WIDTH = 256
local FRAME_HEIGHT = 240
local DEFAULT_SAVE_STATE = resolve_project_path("roms/1-1-nounlock.sav")
local LEVEL_LOAD_TIMEOUT_FRAMES = 300
local POST_LOAD_IDLE_FRAMES = 20
local STATE_SETTLE_FRAMES = 30
local FRAME_SKIP = 1

local BUTTON_MAP = {
  NOOP = nil,
  LEFT = { left = true },
  RIGHT = { right = true },
  A = { A = true },
  B = { B = true },
  START = { start = true },
}

local MONITORED_ADDRS = {
  0x0018,
  0x001C,
  0x00D0,
  0x00AB,
  0x00B5,
  0x00D4,
  0x00B4,
  0x06A9,
  0x0324,
  0x0328,
  0x031D,
  0x0321,
  0x0003,
}

for addr = 0x0400, 0x0413 do
  table.insert(MONITORED_ADDRS, addr)
end

local IN_LEVEL_ADDRS = { 0x0018, 0x001C, 0x00D0 }
local LEVEL_SELECT_ADDRS = { 0x0324, 0x0328 }
local PAUSE_ADDRS = { 0x031D, 0x0321 }
local DEATH_ADDR = 0x0003

local function read_ram_snapshot()
  local out = {}
  for _, addr in ipairs(MONITORED_ADDRS) do
    out[string.format("0x%04X", addr)] = memory.readbyte(addr)
  end
  return out
end

local function is_in_level(ram_snapshot)
  for _, addr in ipairs(IN_LEVEL_ADDRS) do
    local key = string.format("0x%04X", addr)
    if (ram_snapshot[key] or 0) ~= 0 then
      return true
    end
  end
  return false
end

local function is_level_select(ram_snapshot)
  for _, addr in ipairs(LEVEL_SELECT_ADDRS) do
    local key = string.format("0x%04X", addr)
    if (ram_snapshot[key] or 0) ~= 0 then
      return true
    end
  end
  return false
end

local function is_paused(ram_snapshot)
  for _, addr in ipairs(PAUSE_ADDRS) do
    local key = string.format("0x%04X", addr)
    if (ram_snapshot[key] or 0) ~= 0 then
      return true
    end
  end
  return false
end

local function capture_frame()
  local chunks = {}
  local idx = 1
  for y = 0, FRAME_HEIGHT - 1 do
    for x = 0, FRAME_WIDTH - 1 do
      local r, g, b = emu.getscreenpixel(x, y, true)
      chunks[idx] = string.char(r, g, b)
      idx = idx + 1
    end
  end
  return table.concat(chunks)
end

local last_in_level = false
local last_death_flag = false
local last_world_index = 0
local last_level_index = 0
local furthest_world_index = 0
local furthest_level_index = 0

local function wait_frames(count)
  for _ = 1, count do
    emu.frameadvance()
  end
end

local function wait_until(predicate, timeout_frames)
  local frames = timeout_frames or LEVEL_LOAD_TIMEOUT_FRAMES
  for _ = 1, frames do
    if predicate() then
      return true
    end
    emu.frameadvance()
  end
  return false
end

local function load_saved_state(path)
  local target = resolve_project_path(path) or DEFAULT_SAVE_STATE
  if not target then
    return false, "no savestate path configured"
  end

  local probe = io.open(target, "rb")
  if not probe then
    return false, string.format("savestate not found: %s", target)
  end
  probe:close()

  local ok, err = pcall(function()
    local state = savestate.create(target)
    savestate.load(state)
  end)
  if not ok then
    return false, err or "failed to load savestate"
  end
  wait_frames(STATE_SETTLE_FRAMES)
  return true
end

local function set_game_level(world_num, level_num)
  -- Lua save state always loads into an active level; rewrite level selection memory to jump.
  local world_idx = math.max(0, (world_num or 1) - 1)
  local level_idx = math.max(0, (level_num or 1) - 1)
  memory.writebyte(0x00B4, 10 * world_idx + level_idx)
  memory.writebyte(0x0002, 0x06)
end

local function is_level_ready()
  local game_state = memory.readbyte(0x0002) or 0
  local stage_id = memory.readbyte(0x00B4) or 0xFF
  return game_state >= 0x06 and stage_id ~= 0xFF
end

local function restart_level(world_num, level_num, opts)
  local options = opts or {}
  local save_path = options.save_path or DEFAULT_SAVE_STATE

  local loaded, err = load_saved_state(save_path)
  if not loaded then
    return false, ("savestate load failed: %s"):format(tostring(err))
  end

  local level_ready = wait_until(function()
    return is_level_ready()
  end)

  if not level_ready then
    return false, "timeout waiting for level after savestate load"
  end

  set_game_level(world_num, level_num)

  local success = wait_until(function()
    if is_level_ready() then
      return true
    end
    local ram = read_ram_snapshot()
    return is_in_level(ram)
  end)

  if not success then
    return false, "timeout waiting for level load after jump"
  end

  wait_frames(POST_LOAD_IDLE_FRAMES)

  last_in_level = true
  last_death_flag = false
  return true
end

local function build_payload()
  local ram_snapshot = read_ram_snapshot()
  local in_level = is_in_level(ram_snapshot)
  local on_level_select = is_level_select(ram_snapshot)
  local paused = is_paused(ram_snapshot)
  local death_flag = (ram_snapshot[string.format("0x%04X", DEATH_ADDR)] or 0) == 8
  local level_complete_flag = ram_snapshot["0x06A9"] or 0

  local info = {
    fires = ram_snapshot["0x00AB"],
    level_flag = level_complete_flag,
    frame = emu.framecount(),
    in_level = in_level,
    level_select = on_level_select,
    paused = paused,
    world_index = ram_snapshot["0x00D4"] or last_world_index,
    level_index = ram_snapshot["0x00B5"] or last_level_index,
  }

  last_world_index = info.world_index or last_world_index
  last_level_index = info.level_index or last_level_index

  if in_level then
    if last_world_index > furthest_world_index then
      furthest_world_index = last_world_index
      furthest_level_index = last_level_index
    elseif last_world_index == furthest_world_index and last_level_index > furthest_level_index then
      furthest_level_index = last_level_index
    end
  end

  local payload = {
    status = "ok",
    frame = mime.b64(capture_frame()),
    ram = ram_snapshot,
    info = info,
    timestamp = luasocket.gettime(),
  }

  info.level_completed_event = false
  info.death_event = false
  info.furthest_world_index = furthest_world_index
  info.furthest_level_index = furthest_level_index

  if last_in_level and not in_level and on_level_select and level_complete_flag == 0 then
    info.level_completed_event = true
  end

  if death_flag and not last_death_flag then
    info.death_event = true
  end

  last_in_level = in_level
  last_death_flag = death_flag
  return payload
end

local function soft_reset()
  last_in_level = false
  last_death_flag = false
end

local function step(action_name)
  local frames_to_run = math.max(1, math.floor(FRAME_SKIP))
  local button_state = BUTTON_MAP[action_name]

  if button_state then
    joypad.set(1, button_state)
  end

  for _ = 1, frames_to_run do
    emu.frameadvance()
  end

  if button_state then
    joypad.set(1, {})
  end

  return build_payload()
end

local function handle_request(request)
  local cmd = request.cmd

  if cmd == "handshake" then
    return {
      status = "ok",
      server = "fceux-fireice",
      version = "0.1.0",
      port = ACTIVE_PORT,
    }
  elseif cmd == "reset" then
    soft_reset()
    local target_world = request.world or (last_world_index + 1)
    local target_level = request.level or (last_level_index + 1)
    local ok, err = restart_level(target_world, target_level, { save_path = request.save_state })
    if not ok then
      return { status = "error", message = err }
    end
    return build_payload()
  elseif cmd == "restart_level" then
    local target_world = request.world or (last_world_index + 1)
    local target_level = request.level or (last_level_index + 1)
    local ok, err = restart_level(target_world, target_level, { save_path = request.save_state })
    if not ok then
      return { status = "error", message = err }
    end
    local payload = build_payload()
    payload.info = payload.info or {}
    payload.info.level_restart_event = true
    payload.terminated = true
    payload.done = true
    return payload
  elseif cmd == "set_speed" then
    local mode = request.mode or "normal"
    if type(mode) == "string" then
      emu.speedmode(mode)
      return { status = "ok", mode = mode }
    else
      return { status = "error", message = "invalid speed mode" }
    end
  elseif cmd == "set_frame_skip" then
    local skip = tonumber(request.skip)
    if not skip or skip < 1 then
      return { status = "error", message = "frame skip must be >= 1" }
    end
    FRAME_SKIP = math.max(1, math.floor(skip))
    return { status = "ok", frame_skip = FRAME_SKIP }
  elseif cmd == "step" then
    local action = request.action
    if action == "LEVELRESTART" then
      -- Treat as a no-op here; environment should explicitly call restart_level command.
      action = nil
    end
    local payload = step(action)
    local info = payload.info or {}
    local level_completed = info.level_completed_event
    local death_happened = info.death_event or ((payload.ram or {})["0x0003"] == 8)
    payload.terminated = level_completed or death_happened
    payload.done = payload.terminated

    return payload
  elseif cmd == "pause" then
    return { status = "ok", paused = true }
  else
    return { status = "error", message = "unknown command: " .. tostring(cmd) }
  end
end

local ZMQ_DONTWAIT = zmq.DONTWAIT or zmq.NOBLOCK or 1

while true do
  local msg = responder:recv(ZMQ_DONTWAIT)
  if msg then
    local request, pos, err = json.decode(msg, 1, nil)
    if err then
      responder:send(json.encode({ status = "error", message = err }))
    else
      local response = handle_request(request)
      responder:send(json.encode(response))
    end
  else
    emu.frameadvance()
  end
end

if responder.close then
  responder:close()
elseif zmq.close then
  zmq.close(responder)
end

if context.term then
  context:term()
elseif zmq.term then
  zmq.term(context)
elseif zmq.context_term then
  zmq.context_term(context)
end
