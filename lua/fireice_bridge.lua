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

local bind_ok, bind_err
if type(responder.bind) == "function" then
  bind_ok, bind_err = responder:bind("tcp://*:5555")
elseif zmq.bind then
  bind_ok, bind_err = zmq.bind(responder, "tcp://*:5555")
else
  error("ZeroMQ binding cannot bind sockets via known APIs.")
end
assert(bind_ok, bind_err)
print("[fireice] bound to tcp://*:5555")

local FRAME_WIDTH = 256
local FRAME_HEIGHT = 240

local BUTTON_MAP = {
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
  }

  local payload = {
    status = "ok",
    frame = mime.b64(capture_frame()),
    ram = ram_snapshot,
    info = info,
    timestamp = luasocket.gettime(),
  }

  info.level_completed_event = false
  info.death_event = false

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
  emu.softreset()
  for _ = 1, 15 do
    emu.frameadvance()
  end
  last_in_level = false
  last_death_flag = false
end

local function apply_action(action_name)
  local button_state = BUTTON_MAP[action_name] or {}
  joypad.set(1, button_state)
  emu.frameadvance()
  joypad.set(1, {}) -- release buttons
end

local function step(action_name)
  if action_name then
    apply_action(action_name)
  else
    emu.frameadvance()
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
    }
  elseif cmd == "reset" then
    soft_reset()
    return build_payload()
  elseif cmd == "set_speed" then
    local mode = request.mode or "normal"
    if type(mode) == "string" then
      emu.speedmode(mode)
      return { status = "ok", mode = mode }
    else
      return { status = "error", message = "invalid speed mode" }
    end
  elseif cmd == "step" then
    local payload = step(request.action)
    local info = payload.info or {}
    local reset_on_complete = request.reset_on_level_complete
    local reset_on_death = request.reset_on_death

    if info.level_completed_event and reset_on_complete then
      payload.done = true
      payload.terminated = true
    end

    if info.death_event and reset_on_death then
      payload.done = true
      payload.terminated = true
    end

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
