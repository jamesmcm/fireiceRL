## Fire 'n Ice RL Trainer

Train a Deep Q-Network (DQN) or Proximal Policy Optimisation (PPO) agent to master *Fire 'n Ice* using an FCEUX emulator bridge, stacked CNN observations, and memory-based reward shaping. DQN with an experience replay buffer is now the default algorithm (`--algorithm dqn`), while the previous PPO pipeline remains available via `--algorithm ppo`. The Lua bridge restores a savestate and force-loads the requested world/level on every reset so the agent never has to traverse title or level-select menus.

![Gameplay Screenshot](screenshot.png)

At the moment the model fails to reliably pass level 1-5 due to the extremely sparse reward here and long, specific series of inputs required to succeed:

![Level 1-5](level1-5.png)

A more sophisticated reinforcement learning algorithm / world model may be required here.


### Prerequisites
- [uv](https://docs.astral.sh/uv/) for Python dependency management
- FCEUX ≥ 2.6.6 rebuilt against **system Lua 5.1** (Qt front end) - e.g. build from source on Arch Linux
- LuaRocks-installed modules for Lua 5.1: `lua-zmq`, `luasocket`, `dkjson`
- Fire 'n Ice (USA) ROM — SHA256 `197833f0ac0e87a824f008bdaf5e429b9688a5dbfee0b65cb6b731e1a3ee77e5`

### Repository Layout
- `fireicerl/`
  - `bridge.py` – ZeroMQ REQ client for Lua bridge
  - `environment.py` – Gymnasium wrapper with frame stacking & speed control
  - `reward.py` – shaped rewards (fires, completion, restarts, death; menu signals logged for diagnostics)
  - `dqn.py` – replay buffer, epsilon-greedy action selection, and DQN training loop with target network syncs
  - `ppo.py` / `logging.py` – PPO training loop, metrics, stagnation resets
- `lua/fireice_bridge.lua` – FCEUX-side bridge (frame capture, events, direct level injection)
- `main.py` – CLI entry point for training/resume
- `PLAN.md`, `notes.txt` – memory map notes and roadmap
- `roms/1-1-nounlock.sav` – default savestate used when jumping straight into a level

### Setup & Emulator Launch
```bash
uv sync  # install Python deps

# The trainer now spawns FCEUX workers automatically.
# Provide explicit paths if the defaults do not match your system.
```
The training CLI launches each emulator with the Lua bridge attached. Supply `--fceux-path`, `--rom-path`, and `--lua-script` if they differ from the defaults. To manage FCEUX manually, run the usual command and add `--no-launch-fceux` when starting training:
```bash
fceux --loadlua lua/fireice_bridge.lua "roms/Fire 'n Ice (USA).nes"
```
Wait for “bound to tcp://*:<port>” in the Lua console, then optionally disable audio/video sync for max speed. Make sure the savestate pointed to by `--save-state-path` (default `roms/1-1-nounlock.sav`) exists; the bridge restores it and rewrites the world/level bytes so training always resumes directly inside a stage.

### Training the Agent
The default configuration launches a DQN agent:
```bash
uv run python main.py train \
  --total-timesteps 2_000_000 \
  --replay-capacity 300000 \
  --dqn-log-interval 10000 \
  --log-dir logs/dqn-run1 \
  --checkpoint-dir checkpoints/dqn-run1 \
  --speed-mode nothrottle
```
To warm-start from a previous run, pass `--load-checkpoint path/to/checkpoint.pt`. Include `--load-optimizer` if you also want to restore optimizer and RNG state (supported by PPO; DQN always refreshes its replay buffer on launch).

Switch back to PPO by passing `--algorithm ppo` along with the usual rollout arguments:
```bash
uv run python main.py train \
  --algorithm ppo \
  --total-timesteps 8_000_000 \
  --rollout-steps 256 \
  --log-dir logs/ppo-run14 \
  --checkpoint-dir checkpoints/ppo-run14 \
  --speed-mode nothrottle
```
To exploit parallel environments, choose a base port and spawn 26 FCEUX workers (adjust as needed):
```bash
uv run python main.py train \
  --num-workers 26 \
  --base-port 6000 \
  --fceux-path /usr/bin/fceux \
  --rom-path "roms/Fire 'n Ice (USA).nes" \
  --lua-script lua/fireice_bridge.lua \
  --total-timesteps 12000000
```
Useful flags:
- `--speed-mode {normal|turbo|nothrottle}` – mirrors `emu.speedmode`
- `--checkpoint-interval`, `--checkpoint-dir`, `--log-dir`, `--load-checkpoint`, `--load-optimizer`
- `--replay-capacity`, `--learning-starts`, `--dqn-batch-size`, `--dqn-train-frequency`, `--dqn-target-update`, `--epsilon-*` – tune DQN replay and exploration behaviour
- `--num-workers` spawns parallel environments for both DQN (shared replay buffer) and PPO (batched rollouts)
- `--initial-world`, `--initial-level`, `--save-state-path` – configure which stage the loader injects after every reset (no menu navigation required)
- `--num-workers`, `--base-port` – control how many emulator sessions are launched and which TCP ports they target (ports increment by one per worker)
- `--fceux-path`, `--rom-path`, `--lua-script` – override the auto-launch command paths
- `--no-launch-fceux`, `--fceux-extra-arg` – opt out of auto-launching or append flags (e.g. `--fceux-extra-arg --nogui`)

`metrics.csv` records losses, TD errors (for DQN), returns, and `stagnation_reset` events; `reward_components.csv` breaks down components (`fire`, `completion`, `restart`, `death`, `pause`, `menu_entry` — this should stay near zero), etc.

### Reward & Event Signals
- Fires remaining `$00AB` (positive on decrease, bonus at zero)
- Level completion `$06A9 == 0` (single reward per *new* completion per run)
- World progress bitmasks `$0400–$0413`
- Level-select diagnostics `$0324/$0328` (these should stay zero during training; any spike means the loader slipped back to a menu)
- Pause flags `$031D/$0321` and death animation `$0003 == 8` (penalties)
- In-level flags `$0018/$001C/$00D0` sanity-check that the loader really spawned us into gameplay
- World/level index: `$00B4` is world index*10 + level index (note index starts from 0), `$00B5` is level index (extracted from `$00B4`),  `$00B6` is world index (extracted from `$00B4`), 
- Colour palette: `$00D4` (corresponds to world index)
- Game loop function setting: `$0002` - a value of `6` means to reset /
  load a level.

Tune magnitudes in `fireicerl/reward.py` as needed; metrics automatically export any new components.

### Tips & Troubleshooting
- `--nogui` disables Lua so cannot be used for headless runs here.
- Ensure the entire port range `base_port ... base_port + num_workers - 1` is free before launching; restart any failed worker if the bridge cannot bind.
- If you ever land on the title or level-select screen during training, double-check the savestate path and loader configuration—menus should never appear under normal operation.
- For speed, combine `--speed-mode nothrottle`, disabled sync in FCEUX, and higher `--frame-skip`.
- If reward totals stay positive after finishing a level, confirm `reward_components.csv` shows only one `completion` entry—otherwise the level may not have exited properly.

### TODOs

- Improve rewards for better convergence (we have sparse rewards at the
  moment) - e.g. see level 1-5.
- Is there a way of building fceux so it will run without graphics but
  still run Lua? Can we still read the VRAM in this case?
- Is there a faster emulator where we can directly send input and read
  VRAM without the Lua bridge?
