# Repository Guidelines

## Project Structure & Module Organization
- `fireicerl/`: Core Python package. Notable modules include `bridge.py` (ZeroMQ bridge to FCEUX), `environment.py` (Gymnasium wrapper that drives direct level loads), `reward.py` (reward shaping), `launcher.py` (FCEUX process manager), and `ppo.py` (training loop).
- `lua/fireice_bridge.lua`: Lua script loaded inside FCEUX to stream frames, RAM, events, and force levels to load directly without visiting menus.
- `main.py`: CLI entry point for training (`uv run python main.py …`).
- `logs/`, `checkpoints/`: Created during training for metrics and model snapshots (configurable via CLI).
- `roms/1-1-nounlock.sav`: Baseline savestate used by the loader when jumping straight into a level.
- `PLAN.md`, `notes.txt`: Design notes and memory map references.

## Build, Test, and Development Commands
- Install/update dependencies: `uv sync`
- Lint/format (Python): `uv run ruff check .` (add if linting configured)
- Byte-compile check: `uv run python -m compileall fireicerl lua/fireice_bridge.lua main.py`
- Launch training: `uv run python main.py train --help` for options; common flags include `--speed-mode nothrottle`, `--num-workers 26`, `--base-port 6000`, `--log-dir logs/run1`
- Auto-launch uses `fceux --loadlua lua/fireice_bridge.lua ROM`; add `--no-launch-fceux` if you prefer to bring up emulator sessions yourself.
- Run Lua bridge manually: `fceux --loadlua lua/fireice_bridge.lua …/Fire\ 'n\ Ice\ (USA).nes`

## Coding Style & Naming Conventions
- Python: follow PEP 8, 4-space indentation, snake_case for functions/variables, PascalCase for classes.
- Lua: maintain existing style (two-space indentation, uppercase snake_case for constants, descriptive locals).
- Keep comments concise; prefer docstrings for public functions.
- Use Ruff/black-compatible formatting when touching Python files.

## Testing Guidelines
- No formal test suite yet; use `compileall` smoke test before committing.
- Validate training runs by examining `logs/*/metrics.csv` and `reward_components.csv`.
- When adding reward logic, include unit-style assertions or temporary scripts to verify new components (remove before commit).
- During manual runs, confirm `reset`/`restart_level` drop the agent directly into the requested stage; menu screens indicate a regression in the loader or savestate.
- When scaling workers, ensure the full port range `base_port..base_port + num_workers - 1` is free and that CNN snapshot directories are per-worker to avoid collisions.

## Commit & Pull Request Guidelines
- Commit messages: imperative mood (e.g., “Add stagnation reset logic”), scope prefix optional.
- Group related changes; avoid mixing Lua/Python modifications unless logically connected.
- PRs should describe feature/bugfix, list key commands run, attach logs or screenshots for training output when relevant.
- Link to design notes (`PLAN.md`) or issues when expanding rewards, bridge events, or CLI flags.
