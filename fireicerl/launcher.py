from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence


@dataclass
class FCEUXLaunchConfig:
    """Configuration for launching multiple FCEUX processes."""

    fceux_path: Path
    rom_path: Path
    lua_script: Path
    base_port: int = 5555
    num_workers: int = 1
    port_step: int = 1
    extra_args: Sequence[str] = ()
    working_dir: Optional[Path] = None
    env_overrides: Optional[Dict[str, str]] = None
    launch_delay_s: float = 0.25


@dataclass
class FCEUXProcess:
    """Represents a running FCEUX process."""

    worker_id: int
    port: int
    process: subprocess.Popen

    def terminate(self, timeout: float = 5.0) -> None:
        if self.process.poll() is not None:
            return
        try:
            self.process.terminate()
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=timeout)


class FCEUXProcessManager:
    """Launches and supervises multiple FCEUX emulator instances."""

    def __init__(self, config: FCEUXLaunchConfig) -> None:
        self.config = config
        self.processes: List[FCEUXProcess] = []

    def __enter__(self) -> "FCEUXProcessManager":
        self.start_all()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop_all()

    def start_all(self) -> None:
        """Launch the configured number of FCEUX instances."""
        if self.processes:
            return

        base_env = os.environ.copy()
        if self.config.env_overrides:
            base_env.update(self.config.env_overrides)

        fceux_path = str(self.config.fceux_path)
        rom_path = str(self.config.rom_path)
        lua_script = str(self.config.lua_script)

        for worker in range(self.config.num_workers):
            port = self.config.base_port + worker * self.config.port_step
            env = base_env.copy()
            env["FIREICE_PORT"] = str(port)
            env["FIREICE_PORT_ATTEMPTS"] = "1"
            env["FIREICE_PORT_STEP"] = "1"
            env["FIREICE_INSTANCE_ID"] = str(worker)

            cmd: List[str] = [fceux_path, "--loadlua", lua_script, rom_path]
            if self.config.extra_args:
                # Insert additional arguments after the executable but before ROM.
                cmd = [fceux_path, *self.config.extra_args,"--sound", "0", "--loadlua", lua_script, rom_path]

            proc = subprocess.Popen(
                cmd,
                cwd=str(self.config.working_dir) if self.config.working_dir else None,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.processes.append(FCEUXProcess(worker_id=worker, port=port, process=proc))

            if self.config.launch_delay_s > 0:
                time.sleep(self.config.launch_delay_s)

    def stop_all(self) -> None:
        """Terminate all launched FCEUX processes."""
        for proc in self.processes:
            proc.terminate()
        self.processes.clear()

    def ensure_alive(self) -> None:
        """Raise RuntimeError if any managed process has exited unexpectedly."""
        for proc in self.processes:
            if proc.process.poll() is not None:
                raise RuntimeError(
                    f"FCEUX worker {proc.worker_id} exited with return code {proc.process.returncode}"
                )
