from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import zmq


@dataclass
class FCEUXConfig:
    """Configuration parameters for the FCEUX ZeroMQ bridge."""

    endpoint: str = "tcp://127.0.0.1:5555"
    request_timeout_s: float = 2.0
    handshake_retries: int = 5
    frame_height: int = 240
    frame_width: int = 256
    frame_channels: int = 3
    expect_base64: bool = True
    compression: Optional[str] = None  # reserved for future use
    lua_scripts_dir: str = "lua"
    handshake_message: Dict[str, str] = field(
        default_factory=lambda: {"cmd": "handshake", "client": "fireicerl"}
    )


class FCEUXBridge:
    """Handles socket communication with the Lua script running inside fceux."""

    def __init__(self, config: Optional[FCEUXConfig] = None) -> None:
        self.config = config or FCEUXConfig()
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._poller = zmq.Poller()
        self._poller.register(self._socket, zmq.POLLIN)
        self._socket.connect(self.config.endpoint)
        self._handshake_completed = False

    def close(self) -> None:
        if self._socket is not None:
            self._poller.unregister(self._socket)
            self._socket.close(0)
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None

    def __del__(self) -> None:
        self.close()

    def handshake(self) -> None:
        if self._handshake_completed:
            return

        for attempt in range(1, self.config.handshake_retries + 1):
            try:
                reply = self._request(self.config.handshake_message)
            except TimeoutError:
                time.sleep(0.25 * attempt)
                continue

            if reply.get("status") == "ok":
                self._handshake_completed = True
                return

        raise ConnectionError(
            "Unable to complete handshake with fceux Lua bridge after "
            f"{self.config.handshake_retries} attempts."
        )

    def reset(
        self,
        *,
        world: Optional[int] = None,
        level: Optional[int] = None,
        save_state: Optional[str] = None,
    ) -> Dict:
        """Ask the emulator to reset the environment and return the initial observation payload."""
        self.handshake()
        payload: Dict[str, object] = {"cmd": "reset"}
        if world is not None:
            payload["world"] = int(world)
        if level is not None:
            payload["level"] = int(level)
        if save_state is not None:
            payload["save_state"] = save_state
        return self._request(payload)

    def restart_level(
        self,
        *,
        world: Optional[int] = None,
        level: Optional[int] = None,
        save_state: Optional[str] = None,
    ) -> Dict:
        """Force the current level to restart from a clean savestate."""
        self.handshake()
        payload: Dict[str, object] = {"cmd": "restart_level"}
        if world is not None:
            payload["world"] = int(world)
        if level is not None:
            payload["level"] = int(level)
        if save_state is not None:
            payload["save_state"] = save_state
        return self._request(payload)

    def step(self, action: str, options: Optional[Dict[str, object]] = None) -> Dict:
        """Send an action identifier to the emulator and return the resulting payload."""
        self.handshake()
        payload: Dict[str, object] = {"cmd": "step", "action": action}
        if options:
            payload.update(options)
        return self._request(payload)

    def pause(self) -> Dict:
        """Optional utility for pausing the emulator."""
        self.handshake()
        return self._request({"cmd": "pause"})

    def set_speed_mode(self, mode: str) -> Dict:
        """Configure the emulator speed (e.g. normal, turbo, nothrottle)."""
        self.handshake()
        return self._request({"cmd": "set_speed", "mode": mode})

    def set_frame_skip(self, skip: int) -> Dict:
        """Configure how many emulator frames to advance per environment step."""
        self.handshake()
        return self._request({"cmd": "set_frame_skip", "skip": int(skip)})

    def _request(self, payload: Dict) -> Dict:
        if self._socket is None:
            raise RuntimeError("Attempted to use a closed FCEUXBridge.")

        self._socket.send_json(payload)

        socks = dict(
            self._poller.poll(timeout=int(self.config.request_timeout_s * 1000))
        )
        if socks.get(self._socket) != zmq.POLLIN:
            raise TimeoutError("Timed out waiting for response from fceux Lua bridge.")

        reply_raw = self._socket.recv()
        try:
            reply = json.loads(reply_raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to decode Lua reply: {reply_raw!r}") from exc

        if reply.get("status", "ok") != "ok":
            raise RuntimeError(f"Lua bridge reported error: {reply}")

        return reply

    def decode_frame(self, payload: Dict) -> np.ndarray:
        """Convert a payload dictionary returned by Lua into an RGB frame array."""
        frame_blob = payload.get("frame")
        if frame_blob is None:
            raise KeyError("Missing 'frame' field in bridge payload.")

        if self.config.expect_base64:
            raw_bytes = base64.b64decode(frame_blob)
        else:
            raw_bytes = bytes(frame_blob)

        frame = np.frombuffer(raw_bytes, dtype=np.uint8)
        expected_size = (
            self.config.frame_height
            * self.config.frame_width
            * self.config.frame_channels
        )
        if frame.size != expected_size:
            raise ValueError(
                f"Unexpected frame size {frame.size}, expected {expected_size} bytes."
            )
        return frame.reshape(
            self.config.frame_height, self.config.frame_width, self.config.frame_channels
        )

    @staticmethod
    def decode_ram(payload: Dict) -> Dict[int, int]:
        """Extract observed RAM addresses from payload."""
        ram_snapshot = payload.get("ram", {})
        return {int(addr, 16): value for addr, value in ram_snapshot.items()}

    @staticmethod
    def decode_metadata(payload: Dict) -> Dict:
        """Extract auxiliary metadata from payload."""
        info = payload.get("info", {})
        info["timestamp"] = payload.get("timestamp")
        return info
