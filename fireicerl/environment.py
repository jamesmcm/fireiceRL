from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np

from .bridge import FCEUXBridge, FCEUXConfig
from .reward import RewardConfig, RewardTracker


@dataclass
class FireIceEnvConfig:
    """Environment configuration for Fire 'n Ice PPO training."""

    observation_height: int = 84
    observation_width: int = 84
    stack_size: int = 4
    bridge_config: FCEUXConfig = field(default_factory=FCEUXConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    frame_skip: int = 4  # how many emulator frames per environment step
    grayscale: bool = True
    speed_mode: Optional[str] = None  # normal, turbo, nothrottle
    reset_on_level_complete: bool = False
    reset_on_death: bool = False


class FireIceEnv(gym.Env):
    """Gymnasium-compatible environment wrapper around the fceux Lua bridge."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    ACTION_NAMES = ["LEFT", "RIGHT", "A", "B", "START"]

    def __init__(
        self,
        bridge: Optional[FCEUXBridge] = None,
        config: Optional[FireIceEnvConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config or FireIceEnvConfig()
        self.bridge = bridge or FCEUXBridge(self.config.bridge_config)
        self.reward_tracker = RewardTracker(self.config.reward_config)

        self._frame_buffer: Deque[np.ndarray] = deque(maxlen=self.config.stack_size)
        self._last_raw_frame: Optional[np.ndarray] = None
        self._speed_configured = False

        obs_shape = (
            self.config.stack_size,
            self.config.observation_height,
            self.config.observation_width,
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(len(self.ACTION_NAMES))

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._frame_buffer.clear()
        self.reward_tracker = RewardTracker(self.config.reward_config)

        if self.config.speed_mode and not self._speed_configured:
            try:
                self.bridge.set_speed_mode(self.config.speed_mode)
            except Exception:
                pass
            else:
                self._speed_configured = True

        payload = self.bridge.reset()
        frame = self.bridge.decode_frame(payload)
        ram_snapshot = self.bridge.decode_ram(payload)
        info = self.bridge.decode_metadata(payload)

        processed = self._preprocess_frame(frame)
        for _ in range(self.config.stack_size):
            self._push_frame(processed)

        info["ram_snapshot"] = ram_snapshot
        info["paused"] = info.get("paused", False)
        info["level_completed_event"] = info.get("level_completed_event", False)
        info["death_event"] = info.get("death_event", False)
        self._last_raw_frame = frame
        return self._stack_frames(), info

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action_name = self.ACTION_NAMES[int(action_idx)]
        total_reward = 0.0
        info: Dict = {}
        payload = {}

        step_options = {
            "reset_on_level_complete": self.config.reset_on_level_complete,
            "reset_on_death": self.config.reset_on_death,
        }

        for _ in range(max(1, self.config.frame_skip)):
            payload = self.bridge.step(action_name, step_options)
            frame = self.bridge.decode_frame(payload)
            ram_snapshot = self.bridge.decode_ram(payload)
            reward, reward_components = self.reward_tracker.compute_reward(ram_snapshot)
            total_reward += reward

            processed = self._preprocess_frame(frame)
            self._push_frame(processed)
            self._last_raw_frame = frame

            info = self.bridge.decode_metadata(payload)
            info.update(
                {
                    "action_name": action_name,
                    "reward_components": reward_components,
                    "ram_snapshot": ram_snapshot,
                    "paused": info.get("paused", False),
                    "level_completed_event": info.get("level_completed_event", False),
                    "death_event": info.get("death_event", False),
                }
            )

            if payload.get("done"):
                break

        observation = self._stack_frames()
        terminated = bool(payload.get("terminated", payload.get("done", False)))
        truncated = bool(payload.get("truncated", False))
        return observation, total_reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        return self._last_raw_frame

    def close(self) -> None:
        self.bridge.close()

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.config.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(
            frame,
            (self.config.observation_width, self.config.observation_height),
            interpolation=cv2.INTER_AREA,
        )
        if self.config.grayscale:
            return resized.astype(np.uint8)
        return resized.transpose(2, 0, 1).astype(np.uint8)

    def _push_frame(self, frame: np.ndarray) -> None:
        if self.config.grayscale:
            frame = frame[np.newaxis, ...]
        self._frame_buffer.appendleft(frame)

    def _stack_frames(self) -> np.ndarray:
        if len(self._frame_buffer) < self.config.stack_size:
            missing = self.config.stack_size - len(self._frame_buffer)
            filler = self._frame_buffer[0] if self._frame_buffer else np.zeros(
                (1, self.config.observation_height, self.config.observation_width),
                dtype=np.uint8,
            )
            for _ in range(missing):
                self._frame_buffer.append(filler.copy())
        stacked = np.concatenate(list(self._frame_buffer), axis=0)
        return stacked.astype(np.uint8)
