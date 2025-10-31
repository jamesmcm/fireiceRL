from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import random

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
    frame_skip: int = 1  # how many emulator frames per environment step
    grayscale: bool = True
    speed_mode: Optional[str] = "turbo"  # normal, turbo, nothrottle
    initial_world: int = 1
    initial_level: int = 1
    max_world: int = 8
    levels_per_world: int = 10
    save_state_path: str = "roms/1-1-nounlock.sav"
    cnn_snapshot_dir: Optional[str] = None
    cnn_snapshot_interval: int = 0
    stagnation_no_positive_limit: int = 50
    stagnation_no_completion_limit: int = 100
    frame_stagnation_limit: int = 10000
    noop_penalty: float = -0.01
    level_stagnation_limit_frames: int = 21600
    level_stagnation_penalty: float = -5.0
    level_stagnation_skip_probability: float = 0.5


class FireIceEnv(gym.Env):
    """Gymnasium-compatible environment wrapper around the fceux Lua bridge."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    ACTION_NAMES = ["NOOP", "LEFT", "RIGHT", "A"]

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
        self._steps_since_snapshot = 0
        self._snapshot_index = 0
        self._rng = random.Random()

        self.current_world = self._clamp_world(self.config.initial_world)
        self.current_level = self._clamp_level(self.config.initial_level)
        self.furthest_world = self.current_world
        self.furthest_level = self.current_level

        self._pending_level_advance = False
        self._pending_full_restart = False
        self._episodes_since_positive = 0
        self._episodes_since_completion = 0
        self._current_episode_reward = 0.0
        self._frames_since_positive = 0
        self._last_frame_id: Optional[int] = None
        self._frames_on_level = 0

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
        self.reward_tracker.reset()
        self._current_episode_reward = 0.0

        if self.config.speed_mode and not self._speed_configured:
            try:
                self.bridge.set_speed_mode(self.config.speed_mode)
            except Exception:
                pass
            else:
                self._speed_configured = True

        if self._pending_full_restart:
            self.current_world = self._clamp_world(self.config.initial_world)
            self.current_level = self._clamp_level(self.config.initial_level)
            self._pending_full_restart = False

        if self._pending_level_advance:
            self._advance_level()
            self._pending_level_advance = False

        payload = self.bridge.reset(
            world=self.current_world,
            level=self.current_level,
            save_state=self.config.save_state_path,
        )
        observation, info, _ = self._consume_payload(
            payload, action_name="RESET", reset_stack=True
        )
        self._steps_since_snapshot += 1
        self._maybe_save_cnn_input(observation, info, context="reset")
        self._frames_since_positive = 0
        self._last_frame_id = info.get("frame")
        self._frames_on_level = 0
        info["current_world"] = self.current_world
        info["current_level"] = self.current_level
        info["furthest_world"] = self.furthest_world
        info["furthest_level"] = self.furthest_level
        info.setdefault("full_game_restart", False)
        return observation, info

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action_name = self.ACTION_NAMES[int(action_idx)]
        total_reward = 0.0
        reward_components: Dict[str, float] = {}
        observation: Optional[np.ndarray] = None
        info: Dict = {}
        terminated = False
        truncated = False

        # Agent-triggered level restarts are intentionally disabled for now.
        # if action_name == "LEVELRESTART":
        #     ...

        steps = max(1, int(self.config.frame_skip))
        last_payload: Dict = {}
        last_info: Dict = {}
        for _ in range(steps):
            last_payload = self.bridge.step(action_name)
            observation, step_info, reward = self._consume_payload(
                last_payload, action_name
            )
            last_info = step_info
            total_reward += reward
            for key, value in step_info.get("reward_components", {}).items():
                reward_components[key] = reward_components.get(key, 0.0) + float(value)

            self._register_frame_progress(step_info, reward)

            if last_payload.get("done"):
                break
        if last_info:
            info = last_info
        else:
            info = {}
        terminated = bool(last_payload.get("terminated", last_payload.get("done", False)))
        truncated = bool(last_payload.get("truncated", False))

        if action_name == "NOOP":
            noop_penalty = float(self.config.noop_penalty)
            total_reward += noop_penalty
            reward_components["noop"] = reward_components.get("noop", 0.0) + noop_penalty

        if (
            not terminated
            and not truncated
            and self.config.frame_stagnation_limit > 0
            and self._frames_since_positive >= self.config.frame_stagnation_limit
        ):
            observation, info, restart_reward = self._trigger_frame_stagnation_restart()
            total_reward += restart_reward
            restart_components = info.get("reward_components", {})
            for key, value in restart_components.items():
                reward_components[key] = reward_components.get(key, 0.0) + float(value)
            terminated = True
            truncated = False

        if (
            not terminated
            and not truncated
            and self.config.level_stagnation_limit_frames > 0
            and self._frames_on_level >= self.config.level_stagnation_limit_frames
        ):
            observation, info, level_reward = self._handle_level_stagnation()
            total_reward += level_reward
            level_components = info.get("reward_components", {})
            for key, value in level_components.items():
                reward_components[key] = reward_components.get(key, 0.0) + float(value)
            terminated = True
            truncated = False

        info["reward_components"] = reward_components
        self._current_episode_reward += total_reward
        self._steps_since_snapshot += 1
        self._maybe_save_cnn_input(observation, info, context="step")

        if terminated or truncated:
            self._handle_episode_end(info, self._current_episode_reward)
        else:
            info.setdefault("full_game_restart", False)

        info["current_world"] = self.current_world
        info["current_level"] = self.current_level
        info["furthest_world"] = self.furthest_world
        info["furthest_level"] = self.furthest_level

        return observation, total_reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        return self._last_raw_frame

    def close(self) -> None:
        self.bridge.close()

    def _consume_payload(
        self,
        payload: Dict,
        action_name: str,
        *,
        reset_stack: bool = False,
        extra_info: Optional[Dict[str, object]] = None,
    ) -> Tuple[np.ndarray, Dict, float]:
        frame = self.bridge.decode_frame(payload)
        processed = self._preprocess_frame(frame)
        if reset_stack:
            self._frame_buffer.clear()
            for _ in range(self.config.stack_size):
                self._push_frame(processed)
        else:
            self._push_frame(processed)
        self._last_raw_frame = frame

        ram_snapshot = self.bridge.decode_ram(payload)
        info = self.bridge.decode_metadata(payload)
        if extra_info:
            info.update(extra_info)
        info["ram_snapshot"] = ram_snapshot
        info["action_name"] = action_name

        world_index = info.pop("world_index", None)
        level_index = info.pop("level_index", None)
        in_level = bool(info.get("in_level"))

        if in_level and world_index is not None:
            world_value = int(world_index) + 1
        else:
            world_value = self.current_world

        if in_level and level_index is not None:
            level_value = int(level_index) + 1
        else:
            level_value = self.current_level

        world_value = self._clamp_world(world_value)
        level_value = self._clamp_level(level_value)
        info["world"] = world_value
        info["level"] = level_value

        if world_value != self.current_world or level_value != self.current_level:
            self._frames_on_level = 0
        self.current_world = world_value
        self.current_level = level_value

        self._update_progress(world_value, level_value)

        reward, reward_components = self.reward_tracker.compute_reward(
            ram_snapshot, info, action_name
        )
        info["reward_components"] = reward_components
        observation = self._stack_frames()
        return observation, info, float(reward)

    def _update_progress(self, world: int, level: int) -> None:
        world = self._clamp_world(world)
        level = self._clamp_level(level)
        if (
            world > self.furthest_world
            or (world == self.furthest_world and level > self.furthest_level)
        ):
            self.furthest_world = world
            self.furthest_level = level

    def _advance_level(self) -> None:
        if (
            self.current_world >= self.config.max_world
            and self.current_level >= self.config.levels_per_world
        ):
            self.current_world = self.config.max_world
            self.current_level = self.config.levels_per_world
            return

        if self.current_level < self.config.levels_per_world:
            self.current_level += 1
        else:
            self.current_level = 1
            if self.current_world < self.config.max_world:
                self.current_world += 1
        self.current_world = self._clamp_world(self.current_world)
        self.current_level = self._clamp_level(self.current_level)
        self._frames_on_level = 0

    def _handle_episode_end(self, info: Dict, episode_reward: float) -> None:
        completed = bool(info.get("level_completed_event"))
        positive = episode_reward > 0.0

        if positive:
            self._episodes_since_positive = 0
        else:
            self._episodes_since_positive += 1

        if completed:
            self._episodes_since_completion = 0
        else:
            self._episodes_since_completion += 1

        should_restart = (
            self.config.stagnation_no_positive_limit > 0
            and self._episodes_since_positive
            >= self.config.stagnation_no_positive_limit
        ) or (
            self.config.stagnation_no_completion_limit > 0
            and self._episodes_since_completion
            >= self.config.stagnation_no_completion_limit
        )

        if should_restart:
            self._pending_full_restart = True
            self._episodes_since_positive = 0
            self._episodes_since_completion = 0
            self.current_world = self._clamp_world(self.config.initial_world)
            self.current_level = self._clamp_level(self.config.initial_level)
            self._pending_level_advance = False
            info["full_game_restart"] = True
        else:
            info["full_game_restart"] = False

        if completed:
            self._pending_level_advance = True

        self._current_episode_reward = 0.0
        self.reward_tracker.reset()
        self._frames_since_positive = 0
        self._frames_on_level = 0
        if info.get("level_stagnation_reset"):
            self._episodes_since_positive = 0
            self._episodes_since_completion = 0

    def _maybe_save_cnn_input(
        self, observation: Optional[np.ndarray], info: Dict, *, context: str
    ) -> None:
        if (
            observation is None
            or not self.config.cnn_snapshot_dir
            or self.config.cnn_snapshot_interval <= 0
        ):
            return

        if self._steps_since_snapshot % self.config.cnn_snapshot_interval != 0:
            return

        snapshot_dir = Path(self.config.cnn_snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        if self.config.grayscale:
            tiles = [observation[i] for i in range(observation.shape[0])]
            grid = np.concatenate(tiles, axis=1)
        else:
            grid = observation.transpose(1, 2, 0)

        world = info.get("world", self.current_world)
        level = info.get("level", self.current_level)
        filename = snapshot_dir / f"cnn_{self._snapshot_index:06d}_w{world}_l{level}_{context}.png"
        cv2.imwrite(str(filename), np.ascontiguousarray(grid))
        self._snapshot_index += 1

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

    def _clamp_world(self, world: int) -> int:
        return max(1, min(int(world), self.config.max_world))

    def _clamp_level(self, level: int) -> int:
        return max(1, min(int(level), self.config.levels_per_world))

    def _register_frame_progress(self, info: Dict, reward: float) -> None:
        frame_value = info.get("frame")
        frames_advanced = 0
        if frame_value is not None:
            if self._last_frame_id is not None:
                delta = frame_value - self._last_frame_id
                if delta >= 0:
                    frames_advanced = delta
                else:
                    # Frame counter wrapped; treat as restart.
                    frames_advanced = frame_value
            self._last_frame_id = frame_value

        if not info.get("in_level"):
            return

        if reward > 0:
            self._frames_since_positive = 0
        else:
            self._frames_since_positive += frames_advanced
        self._frames_on_level += frames_advanced

    def _trigger_frame_stagnation_restart(self) -> Tuple[np.ndarray, Dict, float]:
        self._frame_buffer.clear()
        payload = self.bridge.restart_level(
            world=self.current_world,
            level=self.current_level,
            save_state=self.config.save_state_path,
        )
        observation, info, reward_value = self._consume_payload(
            payload,
            action_name="FRAME_STAGNATION_RESET",
            reset_stack=True,
            extra_info={"frame_stagnation_reset": True},
        )
        info.setdefault("frame_stagnation_reset", True)
        info.setdefault("full_game_restart", False)
        self._frames_since_positive = 0
        self._last_frame_id = info.get("frame")
        self._frames_on_level = 0
        return observation, info, float(reward_value)

    def _handle_level_stagnation(self) -> Tuple[np.ndarray, Dict, float]:
        self._frame_buffer.clear()
        skip_probability = getattr(self.config, "level_stagnation_skip_probability", 0.5)
        can_skip_forward = not (
            self.current_world >= self.config.max_world
            and self.current_level >= self.config.levels_per_world
        )
        skip_level = can_skip_forward and self._rng.random() < skip_probability

        if skip_level:
            self._advance_level()
            target_world = self.current_world
            target_level = self.current_level
            info_flags: Dict[str, object] = {
                "level_stagnation_reset": True,
                "full_game_restart": False,
                "level_stagnation_skip": True,
            }
        else:
            target_world = self._clamp_world(self.config.initial_world)
            target_level = self._clamp_level(self.config.initial_level)
            self.current_world = target_world
            self.current_level = target_level
            info_flags = {
                "level_stagnation_reset": True,
                "full_game_restart": True,
                "level_stagnation_skip": False,
            }

        penalty = float(self.config.level_stagnation_penalty)
        self.reward_tracker.reset()
        self._frames_since_positive = 0
        self._frames_on_level = 0
        self._episodes_since_positive = 0
        self._episodes_since_completion = 0

        payload = self.bridge.restart_level(
            world=target_world,
            level=target_level,
            save_state=self.config.save_state_path,
        )
        observation, info, reward_value = self._consume_payload(
            payload,
            action_name="LEVEL_STAGNATION_RESET",
            reset_stack=True,
            extra_info=info_flags,
        )
        info.setdefault("reward_components", {})
        info["reward_components"]["level_stagnation"] = (
            info["reward_components"].get("level_stagnation", 0.0) + penalty
        )
        self._last_frame_id = info.get("frame")
        total_reward = float(reward_value) + penalty
        return observation, info, total_reward
