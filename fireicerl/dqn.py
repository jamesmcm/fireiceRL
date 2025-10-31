from __future__ import annotations

import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .environment import FireIceEnv
from .logging import MetricsLogger
from .models import QNetwork


@dataclass
class DQNConfig:
    gamma: float = 0.99
    learning_rate: float = 1e-4
    buffer_capacity: int = 200_000
    batch_size: int = 64
    learning_starts: int = 20_000
    train_frequency: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 4_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 500_000
    epsilon_decay_start: int = 0
    max_grad_norm: float = 10.0
    huber_delta: float = 1.0
    double_dqn: bool = True
    log_interval: int = 5_000
    total_timesteps: int = 1_000_000
    checkpoint_interval: int = 50_000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ExperienceReplayBuffer:
    """Fixed-size replay buffer storing transitions as uint8 for observations."""

    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, ...],
        *,
        seed: Optional[int] = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("Replay buffer capacity must be positive.")
        self.capacity = int(capacity)
        self.obs_shape = obs_shape
        self._obs = np.zeros((self.capacity, *obs_shape), dtype=np.uint8)
        self._next_obs = np.zeros((self.capacity, *obs_shape), dtype=np.uint8)
        self._actions = np.zeros(self.capacity, dtype=np.int64)
        self._rewards = np.zeros(self.capacity, dtype=np.float32)
        self._dones = np.zeros(self.capacity, dtype=np.bool_)
        self._position = 0
        self._full = False
        self._rng = np.random.default_rng(seed)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        index = self._position
        self._obs[index] = np.asarray(obs, dtype=np.uint8)
        self._next_obs[index] = np.asarray(next_obs, dtype=np.uint8)
        self._actions[index] = int(action)
        self._rewards[index] = float(reward)
        self._dones[index] = bool(done)

        self._position = (self._position + 1) % self.capacity
        if self._position == 0:
            self._full = True

    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        size = len(self)
        if size < batch_size:
            raise ValueError("Not enough samples to draw a batch.")

        indices = self._rng.choice(size, size=batch_size, replace=False)
        obs = torch.as_tensor(self._obs[indices], dtype=torch.float32, device=device) / 255.0
        next_obs = (
            torch.as_tensor(self._next_obs[indices], dtype=torch.float32, device=device)
            / 255.0
        )
        actions = torch.as_tensor(self._actions[indices], dtype=torch.long, device=device)
        rewards = torch.as_tensor(self._rewards[indices], dtype=torch.float32, device=device)
        dones = torch.as_tensor(self._dones[indices], dtype=torch.float32, device=device)
        return obs, actions, rewards, next_obs, dones

    def __len__(self) -> int:
        return self.capacity if self._full else self._position


class DQNTrainer:
    """Off-policy Deep Q-learning trainer with replay buffer and optional multi-env support."""

    def __init__(
        self,
        envs: Sequence[FireIceEnv] | FireIceEnv,
        config: Optional[DQNConfig] = None,
        *,
        log_dir: Optional[str | Path] = None,
        checkpoint_dir: Optional[str | Path] = None,
        checkpoint_interval: Optional[int] = None,
    ) -> None:
        if isinstance(envs, FireIceEnv):
            self.envs = [envs]
        else:
            self.envs = list(envs)
        if not self.envs:
            raise ValueError("At least one environment must be provided to DQNTrainer.")

        self.config = config or DQNConfig()
        self.num_envs = len(self.envs)
        sample_env = self.envs[0]
        self.obs_shape = sample_env.observation_space.shape
        self.num_actions = sample_env.action_space.n
        self.device = torch.device(self.config.device)

        in_channels = self.obs_shape[0]
        self.q_network = QNetwork(in_channels, self.num_actions).to(self.device)
        self.target_network = QNetwork(in_channels, self.num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.learning_rate)

        self.buffer = ExperienceReplayBuffer(self.config.buffer_capacity, self.obs_shape)

        self.logger = MetricsLogger(log_dir) if log_dir else None
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        interval = (
            checkpoint_interval
            if checkpoint_interval is not None
            else self.config.checkpoint_interval
        )
        self.checkpoint_interval = max(0, int(interval)) if interval else 0

        self.global_step = 0
        self.total_updates = 0
        self._last_log_step = 0
        self._log_index = 0
        self.training_start = time.time()
        self._rng = random.Random()

        self._episode_returns_window: Deque[float] = deque(maxlen=100)
        self._interval_total_reward = 0.0
        self._interval_loss_sum = 0.0
        self._interval_loss_count = 0
        self._interval_episode_returns: List[float] = []
        self._interval_episode_lengths: List[int] = []
        self._interval_start_time = time.time()
        self._interval_reward_sums: Dict[str, float] = defaultdict(float)
        self._interval_reward_counts: Dict[str, int] = defaultdict(int)
        self._interval_full_restarts = 0
        self._current_worlds = [1 for _ in range(self.num_envs)]
        self._current_levels = [1 for _ in range(self.num_envs)]
        self._furthest_worlds = [1 for _ in range(self.num_envs)]
        self._furthest_levels = [1 for _ in range(self.num_envs)]
        self._interval_epsilon_sum = 0.0
        self._interval_epsilon_count = 0
        self._last_snapshot_step = -1
        self._last_latest_step = -1

    def train(self) -> None:
        observations: List[np.ndarray] = []
        for env_idx, env in enumerate(self.envs):
            obs, info = env.reset()
            observations.append(obs)
            self._update_env_progress(env_idx, info)

        episode_returns = [0.0 for _ in range(self.num_envs)]
        episode_lengths = [0 for _ in range(self.num_envs)]
        steps_since_update = 0

        while self.global_step < self.config.total_timesteps:
            greedy_actions = self._compute_greedy_actions(observations)

            for env_idx, env in enumerate(self.envs):
                epsilon = self._compute_epsilon(self.global_step)
                action = self._epsilon_greedy_action(greedy_actions[env_idx], epsilon)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                self.buffer.add(observations[env_idx], action, reward, next_obs, done)
                observations[env_idx] = next_obs

                episode_returns[env_idx] += float(reward)
                episode_lengths[env_idx] += 1
                self.global_step += 1
                steps_since_update += 1
                self._interval_total_reward += float(reward)
                self._interval_epsilon_sum += epsilon
                self._interval_epsilon_count += 1

                reward_components = info.get("reward_components", {})
                for key, value in reward_components.items():
                    self._interval_reward_sums[key] += float(value)
                    self._interval_reward_counts[key] += 1

                if info.get("full_game_restart", False):
                    self._interval_full_restarts += 1

                self._update_env_progress(env_idx, info)

                if done:
                    self._interval_episode_returns.append(episode_returns[env_idx])
                    self._interval_episode_lengths.append(episode_lengths[env_idx])
                    self._episode_returns_window.append(episode_returns[env_idx])

                    episode_returns[env_idx] = 0.0
                    episode_lengths[env_idx] = 0
                    reset_obs, reset_info = env.reset()
                    observations[env_idx] = reset_obs
                    self._update_env_progress(env_idx, reset_info)

                if self.global_step >= self.config.total_timesteps:
                    break

            performed_updates = 0
            if (
                self.global_step >= self.config.learning_starts
                and steps_since_update >= self.config.train_frequency
                and len(self.buffer) >= self.config.batch_size
            ):
                losses = []
                for _ in range(self.config.gradient_steps):
                    loss_value = self._optimize_model()
                    if loss_value is not None:
                        losses.append(loss_value)
                        performed_updates += 1
                        if (
                            self.config.target_update_interval > 0
                            and self.total_updates % self.config.target_update_interval == 0
                        ):
                            self._sync_target_network()
                if losses:
                    mean_loss = sum(losses) / len(losses)
                    self._interval_loss_sum += mean_loss
                    self._interval_loss_count += 1
            if performed_updates > 0:
                steps_since_update = 0
                self._maybe_checkpoint()

            if self.config.log_interval > 0 and self.global_step - self._last_log_step >= self.config.log_interval:
                self._log_update()

        if self.global_step > self._last_log_step:
            self._log_update()
        self._maybe_checkpoint(force_snapshot=True)

    def save(self, path: str) -> None:
        checkpoint = {
            "model_state": self.q_network.state_dict(),
            "target_state": self.target_network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config.__dict__,
            "global_step": self.global_step,
            "total_updates": self.total_updates,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint["model_state"])
        if "target_state" in checkpoint:
            self.target_network.load_state_dict(checkpoint["target_state"])
        else:
            self._sync_target_network(hard=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.global_step = int(checkpoint.get("global_step", 0))
        self.total_updates = int(checkpoint.get("total_updates", 0))

        torch_rng = checkpoint.get("torch_rng_state")
        if torch_rng is not None:
            if not isinstance(torch_rng, torch.ByteTensor):
                torch_rng = torch.tensor(torch_rng, dtype=torch.uint8)
            torch.set_rng_state(torch_rng)

        numpy_rng = checkpoint.get("numpy_rng_state")
        if numpy_rng is not None:
            if isinstance(numpy_rng, list):
                numpy_rng = tuple(numpy_rng)
            np.random.set_state(numpy_rng)

    def load_weights(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state") if isinstance(checkpoint, dict) else None
        if state_dict is None:
            state_dict = checkpoint
        self.q_network.load_state_dict(state_dict)
        self._sync_target_network(hard=True)

    def close(self) -> None:
        if self.logger:
            self.logger.close()
            self.logger = None

    def _compute_greedy_actions(self, observations: Sequence[np.ndarray]) -> List[int]:
        with torch.no_grad():
            obs_tensor = self._batch_obs_to_tensor(observations)
            q_values = self.q_network(obs_tensor)
            return q_values.argmax(dim=1).detach().cpu().tolist()

    def _epsilon_greedy_action(self, greedy_action: int, epsilon: float) -> int:
        if self._rng.random() < epsilon:
            return self._rng.randrange(self.num_actions)
        return int(greedy_action)

    def _optimize_model(self) -> Optional[float]:
        try:
            obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = self.buffer.sample(
                self.config.batch_size, self.device
            )
        except ValueError:
            return None

        with torch.no_grad():
            next_q_target = self.target_network(next_obs_batch)
            if self.config.double_dqn:
                next_q_online = self.q_network(next_obs_batch)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)
                next_values = next_q_target.gather(1, next_actions).squeeze(1)
            else:
                next_values = next_q_target.max(dim=1).values
            targets = reward_batch + (1.0 - done_batch) * self.config.gamma * next_values

        current_q = self.q_network(obs_batch)
        current_values = current_q.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        loss = F.smooth_l1_loss(current_values, targets, reduction="mean")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        self.total_updates += 1
        return float(loss.detach().cpu())

    def _sync_target_network(self, hard: bool = False) -> None:
        if hard:
            self.target_network.load_state_dict(self.q_network.state_dict())
            return
        self.target_network.load_state_dict(self.q_network.state_dict())

    def _compute_epsilon(self, step: int) -> float:
        if step < self.config.epsilon_decay_start:
            return self.config.epsilon_start
        progress = (step - self.config.epsilon_decay_start) / max(
            1, self.config.epsilon_decay
        )
        progress = min(1.0, max(0.0, progress))
        return self.config.epsilon_start + progress * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32) / 255.0
        if tensor.ndim == len(self.obs_shape):
            tensor = tensor.unsqueeze(0)
        return tensor

    def _batch_obs_to_tensor(self, obs: Sequence[np.ndarray]) -> torch.Tensor:
        stacked = np.stack(obs, axis=0)
        return torch.as_tensor(stacked, device=self.device, dtype=torch.float32) / 255.0

    def _maybe_checkpoint(self, *, force_snapshot: bool = False, force_latest: bool = False) -> None:
        if not self.checkpoint_dir:
            return

        should_snapshot = False
        if force_snapshot:
            should_snapshot = True
        elif (
            self.checkpoint_interval
            and self.global_step
            and self.global_step % self.checkpoint_interval == 0
            and self.global_step != self._last_snapshot_step
        ):
            should_snapshot = True

        write_latest = force_latest or should_snapshot

        if write_latest and (force_snapshot or self.global_step != self._last_latest_step):
            latest_path = self.checkpoint_dir / "latest.pt"
            self.save(str(latest_path))
            self._last_latest_step = self.global_step

        if should_snapshot:
            snapshot_path = self.checkpoint_dir / f"checkpoint_step_{self.global_step:08d}.pt"
            self.save(str(snapshot_path))
            self._last_snapshot_step = self.global_step

    def _update_env_progress(self, env_idx: int, info: Dict) -> None:
        if not info:
            return
        self._current_worlds[env_idx] = int(info.get("current_world", self._current_worlds[env_idx]))
        self._current_levels[env_idx] = int(info.get("current_level", self._current_levels[env_idx]))
        self._furthest_worlds[env_idx] = int(
            info.get("furthest_world", self._furthest_worlds[env_idx])
        )
        self._furthest_levels[env_idx] = int(
            info.get("furthest_level", self._furthest_levels[env_idx])
        )

    def _log_update(self) -> None:
        steps_collected = self.global_step - self._last_log_step
        if steps_collected <= 0:
            return
        self._log_index += 1
        avg_loss = (
            self._interval_loss_sum / self._interval_loss_count
            if self._interval_loss_count
            else 0.0
        )
        avg_return_window = (
            sum(self._episode_returns_window) / len(self._episode_returns_window)
            if self._episode_returns_window
            else 0.0
        )
        mean_episode_return = (
            sum(self._interval_episode_returns) / len(self._interval_episode_returns)
            if self._interval_episode_returns
            else 0.0
        )
        max_episode_return = max(self._interval_episode_returns) if self._interval_episode_returns else 0.0
        mean_episode_length = (
            sum(self._interval_episode_lengths) / len(self._interval_episode_lengths)
            if self._interval_episode_lengths
            else 0.0
        )

        avg_epsilon = (
            self._interval_epsilon_sum / self._interval_epsilon_count
            if self._interval_epsilon_count
            else self._compute_epsilon(self.global_step)
        )

        reward_stats = {}
        for key, total in self._interval_reward_sums.items():
            count = self._interval_reward_counts.get(key, 0)
            reward_stats[key] = {
                "sum": total,
                "count": count,
                "mean_per_step": total / steps_collected if steps_collected > 0 else 0.0,
                "mean_per_occurrence": total / count if count > 0 else 0.0,
            }

        mean_reward_per_step = self._interval_total_reward / steps_collected if steps_collected > 0 else 0.0

        metrics = {
            "update": self._log_index,
            "global_step": self.global_step,
            "total_updates": self.total_updates,
            "steps_collected": steps_collected,
            "total_reward_sum": self._interval_total_reward,
            "policy_loss": avg_loss,
            "value_loss": 0.0,
            "entropy": 0.0,
            "td_loss": avg_loss,
            "avg_return_window": avg_return_window,
            "episodes_completed": len(self._interval_episode_returns),
            "mean_episode_return": mean_episode_return,
            "max_episode_return": max_episode_return,
            "mean_episode_length": mean_episode_length,
            "update_duration_s": time.time() - self._interval_start_time,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "stagnation_reset": float(self._interval_full_restarts > 0),
            "elapsed_s": time.time() - self.training_start,
            "current_world": max(self._current_worlds),
            "current_level": max(self._current_levels),
            "furthest_world": max(self._furthest_worlds),
            "furthest_level": max(self._furthest_levels),
            "full_game_restart_events": self._interval_full_restarts,
            "epsilon": avg_epsilon,
            "buffer_size": len(self.buffer),
            "mean_reward_per_step": mean_reward_per_step,
        }

        print(
            (
                "[DQN] "
                f"step={self.global_step} "
                f"updates={self.total_updates} "
                f"td_loss={avg_loss:.4f} "
                f"reward/step={mean_reward_per_step:.4f} "
                f"episodes={len(self._interval_episode_returns)} "
                f"epsilon={avg_epsilon:.3f} "
                f"buffer={len(self.buffer)} "
                f"curr=W{max(self._current_worlds)}-L{max(self._current_levels)} "
                f"furthest=W{max(self._furthest_worlds)}-L{max(self._furthest_levels)}"
            ),
            flush=True,
        )

        if self.logger:
            self.logger.log_update(metrics, reward_stats)

        self._maybe_checkpoint(force_latest=True)
        self._reset_interval_trackers()
        self._last_log_step = self.global_step

    def _reset_interval_trackers(self) -> None:
        self._interval_total_reward = 0.0
        self._interval_loss_sum = 0.0
        self._interval_loss_count = 0
        self._interval_episode_returns.clear()
        self._interval_episode_lengths.clear()
        self._interval_reward_sums.clear()
        self._interval_reward_counts.clear()
        self._interval_full_restarts = 0
        self._interval_start_time = time.time()
        self._interval_epsilon_sum = 0.0
        self._interval_epsilon_count = 0
