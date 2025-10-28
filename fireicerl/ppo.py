from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .environment import FireIceEnv
from .models import ActorCritic
from .logging import MetricsLogger


@dataclass
class PPOConfig:
    rollout_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    update_epochs: int = 4
    mini_batch_size: int = 32
    learning_rate: float = 2.5e-4
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    normalize_advantages: bool = True
    total_timesteps: int = 1_000_000
    checkpoint_interval: int = 25
    enable_stagnation_resets: bool = True
    stagnation_update_limit: int = 3
    stagnation_reward_threshold: float = 0.0


class RolloutBuffer:
    """Stores trajectories for PPO updates."""

    def __init__(self, buffer_size: int, obs_shape: Tuple[int, ...], device: str) -> None:
        self.device = torch.device(device)
        self.buffer_size = buffer_size
        self.obs = torch.zeros((buffer_size, *obs_shape), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros(buffer_size, dtype=torch.long, device=self.device)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=self.device)
        self.values = torch.zeros(buffer_size, dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32, device=self.device)
        self.advantages = torch.zeros(buffer_size, dtype=torch.float32, device=self.device)
        self.returns = torch.zeros(buffer_size, dtype=torch.float32, device=self.device)
        self.pos = 0

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        done: bool,
    ) -> None:
        self.obs[self.pos].copy_(obs)
        self.actions[self.pos] = action
        self.log_probs[self.pos] = log_prob
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.buffer_size

    def compute_returns_and_advantages(
        self, last_value: torch.Tensor, last_done: torch.Tensor, config: PPOConfig
    ) -> None:
        last_gae = 0.0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_value = last_value
                next_done = last_done
            else:
                next_value = self.values[step + 1]
                next_done = self.dones[step + 1]

            delta = (
                self.rewards[step]
                + config.gamma * next_value * (1.0 - next_done)
                - self.values[step]
            )
            last_gae = (
                delta
                + config.gamma * config.gae_lambda * (1.0 - next_done) * last_gae
            )
            self.advantages[step] = last_gae
        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int):
        indices = torch.randperm(self.buffer_size, device=self.device)
        for start in range(0, self.buffer_size, batch_size):
            yield indices[start : start + batch_size]


class PPOTrainer:
    """High-level helper to collect rollouts and update the PPO policy."""

    def __init__(
        self,
        env: FireIceEnv,
        config: Optional[PPOConfig] = None,
        *,
        log_dir: Optional[str | Path] = None,
        checkpoint_dir: Optional[str | Path] = None,
        checkpoint_interval: Optional[int] = None,
    ) -> None:
        self.env = env
        self.config = config or PPOConfig()
        obs_shape = env.observation_space.shape
        self.num_actions = env.action_space.n
        self.device = torch.device(self.config.device)

        in_channels = obs_shape[0]
        self.model = ActorCritic(in_channels, self.num_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, eps=1e-5)

        self.global_step = 0
        self.episode_returns = []
        self._ongoing_episode_length = 0
        self.training_start = time.time()
        if (
            self.config.enable_stagnation_resets
            and self.config.stagnation_update_limit > 0
        ):
            self._stagnation_window: Optional[Deque[bool]] = deque(
                maxlen=self.config.stagnation_update_limit
            )
        else:
            self._stagnation_window = None

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

    def train(self) -> None:
        obs, _ = self.env.reset()
        obs_tensor = self._obs_to_tensor(obs)
        last_done = torch.zeros(1, device=self.device)

        rollout_steps = self.config.rollout_steps
        num_updates = math.ceil(self.config.total_timesteps / rollout_steps)

        for update in range(1, num_updates + 1):
            update_start = time.time()
            buffer = RolloutBuffer(
                rollout_steps, self.env.observation_space.shape, self.device.type
            )
            episode_reward = 0.0
            steps_collected = 0
            reward_sums: Dict[str, float] = defaultdict(float)
            reward_counts: Dict[str, int] = defaultdict(int)
            update_episode_returns: list[float] = []
            update_episode_lengths: list[int] = []
            current_episode_length = self._ongoing_episode_length

            for _ in range(rollout_steps):
                with torch.no_grad():
                    action, log_prob, value = self.model.act(obs_tensor)

                next_obs, reward, terminated, truncated, info = self.env.step(
                    action.item()
                )
                done = terminated or truncated
                episode_reward += reward
                steps_collected += 1
                current_episode_length += 1

                reward_sums["total_reward"] += float(reward)
                reward_counts["total_reward"] += 1

                reward_components = info.get("reward_components") or {}
                for key, value in reward_components.items():
                    reward_sums[key] += float(value)
                    reward_counts[key] += 1

                buffer.add(
                    obs_tensor.squeeze(0),
                    action,
                    log_prob,
                    reward,
                    value,
                    done,
                )

                self.global_step += 1
                obs_tensor = self._obs_to_tensor(next_obs)

                if done:
                    update_episode_returns.append(episode_reward)
                    update_episode_lengths.append(current_episode_length)
                    self.episode_returns.append(episode_reward)
                    episode_reward = 0.0
                    current_episode_length = 0
                    next_obs, _ = self.env.reset()
                    obs_tensor = self._obs_to_tensor(next_obs)
                    last_done = torch.ones(1, device=self.device)
                else:
                    last_done = torch.zeros(1, device=self.device)

            self._ongoing_episode_length = current_episode_length

            with torch.no_grad():
                _, _, last_value = self.model.act(obs_tensor)
            buffer.compute_returns_and_advantages(
                last_value.detach(), last_done, self.config
            )

            losses = self._update_policy(buffer)
            avg_return = (
                sum(self.episode_returns[-10:]) / min(len(self.episode_returns), 10)
                if self.episode_returns
                else 0.0
            )

            reward_stats: Dict[str, Dict[str, float]] = {}
            for key, total in reward_sums.items():
                count = reward_counts.get(key, 0)
                reward_stats[key] = {
                    "sum": float(total),
                    "count": int(count),
                    "mean_per_step": float(total / steps_collected)
                    if steps_collected
                    else 0.0,
                    "mean_per_occurrence": float(total / count) if count else 0.0,
                }

            total_reward_sum = reward_sums.get("total_reward", 0.0)
            stagnation_reset_triggered = False
            if self._stagnation_window is not None:
                positive_progress = total_reward_sum > self.config.stagnation_reward_threshold
                self._stagnation_window.append(positive_progress)
                if (
                    len(self._stagnation_window) == self._stagnation_window.maxlen
                    and not any(self._stagnation_window)
                ):
                    obs, _ = self.env.reset()
                    obs_tensor = self._obs_to_tensor(obs)
                    last_done = torch.zeros(1, device=self.device)
                    self._stagnation_window.clear()
                    stagnation_reset_triggered = True

            episodes_completed = len(update_episode_returns)
            mean_episode_return = (
                sum(update_episode_returns) / episodes_completed
                if episodes_completed
                else 0.0
            )
            max_episode_return = (
                max(update_episode_returns) if update_episode_returns else 0.0
            )
            mean_episode_length = (
                sum(update_episode_lengths) / episodes_completed
                if episodes_completed
                else 0.0
            )
            update_duration = time.time() - update_start

            reward_mean_per_step = reward_stats.get("total_reward", {}).get(
                "mean_per_step", 0.0
            )
            print(
                f"Update {update}/{num_updates} "
                f"step={self.global_step} "
                f"loss_pi={losses['policy_loss']:.4f} "
                f"loss_vf={losses['value_loss']:.4f} "
                f"entropy={losses['entropy']:.4f} "
                f"avg_return={avg_return:.2f} "
                f"mean_reward={reward_mean_per_step:.4f} "
                f"episodes={episodes_completed}"
                + (" [stagnation reset]" if stagnation_reset_triggered else "")
            )

            if self.logger:
                metrics = {
                    "update": update,
                    "global_step": self.global_step,
                    "total_updates": num_updates,
                    "steps_collected": steps_collected,
                    "policy_loss": losses.get("policy_loss", 0.0),
                    "value_loss": losses.get("value_loss", 0.0),
                    "entropy": losses.get("entropy", 0.0),
                    "avg_return_window": avg_return,
                    "episodes_completed": episodes_completed,
                    "mean_episode_return": mean_episode_return,
                    "max_episode_return": max_episode_return,
                    "mean_episode_length": mean_episode_length,
                    "update_duration_s": update_duration,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "total_reward_sum": total_reward_sum,
                    "stagnation_reset": int(stagnation_reset_triggered),
                    "elapsed_s": time.time() - self.training_start,
                }
                self.logger.log_update(metrics, reward_stats)

            self._maybe_checkpoint(update, num_updates)

    def _maybe_checkpoint(self, update: int, total_updates: int) -> None:
        if not self.checkpoint_dir:
            return

        latest_path = self.checkpoint_dir / "latest.pt"
        self.save(str(latest_path))

        if self.checkpoint_interval and (
            update % self.checkpoint_interval == 0 or update == total_updates
        ):
            snapshot_path = self.checkpoint_dir / f"checkpoint_update_{update:05d}.pt"
            self.save(str(snapshot_path))

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": self.config.__dict__,
                "global_step": self.global_step,
                "torch_rng_state": torch.get_rng_state(),
                "numpy_rng_state": np.random.get_state(),
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.global_step = int(checkpoint.get("global_step", self.global_step))

        torch_rng = checkpoint.get("torch_rng_state")
        if torch_rng is not None:
            torch.set_rng_state(torch_rng)

        numpy_rng = checkpoint.get("numpy_rng_state")
        if numpy_rng is not None:
            np.random.set_state(numpy_rng)

    def close(self) -> None:
        if self.logger:
            self.logger.close()
            self.logger = None

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(obs).to(self.device, dtype=torch.float32) / 255.0
        return tensor.unsqueeze(0)

    def _update_policy(self, buffer: RolloutBuffer) -> Dict[str, float]:
        losses: Dict[str, float] = {}
        advantages = buffer.advantages.clone()
        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.config.update_epochs):
            for indices in buffer.get_batches(self.config.mini_batch_size):
                obs_batch = buffer.obs[indices]
                action_batch = buffer.actions[indices]
                old_log_probs = buffer.log_probs[indices]
                returns_batch = buffer.returns[indices]
                adv_batch = advantages[indices]

                new_log_probs, entropy, values = self.model.evaluate_actions(
                    obs_batch, action_batch
                )

                ratio = torch.exp(new_log_probs - old_log_probs)
                surrogate1 = ratio * adv_batch
                surrogate2 = torch.clamp(
                    ratio, 1.0 - self.config.clip_coef, 1.0 + self.config.clip_coef
                ) * adv_batch
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                value_loss = F.smooth_l1_loss(values, returns_batch)
                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + self.config.vf_coef * value_loss
                    - self.config.ent_coef * entropy_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                losses = {
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "value_loss": float(value_loss.detach().cpu()),
                    "entropy": float(entropy_loss.detach().cpu()),
                }

        return losses
