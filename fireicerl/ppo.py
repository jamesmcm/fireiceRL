from __future__ import annotations

import math
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

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_shape: Tuple[int, ...],
        device: str,
    ) -> None:
        self.device = torch.device(device)
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs = torch.zeros(
            (num_steps, num_envs, *obs_shape), dtype=torch.float32, device=self.device
        )
        self.actions = torch.zeros((num_steps, num_envs), dtype=torch.long, device=self.device)
        self.log_probs = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=self.device)
        self.values = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=self.device)
        self.advantages = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=self.device)
        self.returns = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=self.device)
        self.pos = 0

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        self.obs[self.pos].copy_(obs)
        self.actions[self.pos].copy_(actions)
        self.log_probs[self.pos].copy_(log_probs)
        self.rewards[self.pos].copy_(rewards)
        self.values[self.pos].copy_(values)
        self.dones[self.pos].copy_(dones)
        self.pos = (self.pos + 1) % self.num_steps

    def compute_returns_and_advantages(
        self, last_value: torch.Tensor, last_done: torch.Tensor, config: PPOConfig
    ) -> None:
        last_gae = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
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
        total_samples = self.num_steps * self.num_envs
        indices = torch.randperm(total_samples, device=self.device)
        for start in range(0, total_samples, batch_size):
            yield indices[start : start + batch_size]


class PPOTrainer:
    """High-level helper to collect rollouts and update the PPO policy."""

    def __init__(
        self,
        envs: Sequence[FireIceEnv] | FireIceEnv,
        config: Optional[PPOConfig] = None,
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
            raise ValueError("At least one environment must be provided to PPOTrainer.")

        self.config = config or PPOConfig()
        self.num_envs = len(self.envs)
        first_env = self.envs[0]
        obs_shape = first_env.observation_space.shape
        self.obs_shape = obs_shape
        self.num_actions = first_env.action_space.n
        self.device = torch.device(self.config.device)

        in_channels = obs_shape[0]
        self.model = ActorCritic(in_channels, self.num_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, eps=1e-5)

        self.global_step = 0
        self.episode_returns = []
        self._ongoing_episode_lengths = [0 for _ in range(self.num_envs)]
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
        self._last_snapshot_update = -1
        self._last_latest_update = -1
        self._latest_save_interval = max(1, self.checkpoint_interval or 5)

    def train(self) -> None:
        initial_observations = []
        for env in self.envs:
            obs, _ = env.reset()
            initial_observations.append(obs)
        obs_tensor = self._obs_to_tensor(np.stack(initial_observations))
        last_done = torch.zeros(self.num_envs, device=self.device)

        rollout_steps = self.config.rollout_steps
        steps_per_update = rollout_steps * self.num_envs
        num_updates = max(1, math.ceil(self.config.total_timesteps / steps_per_update))

        episode_rewards = [0.0 for _ in range(self.num_envs)]
        current_episode_lengths = self._ongoing_episode_lengths[:]

        for update in range(1, num_updates + 1):
            update_start = time.time()
            buffer = RolloutBuffer(
                rollout_steps, self.num_envs, self.obs_shape, self.device.type
            )
            reward_sums: Dict[str, float] = defaultdict(float)
            reward_counts: Dict[str, int] = defaultdict(int)
            update_episode_returns: List[float] = []
            update_episode_lengths: List[int] = []
            furthest_worlds = [0 for _ in range(self.num_envs)]
            furthest_levels = [0 for _ in range(self.num_envs)]
            current_worlds = [0 for _ in range(self.num_envs)]
            current_levels = [0 for _ in range(self.num_envs)]
            full_restart_events = 0
            steps_collected = 0

            for _ in range(rollout_steps):
                with torch.no_grad():
                    actions, log_probs, values = self.model.act(obs_tensor)

                actions_cpu = actions.detach().cpu().numpy()
                rewards_step = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
                dones_step = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
                next_observations: List[np.ndarray] = []

                for env_idx, env in enumerate(self.envs):
                    next_obs, reward, terminated, truncated, info = env.step(int(actions_cpu[env_idx]))
                    done = terminated or truncated

                    episode_rewards[env_idx] += reward
                    current_episode_lengths[env_idx] += 1

                    reward_sums["total_reward"] += float(reward)
                    reward_counts["total_reward"] += 1

                    reward_components = info.get("reward_components") or {}
                    for key, value in reward_components.items():
                        reward_sums[key] += float(value)
                        reward_counts[key] += 1

                    world = info.get("current_world") or info.get("world")
                    level = info.get("current_level") or info.get("level")
                    if world is not None:
                        current_worlds[env_idx] = int(world)
                    if level is not None:
                        current_levels[env_idx] = int(level)

                    furthest_world_info = info.get("furthest_world")
                    furthest_level_info = info.get("furthest_level")
                    if furthest_world_info is not None:
                        furthest_worlds[env_idx] = int(furthest_world_info)
                        if furthest_level_info is not None:
                            furthest_levels[env_idx] = int(furthest_level_info)

                    if info.get("full_game_restart"):
                        full_restart_events += 1

                    rewards_step[env_idx] = float(reward)
                    dones_step[env_idx] = 1.0 if done else 0.0

                    if done:
                        update_episode_returns.append(episode_rewards[env_idx])
                        update_episode_lengths.append(current_episode_lengths[env_idx])
                        self.episode_returns.append(episode_rewards[env_idx])
                        episode_rewards[env_idx] = 0.0
                        current_episode_lengths[env_idx] = 0

                        final_info = info
                        next_obs, reset_info = env.reset()
                        reset_info["terminal_info"] = final_info
                        info = reset_info

                    next_observations.append(next_obs)

                buffer.add(
                    obs_tensor.detach().clone(),
                    actions.detach().clone(),
                    log_probs.detach().clone(),
                    rewards_step,
                    values.detach().clone(),
                    dones_step,
                )

                obs_tensor = self._obs_to_tensor(np.stack(next_observations))
                last_done = dones_step
                self.global_step += self.num_envs
                steps_collected += self.num_envs

            self._ongoing_episode_lengths = current_episode_lengths[:]

            with torch.no_grad():
                _, last_value = self.model.get_policy_and_value(obs_tensor)
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
                    refreshed_observations = []
                    for idx, env in enumerate(self.envs):
                        obs, _ = env.reset()
                        refreshed_observations.append(obs)
                        episode_rewards[idx] = 0.0
                        current_episode_lengths[idx] = 0
                    obs_tensor = self._obs_to_tensor(np.stack(refreshed_observations))
                    last_done = torch.zeros(self.num_envs, device=self.device)
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

            mean_reward_per_step = (
                total_reward_sum / steps_collected if steps_collected else 0.0
            )
            print(
                f"Update {update}/{num_updates} "
                f"step={self.global_step} "
                f"loss_pi={losses['policy_loss']:.4f} "
                f"loss_vf={losses['value_loss']:.4f} "
                f"entropy={losses['entropy']:.4f} "
                f"avg_return={avg_return:.2f} "
                f"mean_reward={mean_reward_per_step:.4f} "
                f"episodes={episodes_completed} "
                f"curr=W{max(current_worlds, default=0)}-L{max(current_levels, default=0)} "
                f"furthest=W{max(furthest_worlds, default=0)}-L{max(furthest_levels, default=0)}"
                + (" [stagnation reset]" if stagnation_reset_triggered else ""),
                flush=True,
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
                    "mean_reward_per_step": mean_reward_per_step,
                    "stagnation_reset": int(stagnation_reset_triggered),
                    "elapsed_s": time.time() - self.training_start,
                    "current_world": max(current_worlds, default=0),
                    "current_level": max(current_levels, default=0),
                    "furthest_world": max(furthest_worlds, default=0),
                    "furthest_level": max(furthest_levels, default=0),
                    "full_game_restart_events": full_restart_events,
                    "epsilon": 0.0,
                    "buffer_size": 0,
                    "td_loss": 0.0,
                }
                self.logger.log_update(metrics, reward_stats)

            self._maybe_checkpoint(update, num_updates, force_latest=True)

    def _maybe_checkpoint(
        self, update: int, total_updates: int, *, force_latest: bool = False, force_snapshot: bool = False
    ) -> None:
        if not self.checkpoint_dir:
            return

        should_snapshot = False
        if force_snapshot:
            should_snapshot = True
        elif self.checkpoint_interval and update and update % self.checkpoint_interval == 0:
            should_snapshot = update != self._last_snapshot_update
        elif update == total_updates and update != self._last_snapshot_update:
            should_snapshot = True

        write_latest = False
        if should_snapshot:
            write_latest = True
        elif force_latest and (
            self._last_latest_update == -1
            or update - self._last_latest_update >= self._latest_save_interval
        ):
            write_latest = True

        if write_latest and (should_snapshot or update != self._last_latest_update):
            latest_path = self.checkpoint_dir / "latest.pt"
            self.save(str(latest_path))
            self._last_latest_update = update

        if should_snapshot:
            snapshot_path = self.checkpoint_dir / f"checkpoint_update_{update:05d}.pt"
            self.save(str(snapshot_path))
            self._last_snapshot_update = update

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
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.global_step = int(checkpoint.get("global_step", self.global_step))

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
        self.model.load_state_dict(state_dict)

    def close(self) -> None:
        if self.logger:
            self.logger.close()
            self.logger = None

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(obs).to(self.device, dtype=torch.float32) / 255.0
        if tensor.ndim == len(self.obs_shape):
            tensor = tensor.unsqueeze(0)
        return tensor

    def _update_policy(self, buffer: RolloutBuffer) -> Dict[str, float]:
        losses: Dict[str, float] = {}
        obs_flat = buffer.obs.reshape(-1, *self.obs_shape).detach()
        actions_flat = buffer.actions.reshape(-1)
        old_log_probs_flat = buffer.log_probs.reshape(-1)
        returns_flat = buffer.returns.reshape(-1)
        advantages = buffer.advantages.reshape(-1).clone()

        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.config.update_epochs):
            for indices in buffer.get_batches(self.config.mini_batch_size):
                obs_batch = obs_flat[indices]
                action_batch = actions_flat[indices]
                old_log_probs = old_log_probs_flat[indices]
                returns_batch = returns_flat[indices]
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
