from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class RewardConfig:
    """Configuration for reward shaping based on NES memory addresses."""

    fire_counter_addr: int = 0x00AB
    fire_reward: float = 2.0
    failure_penalty: float = -12.0
    level_complete_bonus: float = 50.0
    stagnation_penalty: float = -12.0


@dataclass
class RewardTracker:
    """Tracks NES memory values to compute shaped rewards."""

    config: RewardConfig = field(default_factory=RewardConfig)
    previous_fires: Optional[int] = None

    def reset(self) -> None:
        self.previous_fires = None

    def compute_reward(
        self,
        ram_snapshot: Dict[int, int],
        info: Optional[Dict[str, object]] = None,
        action_name: Optional[str] = None,
    ) -> tuple[float, Dict[str, float]]:
        reward = 0.0
        components: Dict[str, float] = {}
        fires = ram_snapshot.get(self.config.fire_counter_addr)

        if fires is not None:
            if self.previous_fires is None:
                self.previous_fires = fires
            else:
                if fires < self.previous_fires:
                    delta = self.previous_fires - fires
                    if delta > 0:
                        fire_reward = delta * self.config.fire_reward
                        reward += fire_reward
                        components["fire"] = components.get("fire", 0.0) + fire_reward
                self.previous_fires = fires

        failure_event = False
        stagnation_event = False
        if info:
            failure_event = bool(info.get("death_event") or info.get("level_restart_event"))
            stagnation_event = bool(info.get("frame_stagnation_reset"))
        if action_name == "LEVELRESTART":
            failure_event = True

        if failure_event:
            reward += self.config.failure_penalty
            components["failure"] = components.get("failure", 0.0) + self.config.failure_penalty
            self.reset()
            fires = ram_snapshot.get(self.config.fire_counter_addr)
            if fires is not None:
                self.previous_fires = fires
        elif stagnation_event:
            reward += self.config.stagnation_penalty
            components["stagnation"] = (
                components.get("stagnation", 0.0) + self.config.stagnation_penalty
            )
            self.reset()
            fires = ram_snapshot.get(self.config.fire_counter_addr)
            if fires is not None:
                self.previous_fires = fires
        elif info and info.get("level_completed_event"):
            reward += self.config.level_complete_bonus
            components["level_complete"] = (
                components.get("level_complete", 0.0) + self.config.level_complete_bonus
            )
            self.reset()

        return reward, components
