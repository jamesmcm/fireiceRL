from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RewardConfig:
    """Configuration for reward shaping based on NES memory addresses."""

    fire_counter_addr: int = 0x00AB
    level_complete_addr: int = 0x06A9
    world_bitmask_start: int = 0x0400
    world_bitmask_end: int = 0x0413
    in_level_addrs: tuple[int, int, int] = (0x0018, 0x001C, 0x00D0)
    pause_addrs: tuple[int, int] = (0x031D, 0x0321)
    level_select_addrs: tuple[int, int] = (0x0324, 0x0328)
    death_flag_addr: int = 0x0003

    fire_reward: float = 2.0
    all_fires_cleared_bonus: float = 20.0
    level_complete_bonus: float = 25.0
    world_progress_reward: float = 50.0
    menu_penalty: float = -0.01
    pause_penalty: float = -0.05
    completed_level_penalty: float = -2.0
    new_level_reward: float = 2.5
    restart_level_penalty: float = -1.0
    death_penalty: float = -10.0
    first_menu_reward: float = 2.0


@dataclass
class RewardTracker:
    """Tracks NES memory values to compute shaped rewards."""

    config: RewardConfig = field(default_factory=RewardConfig)
    previous_fires: int | None = None
    previous_bitmask: Dict[int, int] = field(default_factory=dict)
    previous_in_level: bool = False
    current_level_key: tuple[int, int] | None = None
    last_exit_level_key: tuple[int, int] | None = None
    last_exit_completed: bool = False
    previous_death_flag: bool = False
    previous_level_complete_flag: bool = False
    completed_levels: set[tuple[int, int]] = field(default_factory=set)
    menu_reward_given: bool = False

    def compute_reward(self, ram_snapshot: Dict[int, int]) -> tuple[float, Dict[str, float]]:
        reward = 0.0
        components: Dict[str, float] = {}
        in_level = self._is_in_level(ram_snapshot)
        paused = self._is_paused(ram_snapshot)
        level_complete_flag = ram_snapshot.get(self.config.level_complete_addr, 1)
        world_idx = ram_snapshot.get(0x00D4)
        level_idx = ram_snapshot.get(0x00B5)
        level_key = (world_idx, level_idx) if world_idx is not None and level_idx is not None else None
        just_entered = in_level and not self.previous_in_level

        restart_penalty = 0.0
        exit_level_key = self.current_level_key

        if just_entered:
            if (
                level_key is not None
                and self.last_exit_level_key == level_key
                and not self.last_exit_completed
            ):
                restart_penalty -= abs(self.config.restart_level_penalty)
            self.current_level_key = level_key
            self.previous_fires = ram_snapshot.get(self.config.fire_counter_addr)

        if self.previous_in_level and not in_level and exit_level_key is not None:
            self.last_exit_level_key = exit_level_key
            self.last_exit_completed = level_complete_flag == 0
            self.current_level_key = None
            self.previous_fires = None
        elif not in_level and not just_entered:
            self.current_level_key = None
            self.previous_fires = None

        fire_reward, fire_restart_penalty = self._fire_change_reward(
            ram_snapshot, in_level, just_entered
        )
        restart_penalty += fire_restart_penalty
        if fire_reward:
            reward += fire_reward
            components["fire"] = fire_reward
        if restart_penalty:
            reward += restart_penalty
            components["restart"] = components.get("restart", 0.0) + restart_penalty

        completion_reward = self._completion_reward(level_complete_flag, in_level, level_key)
        if completion_reward:
            reward += completion_reward
            components["completion"] = completion_reward
            if level_key is not None:
                self.completed_levels.add(level_key)
                self.last_exit_completed = True

        world_reward = self._world_progress_reward(ram_snapshot)
        if world_reward:
            reward += world_reward
            components["world_progress"] = world_reward

        level_transition_reward = self._level_transition_reward(ram_snapshot, in_level, level_key)
        if level_transition_reward:
            reward += level_transition_reward
            components["level_transition"] = level_transition_reward

        death_penalty = self._death_penalty(ram_snapshot)
        if death_penalty:
            reward += death_penalty
            components["death"] = components.get("death", 0.0) + death_penalty
        death_event = death_penalty != 0.0

        if paused and in_level:
            reward += self.config.pause_penalty
            components["pause"] = self.config.pause_penalty
        elif not in_level:
            reward += self.config.menu_penalty
            components["menu_penalty"] = self.config.menu_penalty

        if not self.menu_reward_given and not in_level and self._is_level_select(ram_snapshot):
            reward += self.config.first_menu_reward
            components["menu_entry"] = self.config.first_menu_reward
            self.menu_reward_given = True

        if death_event:
            self.previous_in_level = False
        else:
            self.previous_in_level = in_level
        if not in_level:
            self.previous_level_complete_flag = False
        return reward, components

    def _fire_change_reward(
        self,
        ram: Dict[int, int],
        in_level: bool,
        just_entered: bool,
    ) -> tuple[float, float]:
        if not in_level:
            return 0.0, 0.0

        fires = ram.get(self.config.fire_counter_addr)
        if fires is None:
            return 0.0, 0.0

        if just_entered:
            self.previous_fires = fires
            return 0.0, 0.0

        reward = 0.0
        penalty = 0.0
        if self.previous_fires is not None and fires < self.previous_fires:
            reward += (self.previous_fires - fires) * self.config.fire_reward
            if fires == 0:
                reward += self.config.all_fires_cleared_bonus
        elif self.previous_fires is not None and fires > self.previous_fires and not just_entered:
            penalty -= (fires - self.previous_fires) * abs(self.config.restart_level_penalty)

        self.previous_fires = fires
        return reward, penalty

    def _completion_reward(
        self, level_complete_flag: int, in_level: bool, level_key: tuple[int, int] | None
    ) -> float:
        if not in_level:
            self.previous_level_complete_flag = False
            return 0.0

        flag_zero = level_complete_flag == 0
        reward = 0.0
        if (
            flag_zero
            and not self.previous_level_complete_flag
            and level_key is not None
            and level_key not in self.completed_levels
        ):
            reward = self.config.level_complete_bonus
        self.previous_level_complete_flag = flag_zero
        return reward

    def _world_progress_reward(self, ram: Dict[int, int]) -> float:
        reward = 0.0
        for addr in range(
            self.config.world_bitmask_start, self.config.world_bitmask_end + 1
        ):
            new_value = ram.get(addr)
            if new_value is None:
                continue
            old_value = self.previous_bitmask.get(addr, 0)
            delta = (new_value - old_value) & 0xFF
            if delta > 0:
                reward += delta * self.config.world_progress_reward
            self.previous_bitmask[addr] = new_value
        return reward

    def _level_transition_reward(
        self, ram: Dict[int, int], in_level: bool, level_key: tuple[int, int] | None
    ) -> float:
        world_idx = ram.get(0x00D4)
        level_idx = ram.get(0x00B5)
        if world_idx is None or level_idx is None:
            return 0.0

        key = (world_idx, level_idx)
        reward = 0.0
        if in_level and not self.previous_in_level:
            # Just entered a level
            addr_offset = world_idx * 2 + (1 if level_idx >= 8 else 0)
            addr = self.config.world_bitmask_start + addr_offset
            world_mask = ram.get(addr, 0)
            mask_bit = 1 << (level_idx % 8)
            if world_mask & mask_bit:
                reward += self.config.completed_level_penalty
            else:
                if not (
                    self.last_exit_level_key == key and not self.last_exit_completed
                    or (level_key is not None and level_key in self.completed_levels)
                ):
                    reward += self.config.new_level_reward

        return reward

    def _is_in_level(self, ram: Dict[int, int]) -> bool:
        values = [ram.get(addr, 0) for addr in self.config.in_level_addrs]
        return any(values)

    def _is_paused(self, ram: Dict[int, int]) -> bool:
        return any(ram.get(addr, 0) for addr in self.config.pause_addrs)

    def _death_penalty(self, ram: Dict[int, int]) -> float:
        death_active = ram.get(self.config.death_flag_addr, 0) == 8
        penalty = 0.0
        if death_active and not self.previous_death_flag:
            penalty = self.config.death_penalty
            if self.current_level_key is not None:
                self.last_exit_level_key = self.current_level_key
                self.last_exit_completed = False
            self.current_level_key = None
            self.previous_fires = None
            self.previous_level_complete_flag = False
        self.previous_death_flag = death_active
        return penalty

    def _is_level_select(self, ram: Dict[int, int]) -> bool:
        return any(ram.get(addr, 0) for addr in self.config.level_select_addrs)
