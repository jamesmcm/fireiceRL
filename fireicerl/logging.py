from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, Optional


class MetricsLogger:
    """Utility for persisting PPO training metrics and reward breakdowns."""

    METRIC_FIELDS = [
        "update",
        "global_step",
        "total_updates",
        "steps_collected",
        "total_reward_sum",
        "policy_loss",
        "value_loss",
        "entropy",
        "avg_return_window",
        "episodes_completed",
        "mean_episode_return",
        "max_episode_return",
        "mean_episode_length",
        "update_duration_s",
        "learning_rate",
        "stagnation_reset",
        "elapsed_s",
        "current_world",
        "current_level",
        "furthest_world",
        "furthest_level",
        "full_game_restart_events",
    ]

    REWARD_FIELDS = ["update", "global_step", "component", "sum", "count", "mean_per_step", "mean_per_occurrence"]

    def __init__(self, log_dir: Path | str) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_path = self.log_dir / "metrics.csv"
        self.rewards_path = self.log_dir / "reward_components.csv"
        self.events_path = self.log_dir / "events.jsonl"

        metrics_exists = self.metrics_path.exists()
        rewards_exists = self.rewards_path.exists()

        self.metrics_file = self.metrics_path.open("a", newline="")
        self.rewards_file = self.rewards_path.open("a", newline="")
        self.events_file = self.events_path.open("a")

        self.metrics_writer = csv.DictWriter(self.metrics_file, fieldnames=self.METRIC_FIELDS)
        self.reward_writer = csv.DictWriter(self.rewards_file, fieldnames=self.REWARD_FIELDS)

        if not metrics_exists or self.metrics_path.stat().st_size == 0:
            self.metrics_writer.writeheader()
            self.metrics_file.flush()

        if not rewards_exists or self.rewards_path.stat().st_size == 0:
            self.reward_writer.writeheader()
            self.rewards_file.flush()

        self.start_time = time.time()

    def log_update(
        self,
        metrics: Dict[str, float],
        reward_stats: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        row = {field: metrics.get(field, 0.0) for field in self.METRIC_FIELDS}
        row["elapsed_s"] = metrics.get("elapsed_s", time.time() - self.start_time)

        self.metrics_writer.writerow(row)
        self.metrics_file.flush()

        if reward_stats:
            for component, stats in reward_stats.items():
                reward_row = {
                    "update": metrics.get("update"),
                    "global_step": metrics.get("global_step"),
                    "component": component,
                    "sum": stats.get("sum", 0.0),
                    "count": stats.get("count", 0),
                    "mean_per_step": stats.get("mean_per_step", 0.0),
                    "mean_per_occurrence": stats.get("mean_per_occurrence", 0.0),
                }
                self.reward_writer.writerow(reward_row)
            self.rewards_file.flush()

        event_payload = {
            "timestamp": time.time(),
            "metrics": metrics,
            "reward_components": reward_stats or {},
        }
        self.events_file.write(json.dumps(event_payload) + "\n")
        self.events_file.flush()

    def close(self) -> None:
        try:
            self.metrics_file.close()
        finally:
            try:
                self.rewards_file.close()
            finally:
                self.events_file.close()
