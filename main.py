from __future__ import annotations

import argparse
from pathlib import Path

from fireicerl.environment import FireIceEnv, FireIceEnvConfig
from fireicerl.ppo import PPOConfig, PPOTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PPO agent trainer for Fire 'n Ice using the fceux emulator."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Launch PPO training.")
    train_parser.add_argument(
        "--total-timesteps",
        type=int,
        default=200_000,
        help="Number of environment steps to train for.",
    )
    train_parser.add_argument(
        "--rollout-steps",
        type=int,
        default=128,
        help="Number of steps per PPO rollout before an update.",
    )
    train_parser.add_argument(
        "--frame-skip",
        type=int,
        default=4,
        help="Number of emulator frames to advance per environment step.",
    )
    train_parser.add_argument(
        "--stack-size",
        type=int,
        default=4,
        help="Number of grayscale frames to stack in the observation.",
    )
    train_parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("checkpoints/latest.pt"),
        help="File path to store the trained model (will be created).",
    )
    train_parser.add_argument(
        "--resume-from",
        type=Path,
        help="Optional checkpoint path to resume training from.",
    )
    train_parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory to append detailed training metrics and reward logs.",
    )
    train_parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory to store rolling and periodic PPO checkpoints.",
    )
    train_parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=25,
        help="Number of PPO updates between persistent checkpoint snapshots (0 disables).",
    )
    train_parser.add_argument(
        "--speed-mode",
        choices=["normal", "turbo", "nothrottle"],
        default="normal",
        help="Emulator speed setting applied via the Lua bridge (normal, turbo, nothrottle).",
    )
    train_parser.add_argument(
        "--reset-on-level-complete",
        action="store_true",
        help="End an episode whenever a level is completed (default: disabled).",
    )
    train_parser.add_argument(
        "--reset-on-death",
        action="store_true",
        help="End an episode whenever the agent dies (default: disabled).",
    )
    train_parser.add_argument(
        "--stagnation-update-limit",
        type=int,
        default=3,
        help="Number of consecutive PPO updates without positive reward before forcing a reset (0 disables).",
    )
    train_parser.add_argument(
        "--stagnation-reward-threshold",
        type=float,
        default=0.0,
        help="Reward threshold that counts as progress for stagnation detection (default: >0).",
    )
    train_parser.add_argument(
        "--disable-stagnation-resets",
        action="store_true",
        help="Disable automatic resets when the agent makes no progress for several updates.",
    )

    return parser


def run_train(args: argparse.Namespace) -> None:
    env_config = FireIceEnvConfig(
        frame_skip=args.frame_skip,
        stack_size=args.stack_size,
        speed_mode=args.speed_mode,
        reset_on_level_complete=args.reset_on_level_complete,
        reset_on_death=args.reset_on_death,
    )
    env = FireIceEnv(config=env_config)
    ppo_config = PPOConfig(
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        checkpoint_interval=args.checkpoint_interval,
        stagnation_update_limit=args.stagnation_update_limit,
        stagnation_reward_threshold=args.stagnation_reward_threshold,
        enable_stagnation_resets=not args.disable_stagnation_resets,
    )
    trainer = PPOTrainer(
        env,
        config=ppo_config,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
    )

    if args.resume_from:
        trainer.load(str(args.resume_from))

    try:
        trainer.train()
    finally:
        trainer.close()
        env.close()

    # If the user kept the default save path but customised the checkpoint directory,
    # mirror the latest checkpoint inside that location.
    default_save = Path("checkpoints/latest.pt")
    if args.save_path == default_save and args.checkpoint_dir != default_save.parent:
        args.save_path = args.checkpoint_dir / default_save.name

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(str(args.save_path))
    print(f"Saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    parser = build_parser()
    parsed_args = parser.parse_args()

    if parsed_args.command == "train":
        run_train(parsed_args)
    else:
        parser.error(f"Unknown command: {parsed_args.command}")
