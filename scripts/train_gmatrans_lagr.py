from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gmarlagr import GMATransLagrTrainer, GraphActorCritic, MultiUAVNavEnv
from gmarlagr.env import EnvConfig
from gmarlagr.trainer import TrainConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train G-MATrans-Lagr on multi-UAV navigation.")
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--run-name", type=str, default="gmatrans_lagr")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    env = MultiUAVNavEnv(EnvConfig(n_agents=args.agents, seed=args.seed))
    model = GraphActorCritic(n_actions=env.n_actions)
    cfg = TrainConfig(
        total_updates=args.updates,
        rollout_steps=args.steps,
        checkpoint_dir=args.checkpoint_dir,
        save_interval=args.save_interval,
        run_name=args.run_name,
    )
    if args.device:
        cfg.device = args.device
    trainer = GMATransLagrTrainer(env, model, cfg)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
