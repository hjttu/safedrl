from __future__ import annotations

import argparse

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
    args = parser.parse_args()

    env = MultiUAVNavEnv(EnvConfig(n_agents=args.agents, seed=args.seed))
    model = GraphActorCritic(n_actions=env.n_actions)
    cfg = TrainConfig(total_updates=args.updates, rollout_steps=args.steps)
    if args.device:
        cfg.device = args.device
    GMATransLagrTrainer(env, model, cfg).train()


if __name__ == "__main__":
    main()
