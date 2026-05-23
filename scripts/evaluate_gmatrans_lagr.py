from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gmarlagr import GraphActorCritic, MultiUAVNavEnv
from gmarlagr.buffer import collate_graphs
from gmarlagr.env import EnvConfig


def env_config_from_checkpoint(raw: dict, seed: int | None) -> EnvConfig:
    allowed = {field.name for field in fields(EnvConfig)}
    cfg = {key: value for key, value in raw.items() if key in allowed}
    if seed is not None:
        cfg["seed"] = seed
    return EnvConfig(**cfg)


@torch.no_grad()
def select_actions(model: GraphActorCritic, obs, device: torch.device, deterministic: bool) -> np.ndarray:
    nodes, edge, mask, self_state = collate_graphs([obs], device)
    graph_emb = model.encode(nodes, edge, mask)
    dist = model.actor.distribution(graph_emb, self_state)
    if deterministic:
        action = dist.probs.argmax(dim=-1)
    else:
        action = dist.sample()
    return action.cpu().numpy()


def evaluate(args) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    env_cfg = env_config_from_checkpoint(checkpoint.get("env_config", {}), args.seed)
    env = MultiUAVNavEnv(env_cfg)
    model = GraphActorCritic(n_actions=env.n_actions).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    returns = []
    costs = []
    success_rates = []
    finish_steps = []
    min_distances = []

    for episode in range(args.episodes):
        obs = env.reset(seed=None if args.seed is None else args.seed + episode)
        ep_return = np.zeros(env.cfg.n_agents, dtype=np.float32)
        ep_cost = np.zeros(env.cfg.n_agents, dtype=np.float32)
        reached_once = np.zeros(env.cfg.n_agents, dtype=bool)
        collided_once = np.zeros(env.cfg.n_agents, dtype=bool)
        min_dist = np.full(env.cfg.n_agents, np.inf, dtype=np.float32)

        for step in range(1, env.cfg.episode_len + 1):
            actions = select_actions(model, obs, device, deterministic=not args.stochastic)
            obs, reward, cost, terminated, truncated, info = env.step(actions)
            ep_return += reward
            ep_cost += cost
            reached_once |= info["reached"]
            collided_once |= info["collisions"]
            min_dist = np.minimum(min_dist, info["min_dist"])
            if terminated or truncated:
                break

        success = reached_once & ~collided_once
        returns.append(ep_return.mean())
        costs.append(ep_cost.mean())
        success_rates.append(success.mean())
        finish_steps.append(step)
        min_distances.append(min_dist.mean())

    print(f"checkpoint: {args.checkpoint}")
    print(f"agents: {env.cfg.n_agents}, episodes: {args.episodes}")
    print(f"success_rate: {100.0 * np.mean(success_rates):.2f}%")
    print(f"finish_steps: {np.mean(finish_steps):.2f}")
    print(f"episode_reward: {np.mean(returns):.2f}")
    print(f"episode_cost: {np.mean(costs):.3f}")
    print(f"minimal_distance: {np.mean(min_distances):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained G-MATrans-Lagr checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--stochastic", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
