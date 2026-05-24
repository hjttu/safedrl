"""Render a GIF animation of a trained G-MATrans-Lagr agent.

Usage:
    python scripts/render_gif.py --checkpoint checkpoints/gmatrans_lagr_agents3_250.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gmarlagr import GraphActorCritic, MultiUAVNavEnv
from gmarlagr.buffer import collate_graphs
from gmarlagr.env import EnvConfig


AGENT_COLOR = "#2196F3"
AGENT_REACHED_COLOR = "#4CAF50"
AGENT_COLLISION_COLOR = "#F44336"
TARGET_COLOR = "#81C784"
TARGET_MARKER_COLOR = "#2E7D32"
OBSTACLE_COLOR = "#FF9800"
OBSTACLE_EDGE = "#E65100"
VELOCITY_COLOR = "#0D47A1"
BACKGROUND = "#FAFAFA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a GIF from a trained checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/gmatrans_lagr_agents3_250.pt")
    parser.add_argument("--output", type=str, default="render.gif")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args()


@torch.no_grad()
def select_action(model: GraphActorCritic, obs, device: torch.device) -> np.ndarray:
    nodes, edge, mask, self_state = collate_graphs([obs], device)
    graph_emb = model.encode(nodes, edge, mask)
    dist = model.actor.distribution(graph_emb, self_state)
    return dist.probs.argmax(dim=-1).cpu().numpy()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # ── load checkpoint ──────────────────────────────────────────────────────
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(str(checkpoint_path), map_location=device)
    env_cfg_dict = ckpt.get("env_config", {})
    if not isinstance(env_cfg_dict, dict):
        env_cfg_dict = {}
    env_cfg = EnvConfig(n_agents=env_cfg_dict.get("n_agents", 3), seed=args.seed)
    env = MultiUAVNavEnv(env_cfg)

    model = GraphActorCritic(n_actions=env.n_actions).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ── run one episode, recording states ────────────────────────────────────
    obs = env.reset(seed=args.seed)
    frames_state = []
    ep_return = 0.0
    ep_cost = 0.0

    for step in range(1, args.max_steps + 1):
        actions = select_action(model, obs, device)
        obs, reward, cost, terminated, truncated, info = env.step(actions)

        frames_state.append({
            "pos": env.pos.copy(),
            "vel": env.vel.copy(),
            "targets": env.targets.copy(),
            "obstacles": env.obstacles.copy(),
            "world_size": env.world_size,
            "step": step,
            "reward": float(reward.mean()),
            "cost": float(cost.mean()),
            "reached": info["reached"].copy(),
            "collisions": info["collisions"].copy(),
        })
        ep_return += float(reward.mean())
        ep_cost += float(cost.mean())

        if terminated or truncated:
            break

    n_frames = len(frames_state)
    print(f"Episode: {step} steps, return={ep_return:.2f}, cost={ep_cost:.3f}, frames={n_frames}")

    # ── build matplotlib animation ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    def animate(idx: int):
        ax.clear()
        state = frames_state[idx]
        ws = state["world_size"]
        pos = state["pos"]
        vel = state["vel"]
        targets = state["targets"]
        obstacles = state["obstacles"]
        reached = state["reached"]
        collisions = state["collisions"]
        step_i = state["step"]

        ax.set_xlim(-0.3, ws + 0.3)
        ax.set_ylim(-0.3, ws + 0.3)
        ax.set_aspect("equal")
        ax.set_facecolor(BACKGROUND)

        # Obstacles
        for o in obstacles:
            ax.add_patch(Circle(o, env.cfg.obstacle_radius, facecolor=OBSTACLE_COLOR,
                                edgecolor=OBSTACLE_EDGE, linewidth=1.2, alpha=0.8, zorder=2))

        # Targets
        for t in targets:
            ax.add_patch(Circle(t, env.cfg.target_radius, facecolor=TARGET_COLOR, alpha=0.4, zorder=1))
            ax.plot(t[0], t[1], marker="x", color=TARGET_MARKER_COLOR, markersize=9,
                    markeredgewidth=2, zorder=3)

        # Agents
        for i, p in enumerate(pos):
            if collisions[i]:
                col = AGENT_COLLISION_COLOR
            elif reached[i]:
                col = AGENT_REACHED_COLOR
            else:
                col = AGENT_COLOR
            ax.add_patch(Circle(p, env.cfg.agent_radius, facecolor=col, edgecolor="white",
                                linewidth=1.0, alpha=0.9, zorder=4))

        # Velocity arrows & labels
        for i, (p, v) in enumerate(zip(pos, vel)):
            if np.linalg.norm(v) > 1e-6:
                vn = v / (np.linalg.norm(v) + 1e-8) * 0.25
                ax.arrow(p[0], p[1], vn[0], vn[1], head_width=0.06, head_length=0.07,
                         fc=VELOCITY_COLOR, ec=VELOCITY_COLOR, alpha=0.6, zorder=5)
            ax.text(p[0] + 0.18, p[1] + 0.18, str(i), fontsize=8, fontweight="bold",
                    color="#333333", zorder=6)

        # Legend
        legend_elements = [
            Circle((0, 0), 0.12, facecolor=AGENT_COLOR, edgecolor="white", linewidth=0.8, label="Agent"),
            Circle((0, 0), 0.12, facecolor=AGENT_REACHED_COLOR, edgecolor="white", linewidth=0.8, label="Reached"),
            Circle((0, 0), 0.12, facecolor=AGENT_COLLISION_COLOR, edgecolor="white", linewidth=0.8, label="Collision"),
            Circle((0, 0), 0.12, facecolor=OBSTACLE_COLOR, edgecolor=OBSTACLE_EDGE, linewidth=0.8, label="Obstacle"),
            Line2D([0], [0], marker="x", color=TARGET_MARKER_COLOR, markersize=8, linewidth=0, label="Target"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=7,
                  framealpha=0.9, edgecolor="#CCCCCC", ncol=1)

        ax.set_title(
            f"G-MATrans-Lagr  |  3 Agents  |  Step {step_i}/{n_frames}",
            fontsize=11, fontweight="bold", pad=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return []

    ani = animation.FuncAnimation(
        fig, animate, frames=n_frames,
        interval=1000 // args.fps, blit=False, repeat=True, repeat_delay=2000,
    )

    output_path = Path(args.output)
    print(f"Saving GIF ({n_frames} frames, {args.fps} fps) to {output_path} ...")
    ani.save(str(output_path), writer="pillow", fps=args.fps, dpi=args.dpi)
    print("Done!")
    plt.close(fig)


if __name__ == "__main__":
    main()
