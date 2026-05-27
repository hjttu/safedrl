"""Render GIF using official trained model (matplotlib-based, no pyglet needed).

Usage from project root:
    python scripts/render_from_official_model.py

This reads the models from autodl_results, runs the official GraphMPEEnv,
and produces a matplotlib GIF animation.
"""

from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Must be at official_G_MAT_Lagr root for imports to work
OFFICIAL_ROOT = Path(__file__).resolve().parents[1] / "official_G_MAT_Lagr"
sys.path.insert(0, str(OFFICIAL_ROOT))

from onpolicy.config import get_config
from multiagent.MPE_env import GraphMPEEnv
from onpolicy.envs.env_wrappers import GraphDummyVecEnv


def _t2n(x):
    return x.detach().cpu().numpy()


def main():
    parser = get_config()
    args_list = [
        "--use_valuenorm", "--use_popart",
        "--env_name", "GSMPE",
        "--algorithm_name", "rmappo",
        "--seed", "1",
        "--experiment_name", "check",
        "--scenario_name", "graph_navigation_3agts",
        "--hidden_size", "128",
        "--layer_N", "2",
        "--use_wandb", "False",
        "--save_gifs", "False",
        "--use_render", "True",
        "--save_data", "False",
        "--use_curriculum", "False",
        "--use_policy", "False",
        "--gp_type", "navigation",
        "--num_target", "3",
        "--num_agents", "3",
        "--num_obstacle", "3",
        "--num_dynamic_obs", "0",
        "--n_rollout_threads", "1",
        "--use_lstm", "True",
        "--episode_length", "100",
        "--ppo_epoch", "15",
        "--use_ReLU",
        "--gain", "0.01",
        "--use_cent_obs", "False",
        "--graph_feat_type", "relative",
        "--use_att_gnn", "False",
        "--monte_carlo_test", "True",
        "--model_dir", "",
    ]
    all_args = parser.parse_known_args(args_list)[0]

    # Override with graph config
    from onpolicy.config import graph_config
    all_args, _ = graph_config(args_list, parser)
    all_args.cuda = False

    # Create env
    env = GraphMPEEnv(all_args)
    envs = GraphDummyVecEnv([lambda: env])

    # Load model
    model_dir = Path(r"D:\safedrl\autodl_results\GSMPE\graph_navigation_3agts\rmappo\check\run1\models")
    device = torch.device("cpu")

    all_args.use_valuenorm = True
    all_args.use_popart = True

    from onpolicy.algorithms.graph_lagr_mappo import GS_MAPPO as TrainAlgo
    from onpolicy.algorithms.graph_lagr_MAPPOPolicy import GS_MAPPOPolicy as Policy

    policy = Policy(
        all_args,
        envs.observation_space[0],
        envs.observation_space[0],   # cent_obs_space
        envs.node_observation_space[0],
        envs.adj_observation_space[0],
        envs.action_space[0],
        device=device,
    )

    policy.actor.load_state_dict(torch.load(str(model_dir / "actor.pt"), map_location=device))
    policy.critic.load_state_dict(torch.load(str(model_dir / "critic.pt"), map_location=device))
    policy.cost_critic.load_state_dict(torch.load(str(model_dir / "cost_critic.pt"), map_location=device))

    trainer = TrainAlgo(all_args, policy, device=device)
    trainer.prep_rollout()

    recurrent_N = all_args.recurrent_N
    hidden_size = all_args.hidden_size
    num_agents = all_args.num_agents
    n_rollout_threads = all_args.n_rollout_threads

    # ── Run one episode, record positions ────────────────────────────────────
    obs, agent_id, node_obs, adj = envs.reset()
    rnn_states = np.zeros((n_rollout_threads, num_agents, recurrent_N, hidden_size), dtype=np.float32)
    masks = np.ones((n_rollout_threads, num_agents, 1), dtype=np.float32)

    frames_state = []
    total_reward = 0.0
    total_cost = 0.0

    for step in range(all_args.episode_length):
        trainer.prep_rollout()
        action, rnn_states = policy.act(
            np.concatenate(obs),
            np.concatenate(node_obs),
            np.concatenate(adj),
            np.concatenate(agent_id),
            np.concatenate(rnn_states),
            np.concatenate(masks),
            deterministic=True,
        )
        actions = np.array(np.split(_t2n(action), n_rollout_threads))

        # Convert discrete actions
        if hasattr(envs.action_space[0], "high"):
            actions_env = np.eye(envs.action_space[0].high[0] + 1)[actions[:, :, 0]]
            for i in range(1, envs.action_space[0].shape):
                uc = np.eye(envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                actions_env = np.concatenate((actions_env, uc), axis=2)
        else:
            actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)

        obs, agent_id, node_obs, adj, rewards, costs, dones, infos = envs.step(actions_env)

        # Record world state
        world = env.world
        frame = {
            "egos": [(e.state.p_pos.copy(), e.state.p_vel.copy(), e.goal.copy(),
                      e.R, e.color, getattr(e, "done", False))
                     for e in world.egos],
            "obstacles": [(o.state.p_pos.copy(), o.R) for o in world.obstacles],
            "targets": [(t.state.p_pos.copy(), t.R) for t in world.targets],
        }
        frames_state.append(frame)
        total_reward += float(rewards.mean())
        total_cost += float(costs.mean())

        rnn_states[dones == True] = 0
        masks[dones == True] = 0

        print(f"Step {step+1:3d}: reward={rewards.mean():.2f}, cost={costs.mean():.2f}", end="\r")

    print(f"\nEpisode done: total_reward={total_reward:.2f}, total_cost={total_cost:.2f}")

    # ── Build matplotlib animation ───────────────────────────────────────────
    print(f"Rendering {len(frames_state)} frames...")
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#FAFAFA")

    # Compute world bounds
    all_x, all_y = [], []
    for f in frames_state:
        for p, *_ in f["egos"]:
            all_x.append(p[0]); all_y.append(p[1])
        for p, _ in f["obstacles"]:
            all_x.append(p[0]); all_y.append(p[1])
        for p, _ in f["targets"]:
            all_x.append(p[0]); all_y.append(p[1])
    x_min, x_max = min(all_x) - 1, max(all_x) + 1
    y_min, y_max = min(all_y) - 1, max(all_y) + 1

    def animate(idx):
        ax.clear()
        f = frames_state[idx]
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_facecolor("#FAFAFA")

        # Obstacles
        for o_pos, o_r in f["obstacles"]:
            ax.add_patch(plt.Circle(o_pos, o_r, fc="#FF9800", ec="#E65100", lw=1.2, alpha=0.8, zorder=2))

        # Targets
        for t_pos, t_r in f["targets"]:
            ax.add_patch(plt.Circle(t_pos, t_r, fc="#81C784", alpha=0.3, zorder=1))
            ax.plot(t_pos[0], t_pos[1], "x", color="#2E7D32", ms=9, mew=2, zorder=3)

        # Agents
        for i, (p, v, goal, r, col, done) in enumerate(f["egos"]):
            fc = "#4CAF50" if done else "#2196F3"
            ax.add_patch(plt.Circle(p, r, fc=fc, ec="white", lw=1, alpha=0.9, zorder=4))
            if np.linalg.norm(v) > 0.01:
                vn = v / np.linalg.norm(v) * 0.3
                ax.arrow(p[0], p[1], vn[0], vn[1], head_width=0.06, head_length=0.07,
                         fc="#0D47A1", ec="#0D47A1", alpha=0.6, zorder=5)
            ax.plot(goal[0], goal[1], "s", color="#4CAF50", ms=5, alpha=0.5, zorder=3)
            ax.text(p[0] + 0.2, p[1] + 0.2, str(i), fontsize=8, fontweight="bold", zorder=6, color="#333")

        ax.set_title(f"G-MATrans-Lagr (official) — Step {idx+1}/{len(frames_state)}",
                     fontsize=12, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        return []

    ani = animation.FuncAnimation(fig, animate, frames=len(frames_state),
                                   interval=125, repeat_delay=2000)
    out_path = "render_official_3agents.gif"
    ani.save(out_path, writer="pillow", fps=8, dpi=120)
    print(f"Saved GIF: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
