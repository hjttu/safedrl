"""Render GIF using official trained model — matches official pyglet render style.

Usage (from official_G_MAT_Lagr root):
    python ../scripts/render_minimal.py
"""

from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

OFFICIAL_ROOT = Path(__file__).resolve().parents[1] / "official_G_MAT_Lagr"
sys.path.insert(0, str(OFFICIAL_ROOT))

from onpolicy.config import get_config
from multiagent.MPE_env import GraphMPEEnv
from onpolicy.envs.env_wrappers import GraphDummyVecEnv


def _t2n(x):
    return x.detach().cpu().numpy()


def make_runner(model_dir: str):
    """Create GSMPERunner same as eval_mpe.py."""
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
        "--num_target", "3", "--num_agents", "3", "--num_obstacle", "3",
        "--num_dynamic_obs", "0",
        "--n_rollout_threads", "1",
        "--use_lstm", "True",
        "--episode_length", "100",
        "--ppo_epoch", "15", "--use_ReLU", "--gain", "0.01",
        "--use_cent_obs", "False",
        "--graph_feat_type", "relative",
        "--use_att_gnn", "False",
        "--monte_carlo_test", "True",
        "--model_dir", model_dir,
    ]
    all_args = parser.parse_known_args(args_list)[0]
    from onpolicy.config import graph_config
    all_args, _ = graph_config(args_list, parser)
    all_args.cuda = False

    env = GraphMPEEnv(all_args)
    envs = GraphDummyVecEnv([lambda: env])

    config = {
        "all_args": all_args, "envs": envs, "eval_envs": None,
        "num_agents": all_args.num_agents,
        "device": torch.device("cpu"), "run_dir": None,
    }
    from onpolicy.runner.shared.graph_lagr_mpe_runner import GSMPERunner
    runner = GSMPERunner(config)
    return runner, all_args


def main():
    MODEL_DIR = "onpolicy/results/GSMPE/graph_navigation_3agts/rmappo/check/run1"
    print("Loading model...")
    runner, all_args = make_runner(MODEL_DIR)

    envs = runner.envs
    raw_env = envs.envs[0]
    trainer = runner.trainer
    policy = trainer.policy
    num_agents = runner.num_agents
    hidden_size = runner.hidden_size
    recurrent_N = runner.recurrent_N
    n_rollout_threads = runner.n_rollout_threads
    episode_length = runner.episode_length

    # ── Run one episode, record world snapshots ───────────────────────────────
    obs, agent_id, node_obs, adj = envs.reset()
    rnn_states = np.zeros((n_rollout_threads, num_agents, recurrent_N, hidden_size), dtype=np.float32)
    masks = np.ones((n_rollout_threads, num_agents, 1), dtype=np.float32)

    frames = []
    total_r, total_c = 0.0, 0.0

    for step in range(episode_length):
        trainer.prep_rollout()
        action, rnn_states_out = policy.act(
            np.concatenate(obs),
            np.concatenate(node_obs),
            np.concatenate(adj),
            np.concatenate(agent_id),
            np.concatenate(rnn_states),
            np.concatenate(masks),
            deterministic=True,
        )
        actions = np.array(np.split(_t2n(action), n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states_out), n_rollout_threads))

        act_space0 = envs.action_space[0]
        if act_space0.__class__.__name__ == "MultiDiscrete":
            for i in range(act_space0.shape):
                uc = np.eye(act_space0.high[i] + 1)[actions[:, :, i]]
                actions_env = uc if i == 0 else np.concatenate((actions_env, uc), axis=2)
        elif act_space0.__class__.__name__ == "Discrete":
            actions_env = np.squeeze(np.eye(act_space0.n)[actions], 2)
        else:
            raise NotImplementedError

        obs, agent_id, node_obs, adj, rewards, costs, dones, infos = envs.step(actions_env)
        total_r += float(rewards.mean())
        total_c += float(costs.mean())

        # Snapshot world state (deep copy positions)
        w = raw_env.world
        frames.append({
            "egos": [
                {
                    "pos": np.array(e.state.p_pos),
                    "vel": np.array(e.state.p_vel),
                    "goal": np.array(e.goal),
                    "r": e.R,
                    "color": np.array(e.color),
                    "goal_color": np.array(e.goal_color),
                    "done": bool(e.done),
                }
                for e in w.egos
            ],
            "obstacles": [
                {"pos": np.array(o.state.p_pos), "r": o.R, "color": np.array(o.color)}
                for o in w.obstacles
            ],
            "targets": [
                {"pos": np.array(t.state.p_pos), "r": t.R, "color": np.array(t.color)}
                for t in w.targets
            ],
        })

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), recurrent_N, hidden_size), dtype=np.float32)
        masks = np.ones((n_rollout_threads, num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if step % 10 == 0 or step == episode_length - 1:
            sys.stdout.write(f"\rStep {step+1}/{episode_length}  r={total_r:.1f}  c={total_c:.2f}")

    print(f"\nDone. Total reward={total_r:.2f}, cost={total_c:.2f}")

    # ── Build matplotlib animation — matches official render style ────────────
    print(f"Rendering {len(frames)} frames...")
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="white")

    # Official uses set_bounds(-10, 10, -5, 15) for a 700x700 window.
    # We match with a square view centered at (0, 5).
    VIEW_X = (-7, 7)
    VIEW_Y = (-2, 12)

    def animate(idx):
        ax.clear()
        ax.set_xlim(*VIEW_X)
        ax.set_ylim(*VIEW_Y)
        ax.set_aspect("equal")
        ax.set_facecolor("white")
        ax.set_xticks([]); ax.set_yticks([])

        f = frames[idx]

        # ── Obstacles ──
        for o in f["obstacles"]:
            c = tuple(o["color"].tolist())
            ax.add_patch(plt.Circle(o["pos"], o["r"], fc=c, ec=c, lw=0, alpha=0.8, zorder=2))

        # ── Targets ──
        for t in f["targets"]:
            c = tuple(t["color"].tolist())
            ax.add_patch(plt.Circle(t["pos"], t["r"], fc=c, ec=c, lw=0, alpha=0.8, zorder=1))

        # ── Goal circles ──
        for e in f["egos"]:
            gc = tuple(e["goal_color"].tolist())
            ax.add_patch(plt.Circle(e["goal"], e["r"], fc="none", ec=gc, lw=2, alpha=0.4, zorder=3))

        # ── Agents + velocity lines ──
        for i, e in enumerate(f["egos"]):
            p, v = e["pos"], e["vel"]
            # Velocity line: pos → pos + vel (same as official)
            if not e["done"]:
                v_end = p + v * 1.0
                c = tuple(e["color"].tolist())
                ax.plot([p[0], v_end[0]], [p[1], v_end[1]], color=c, alpha=0.5, lw=2, zorder=5)

            # Agent: green if done, else agent color
            fc = (0.0, 0.8, 0.0) if e["done"] else tuple(e["color"].tolist())
            ax.add_patch(plt.Circle(p, e["r"], fc=fc, ec="white", lw=0.8, alpha=0.85, zorder=6))
            ax.text(p[0] + 0.2, p[1] + 0.2, str(i), fontsize=8, fontweight="bold", color="#333", zorder=7)

        # ── Title & Legend ──
        ax.set_title(
            f"G-MATrans-Lagr (official)  |  3 Agents  |  Step {idx+1}/{len(frames)}  |  "
            f"R={total_r:.1f}  C={total_c:.2f}",
            fontsize=11, fontweight="bold", pad=6,
        )

        from matplotlib.lines import Line2D
        leg = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=(0.95, 0.45, 0.45), markersize=10, label="Agent"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=(0.0, 0.8, 0.0), markersize=10, label="Reached"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=(0.45, 0.45, 0.95), markersize=10, label="Obstacle"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=(0.95, 0.95, 0.0), markersize=10, label="Target"),
            Line2D([0], [0], marker="o", color="w", markeredgecolor=(0.95, 0.95, 0.0),
                   markerfacecolor="none", markersize=12, mew=1.5, label="Goal"),
        ]
        ax.legend(handles=leg, loc="upper right", fontsize=7, framealpha=0.85, ncol=1)
        return []

    ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=125, repeat_delay=2000)
    out = "render_official_3agents.gif"
    print(f"Saving → {out}")
    ani.save(out, writer="pillow", fps=8, dpi=130)
    print("Done!")
    plt.close(fig)


if __name__ == "__main__":
    main()
