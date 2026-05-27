"""Quick train + render flow for gmarlagr (3 agents, ~2000 updates)."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from gmarlagr import GMATransLagrTrainer, GraphActorCritic, MultiUAVNavEnv
from gmarlagr.buffer import collate_graphs
from gmarlagr.env import EnvConfig
from gmarlagr.trainer import TrainConfig

OUT_GIF = "render_3agents.gif"
TRAIN_UPDATES = 1500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
print(f"Training for {TRAIN_UPDATES} updates...")

# Train
env = MultiUAVNavEnv(EnvConfig(n_agents=3, seed=0))
model = GraphActorCritic(n_actions=env.n_actions)
cfg = TrainConfig(
    total_updates=TRAIN_UPDATES,
    rollout_steps=128,
    save_interval=0,
)
trainer = GMATransLagrTrainer(env, model, cfg)
trainer.train()
print("Training done.")

# Evaluate to find a good seed
device = torch.device(DEVICE)
model.eval()

@torch.no_grad()
def run_episode(seed):
    e = MultiUAVNavEnv(EnvConfig(n_agents=3))
    obs = e.reset(seed=seed)
    frames = []
    total_r, total_c = 0.0, 0.0
    for step in range(1, 101):
        nodes, edge, mask, self_state = collate_graphs([obs], device)
        graph_emb = model.encode(nodes, edge, mask)
        dist = model.actor.distribution(graph_emb, self_state)
        actions = dist.probs.argmax(dim=-1).cpu().numpy()
        frames.append({
            "pos": e.pos.copy(), "vel": e.vel.copy(),
            "targets": e.targets.copy(), "obstacles": e.obstacles.copy(),
            "world_size": e.world_size,
        })
        obs, reward, cost, terminated, truncated, info = e.step(actions)
        total_r += float(reward.mean())
        total_c += float(cost.mean())
        if terminated or truncated:
            break
    return total_r, total_c, frames

best_r, best_frames = -1e9, None
for seed in range(150):
    r, c, frames = run_episode(seed)
    if r > best_r:
        best_r, best_frames = r, frames
        print(f"  seed={seed}: r={r:.1f}, c={c:.2f}, frames={len(frames)} ***BEST***")
    elif seed % 50 == 49:
        print(f"  seed={seed}: r={r:.1f} (best={best_r:.1f})")

print(f"\nBest seed return={best_r:.1f}, {len(best_frames)} frames")

# Render
print("Rendering GIF...")
fig, ax = plt.subplots(figsize=(7, 7))

def animate(idx):
    ax.clear()
    s = best_frames[idx]
    ws = s["world_size"]
    ax.set_xlim(-0.3, ws + 0.3)
    ax.set_ylim(-0.3, ws + 0.3)
    ax.set_aspect("equal")
    ax.set_facecolor("#FAFAFA")
    for o in s["obstacles"]:
        ax.add_patch(plt.Circle(o, 0.15, fc="#FF9800", ec="#E65100", lw=1, alpha=0.8))
    for t in s["targets"]:
        ax.add_patch(plt.Circle(t, 0.1, fc="#81C784", alpha=0.3))
        ax.plot(t[0], t[1], "x", color="#2E7D32", ms=8, mew=2)
    for i, p in enumerate(s["pos"]):
        ax.add_patch(plt.Circle(p, 0.1, fc="#2196F3", ec="white", lw=1, alpha=0.9))
        v = s["vel"][i]
        if np.linalg.norm(v) > 0.01:
            vn = v / np.linalg.norm(v) * 0.2
            ax.arrow(p[0], p[1], vn[0], vn[1], head_width=0.05, head_length=0.06, fc="#0D47A1", alpha=0.6)
        ax.text(p[0]+0.15, p[1]+0.15, str(i), fontsize=8, fontweight="bold")
    ax.set_title(f"G-MATrans-Lagr — 3 Agents — Step {idx+1}/{len(best_frames)}", fontsize=12, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    return []

ani = animation.FuncAnimation(fig, animate, frames=len(best_frames), interval=125, repeat_delay=2000)
ani.save(OUT_GIF, writer="pillow", fps=8, dpi=120)
print(f"Saved: {OUT_GIF}")
plt.close()
