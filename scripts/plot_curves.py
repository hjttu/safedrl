"""Plot training curves from official G-MAT-Lagr summary.json."""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SUMMARY = Path(r"D:\safedrl\autodl_results\GSMPE\graph_navigation_3agts\rmappo\check\run1\logs\summary.json")
OUT_DIR = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path(".")


def extract_metric(data: dict, keyword: str) -> list[tuple[int, float]]:
    """Find all key-value pairs matching keyword."""
    for k, v in data.items():
        if keyword in k and isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
            return [(int(item[1]), float(item[2])) for item in v]
    return []


def smooth(y: np.ndarray, window: int = 10) -> np.ndarray:
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def main():
    with open(SUMMARY) as f:
        data = json.load(f)

    # Extract metrics
    reward = extract_metric(data, "average_episode_rewards")
    cost = extract_metric(data, "average_episode_costs")
    policy_loss = extract_metric(data, "policy_loss")
    value_loss = extract_metric(data, "value_loss")
    cost_loss = extract_metric(data, "cost_loss")
    entropy = extract_metric(data, "dist_entropy")
    lamda = extract_metric(data, "lamda_lagr")

    if not reward:
        print("No reward data found!")
        return

    steps = np.array([s for s, _ in reward])
    r = np.array([v for _, v in reward])
    c = np.array([v for _, v in cost]) if cost else np.zeros_like(r)

    # ── Figure 1: Reward + Cost ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("G-MATrans-Lagr — 3 Agents — Training Curves (official)", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(steps, r, alpha=0.25, color="#2196F3", linewidth=0.8)
    ax.plot(steps, smooth(r, 20), color="#2196F3", linewidth=2, label="Smoothed Reward")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Average Episode Reward")
    ax.set_title("Episode Reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(steps, c, alpha=0.25, color="#F44336", linewidth=0.8)
    ax.plot(steps, smooth(c, 20), color="#F44336", linewidth=2, label="Smoothed Cost")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Average Episode Cost")
    ax.set_title("Episode Cost (Safety Violations)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Figure 1 continued: Policy Loss + Lambda ─────────────────────────────
    ax = axes[1, 0]
    if policy_loss:
        pl_steps = np.array([s for s, _ in policy_loss])
        pl = np.array([v for _, v in policy_loss])
        ax.plot(pl_steps, pl, alpha=0.2, color="#4CAF50", linewidth=0.6)
        ax.plot(pl_steps, smooth(pl, 20), color="#4CAF50", linewidth=2, label="Policy Loss")
    if value_loss:
        vl_steps = np.array([s for s, _ in value_loss])
        vl = np.array([v for _, v in value_loss])
        ax.plot(vl_steps, vl, alpha=0.2, color="#FF9800", linewidth=0.6)
        ax.plot(vl_steps, smooth(vl, 20), color="#FF9800", linewidth=2, label="Value Loss")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Loss")
    ax.set_title("Policy & Value Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if lamda:
        lm_steps = np.array([s for s, _ in lamda])
        lm = np.array([v for _, v in lamda])
        ax.plot(lm_steps, lm, color="#9C27B0", linewidth=1.5, label="Lagrange λ")
    if cost_loss:
        cl_steps = np.array([s for s, _ in cost_loss])
        cl = np.array([v for _, v in cost_loss])
        ax2 = ax.twinx()
        ax2.plot(cl_steps, cl, alpha=0.3, color="#795548", linewidth=0.6)
        ax2.plot(cl_steps, smooth(cl, 20), color="#795548", linewidth=1.5, label="Cost Loss")
        ax2.set_ylabel("Cost Loss", color="#795548")
        ax2.tick_params(axis="y", labelcolor="#795548")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Lagrange Multiplier λ")
    ax.set_title("Lagrange Multiplier")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out1 = OUT_DIR / "training_curves.png"
    fig.savefig(str(out1), dpi=150, bbox_inches="tight")
    print(f"Saved: {out1}")
    plt.close(fig)

    # ── Figure 2: Detailed subplots ──────────────────────────────────────────
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle("G-MATrans-Lagr — Detailed Training Metrics", fontsize=14, fontweight="bold")

    metrics = [
        ("Actor Grad Norm", extract_metric(data, "actor_grad_norm"), "#E91E63"),
        ("Critic Grad Norm", extract_metric(data, "critic_grad_norm"), "#00BCD4"),
        ("Cost Grad Norm", extract_metric(data, "cost_grad_norm"), "#FF5722"),
        ("Distribution Entropy", extract_metric(data, "dist_entropy"), "#3F51B5"),
        ("Importance Ratio", extract_metric(data, r'"ratio"|ratio'), "#607D8B"),
        ("Cost Adv Target", extract_metric(data, "cost_adv_targ"), "#8BC34A"),
    ]

    for idx, (title, metric, color) in enumerate(metrics):
        ax = axes2[idx // 3, idx % 3]
        if not metric:
            ax.set_title(f"{title} (no data)")
            continue
        s = np.array([x[0] for x in metric])
        v = np.array([x[1] for x in metric])
        ax.plot(s, v, alpha=0.3, color=color, linewidth=0.6)
        ax.plot(s, smooth(v, 15), color=color, linewidth=1.5)
        ax.set_xlabel("Steps")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    fig2.tight_layout()
    out2 = OUT_DIR / "training_details.png"
    fig2.savefig(str(out2), dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.close(fig2)

    print(f"\n=== Final Metrics ===")
    print(f"  Steps: {steps[-1]:,}")
    print(f"  Reward: {r[-1]:.2f} (max: {r.max():.2f})")
    print(f"  Cost:   {c[-1]:.2f}")


if __name__ == "__main__":
    main()
