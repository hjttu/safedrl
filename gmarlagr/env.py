from __future__ import annotations

from dataclasses import dataclass

import numpy as np


AGENT = 0
TARGET = 1
OBSTACLE = 2


@dataclass
class EnvConfig:
    n_agents: int = 3
    episode_len: int = 100
    dt: float = 0.1
    sensing_radius: float = 1.0
    vmax: float = 1.0
    umax: float = 0.5
    agent_radius: float = 0.1
    obstacle_radius: float = 0.15
    target_radius: float = 0.1
    action_bins: int = 20
    seed: int | None = 0


class MultiUAVNavEnv:
    """2-D multi-UAV navigation task from the G-MATrans-Lagr paper.

    Each UAV follows second-order integrator dynamics:
    p_dot = v, v_dot = u.  Observations are variable-size local graphs whose
    nodes are visible agents, the agent's own target, and visible obstacles.
    """

    def __init__(self, config: EnvConfig | None = None):
        self.cfg = config or EnvConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.world_size = 4.0 * np.sqrt(self.cfg.n_agents / 3.0)
        self.actions = self._build_discrete_actions()
        self.t = 0
        self.pos = np.zeros((self.cfg.n_agents, 2), dtype=np.float32)
        self.vel = np.zeros_like(self.pos)
        self.targets = np.zeros_like(self.pos)
        self.obstacles = np.zeros_like(self.pos)
        self.done_agents = np.zeros(self.cfg.n_agents, dtype=bool)

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    def reset(self, seed: int | None = None) -> list[dict[str, np.ndarray]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.done_agents[:] = False
        self.pos = self._sample_points(self.cfg.n_agents, self.cfg.agent_radius)
        self.vel = np.zeros_like(self.pos)
        self.targets = self._sample_points(self.cfg.n_agents, self.cfg.target_radius)
        self.obstacles = self._sample_points(self.cfg.n_agents, self.cfg.obstacle_radius)
        return self.observe()

    def step(self, action_idx: np.ndarray):
        action_idx = np.asarray(action_idx, dtype=np.int64)
        accel = self.actions[action_idx]
        accel[self.done_agents] = 0.0
        self.vel = np.clip(self.vel + accel * self.cfg.dt, -self.cfg.vmax, self.cfg.vmax)
        self.pos = np.clip(self.pos + self.vel * self.cfg.dt, 0.0, self.world_size)
        self.t += 1

        dist_to_goal = np.linalg.norm(self.pos - self.targets, axis=-1)
        reached = dist_to_goal <= self.cfg.target_radius
        collisions = self._collision_mask()
        self.done_agents |= reached

        r_dist = 5.0 * (np.exp(-0.2 * dist_to_goal) - 1.0)
        r_goal = 10.0 * reached.astype(np.float32)
        r_col = -5.0 * collisions.astype(np.float32)
        reward = r_dist + r_goal + r_col
        cost = collisions.astype(np.float32)

        terminated = bool(self.done_agents.all())
        truncated = self.t >= self.cfg.episode_len
        info = {
            "reached": reached.copy(),
            "collisions": collisions.copy(),
            "min_dist": self._min_entity_distances(),
        }
        return self.observe(), reward.astype(np.float32), cost, terminated, truncated, info

    def observe(self) -> list[dict[str, np.ndarray]]:
        return [self._agent_graph(i) for i in range(self.cfg.n_agents)]

    def _build_discrete_actions(self) -> np.ndarray:
        values = np.linspace(-self.cfg.umax, self.cfg.umax, self.cfg.action_bins)
        ax, ay = np.meshgrid(values, values, indexing="ij")
        return np.stack([ax.ravel(), ay.ravel()], axis=-1).astype(np.float32)

    def _sample_points(self, n: int, radius: float) -> np.ndarray:
        lo, hi = radius, self.world_size - radius
        return self.rng.uniform(lo, hi, size=(n, 2)).astype(np.float32)

    def _agent_graph(self, i: int) -> dict[str, np.ndarray]:
        nodes: list[list[float]] = []
        center = self.pos[i]

        for j in range(self.cfg.n_agents):
            if j == i or np.linalg.norm(self.pos[j] - center) <= self.cfg.sensing_radius:
                nodes.append([*self.pos[j], *self.vel[j], self.cfg.agent_radius, AGENT])

        nodes.append([*self.targets[i], 0.0, 0.0, self.cfg.target_radius, TARGET])

        for obs in self.obstacles:
            if np.linalg.norm(obs - center) <= self.cfg.sensing_radius:
                nodes.append([*obs, 0.0, 0.0, self.cfg.obstacle_radius, OBSTACLE])

        x = np.asarray(nodes, dtype=np.float32)
        rel = x[:, :2] - x[:1, :2]
        d = np.linalg.norm(rel, axis=-1, keepdims=True).astype(np.float32)
        return {"nodes": x, "edge_dist": d, "self_state": self._self_state(i)}

    def _self_state(self, i: int) -> np.ndarray:
        return np.asarray(
            [*self.pos[i], *self.vel[i], *(self.targets[i] - self.pos[i]), float(self.done_agents[i])],
            dtype=np.float32,
        )

    def _collision_mask(self) -> np.ndarray:
        n = self.cfg.n_agents
        mask = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(self.pos[i] - self.pos[j]) <= 2.0 * self.cfg.agent_radius:
                    mask[i] = True
                    mask[j] = True
            d_obs = np.linalg.norm(self.obstacles - self.pos[i], axis=-1)
            if np.any(d_obs <= self.cfg.agent_radius + self.cfg.obstacle_radius):
                mask[i] = True
        return mask

    def _min_entity_distances(self) -> np.ndarray:
        mins = np.full(self.cfg.n_agents, np.inf, dtype=np.float32)
        for i in range(self.cfg.n_agents):
            others = [np.linalg.norm(self.pos[i] - self.pos[j]) for j in range(self.cfg.n_agents) if j != i]
            obs = np.linalg.norm(self.obstacles - self.pos[i], axis=-1).tolist()
            mins[i] = np.min(others + obs)
        return mins
