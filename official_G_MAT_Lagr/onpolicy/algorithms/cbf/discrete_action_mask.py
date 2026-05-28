from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .features import risk_from_cbf


@dataclass
class CBFActionMaskConfig:
    n_x: int = 20
    n_y: int = 20
    k1: float = 2.0
    k2: float = 2.0
    h_keep: float = 0.05
    tau_ttc: float = 1.0
    d_safe_agent: float = 0.25
    d_safe_obstacle: float = 0.30
    lambda_soft_mask: float = 1.0
    action_mask_hard: bool = True
    action_mask_soft_penalty: bool = True
    neighbor_action_mode: Literal["zero", "last", "nominal"] = "zero"
    empty_mask_fallback: Literal["min_violation", "brake"] = "min_violation"


class CBFDiscreteActionMask:
    """Build a CBF safety mask over the 2-D discrete joint action grid."""

    def __init__(self, cfg: CBFActionMaskConfig):
        self.cfg = cfg
        ax = np.linspace(-1.0, 1.0, cfg.n_x, dtype=np.float32)
        ay = np.linspace(-1.0, 1.0, cfg.n_y, dtype=np.float32)
        xx, yy = np.meshgrid(ax, ay, indexing="ij")
        self.candidates_norm = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)
        self.num_joint_actions = self.candidates_norm.shape[0]

    def build_batch(
        self,
        node_obs: np.ndarray,
        adj: np.ndarray,
        agent_max_accel: np.ndarray | float,
        last_actions_norm: np.ndarray | None = None,
        nominal_actions_norm: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return mask, soft risk penalty, diagnostics for [env, agent]."""
        n_envs, n_agents = node_obs.shape[:2]
        masks = np.ones((n_envs, n_agents, self.num_joint_actions), dtype=np.float32)
        penalties = np.zeros_like(masks)
        diag = np.zeros((n_envs, n_agents, 8), dtype=np.float32)
        max_accels = np.asarray(agent_max_accel, dtype=np.float32)
        if max_accels.ndim == 0:
            max_accels = np.full((n_envs, n_agents), float(max_accels), dtype=np.float32)

        for env_i in range(n_envs):
            for agent_i in range(n_agents):
                graph_node_obs = node_obs[env_i, agent_i]
                graph_adj = adj[env_i, agent_i]
                mask, penalty, info = self.build_agent(
                    graph_node_obs,
                    graph_adj,
                    agent_i,
                    float(max_accels[env_i, agent_i]),
                    None if last_actions_norm is None else last_actions_norm[env_i],
                    None if nominal_actions_norm is None else nominal_actions_norm[env_i],
                )
                masks[env_i, agent_i] = mask
                penalties[env_i, agent_i] = penalty
                diag[env_i, agent_i] = info
        return masks, penalties, diag

    def build_agent(
        self,
        node_obs: np.ndarray,
        adj: np.ndarray,
        agent_i: int,
        max_accel_i: float,
        last_actions_norm: np.ndarray | None = None,
        nominal_actions_norm: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        candidates_acc = self.candidates_norm * max_accel_i
        mask = np.ones(self.num_joint_actions, dtype=np.float32)
        soft_penalty = np.zeros(self.num_joint_actions, dtype=np.float32)
        total_violation = np.zeros(self.num_joint_actions, dtype=np.float32)

        p_i = node_obs[agent_i, 0:2]
        v_i = node_obs[agent_i, 2:4]
        min_phi = np.inf
        hard_edges = 0
        soft_edges = 0

        edge_tensor = adj if adj.ndim == 3 else None
        for j in range(node_obs.shape[0]):
            if j == agent_i:
                continue
            entity_type = int(round(node_obs[j, -1]))
            if entity_type not in (0, 2, 3):
                continue
            dist = float(edge_tensor[agent_i, j, 0] if edge_tensor is not None else np.linalg.norm(p_i - node_obs[j, 0:2]))
            if dist <= 0.0:
                continue
            p_j = node_obs[j, 0:2]
            v_j = node_obs[j, 2:4] if entity_type in (0, 3) else np.zeros(2, dtype=np.float32)
            is_agent_pair = entity_type == 0 and j < last_actions_norm.shape[0] if last_actions_norm is not None else entity_type == 0
            d_safe = self.cfg.d_safe_agent if is_agent_pair else self.cfg.d_safe_obstacle
            diff_p = p_i - p_j
            diff_v = v_i - v_j
            h = float(np.dot(diff_p, diff_p) - d_safe * d_safe)
            hdot = float(2.0 * np.dot(diff_p, diff_v))
            closing_speed = float(-np.dot(p_j - p_i, v_j - v_i) / (dist + 1e-6))
            ttc = dist / closing_speed if closing_speed > 1e-6 else 1e3
            risk = risk_from_cbf(h, hdot, ttc, d_safe)
            a_j = self._neighbor_accel(j, entity_type, max_accel_i, last_actions_norm, nominal_actions_norm)
            hddot = 2.0 * np.dot(diff_v, diff_v) + 2.0 * np.matmul(candidates_acc - a_j, diff_p)
            phi = hddot + (self.cfg.k1 + self.cfg.k2) * hdot + self.cfg.k1 * self.cfg.k2 * h
            min_phi = min(min_phi, float(np.min(phi)))
            violation = np.maximum(-phi, 0.0).astype(np.float32)

            is_hard = h < self.cfg.h_keep or ttc < self.cfg.tau_ttc
            if is_hard and self.cfg.action_mask_hard:
                hard_edges += 1
                mask[phi < 0.0] = 0.0
            else:
                soft_edges += 1
                soft_penalty += risk * violation
            total_violation += violation

        empty_mask = float(mask.sum() <= 0.0)
        fallback = 0.0
        if empty_mask:
            fallback = 1.0
            mask[:] = 0.0
            if self.cfg.empty_mask_fallback == "brake":
                idx = int(np.argmin(np.linalg.norm(self.candidates_norm, axis=-1)))
            else:
                idx = int(np.argmin(total_violation))
            mask[idx] = 1.0
        if not np.isfinite(min_phi):
            min_phi = 0.0
        safe_count = float(mask.sum())
        entropy = float(np.log(max(safe_count, 1.0)))
        info = np.array(
            [
                safe_count / self.num_joint_actions,
                empty_mask,
                safe_count,
                safe_count,
                entropy,
                min_phi,
                float(np.mean(soft_penalty)),
                fallback,
            ],
            dtype=np.float32,
        )
        return mask, soft_penalty.astype(np.float32), info

    def _neighbor_accel(
        self,
        j: int,
        entity_type: int,
        max_accel_i: float,
        last_actions_norm: np.ndarray | None,
        nominal_actions_norm: np.ndarray | None,
    ) -> np.ndarray:
        if entity_type != 0:
            return np.zeros(2, dtype=np.float32)
        if self.cfg.neighbor_action_mode == "last" and last_actions_norm is not None and j < last_actions_norm.shape[0]:
            return last_actions_norm[j].astype(np.float32) * max_accel_i
        if self.cfg.neighbor_action_mode == "nominal" and nominal_actions_norm is not None and j < nominal_actions_norm.shape[0]:
            return nominal_actions_norm[j].astype(np.float32) * max_accel_i
        return np.zeros(2, dtype=np.float32)
