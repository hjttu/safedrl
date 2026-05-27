from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from scipy.optimize import minimize


@dataclass
class HOCBFParams:
    k1: float = 2.0
    k2: float = 2.0
    slack_penalty_base: float = 100.0
    slack_penalty_priority: float = 500.0
    u_min: float = -1.0
    u_max: float = 1.0
    eval_hard_filter: bool = True
    hard_h_threshold: float = 0.02


class HOCBFSafetyFilter:
    """Small scipy-based HOCBF-QP filter for 2-D second-order UAV dynamics."""

    def __init__(self, params: HOCBFParams | None = None) -> None:
        self.params = params or HOCBFParams()
        self.last_actions: Dict[tuple[int, int], np.ndarray] = {}

    def filter_env(
        self,
        env_id: int,
        node_obs: np.ndarray,
        u_rl: np.ndarray,
        priorities: np.ndarray,
        gammas: np.ndarray,
        train: bool = True,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        node_obs = np.asarray(node_obs, dtype=np.float32)
        u_rl = np.asarray(u_rl, dtype=np.float32)
        n_agents = u_rl.shape[0]
        u_safe = np.zeros_like(u_rl)
        infos = []
        for i in range(n_agents):
            ui, info = self.filter_agent(env_id, i, node_obs, u_rl, priorities, gammas, train)
            u_safe[i] = ui
            infos.append(info)
            self.last_actions[(env_id, i)] = ui.copy()
        diag = {
            "qp_infeasible": float(np.mean([x["qp_infeasible"] for x in infos])),
            "slack": float(np.mean([x["slack"] for x in infos])),
            "active_constraints": float(np.sum([x["active_constraints"] for x in infos])),
            "intervention_norm": float(np.mean(np.linalg.norm(u_safe - u_rl, axis=-1))),
        }
        return u_safe, diag

    def filter_agent(
        self,
        env_id: int,
        i: int,
        node_obs: np.ndarray,
        u_rl: np.ndarray,
        priorities: np.ndarray,
        gammas: np.ndarray,
        train: bool,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        p = self.params
        p_i, v_i = node_obs[i, 0:2], node_obs[i, 2:4]
        constraints = []
        for j in range(node_obs.shape[0]):
            if i == j:
                continue
            entity_type = int(round(node_obs[j, -1]))
            if entity_type not in (0, 2, 3):
                continue
            edge = priorities[i, j]
            if edge <= 0.0 and gammas[i, j] <= 0.0:
                continue
            p_j, v_j = node_obs[j, 0:2], node_obs[j, 2:4]
            d_safe = max(float(node_obs[i, 4] + node_obs[j, 4]), 1e-3)
            diff_p = p_i - p_j
            diff_v = v_i - v_j
            h = float(np.dot(diff_p, diff_p) - d_safe * d_safe)
            if h > 1.5 and np.linalg.norm(diff_p) > 1.2:
                continue
            hdot = float(2.0 * np.dot(diff_p, diff_v))
            gamma = float(np.clip(gammas[i, j], 0.0, 1.0))
            u_j_hat = u_rl[j] if j < u_rl.shape[0] else np.zeros(2, dtype=np.float32)
            constraints.append((diff_p, diff_v, h, hdot, u_j_hat, gamma, float(edge)))

        n_c = len(constraints)
        if n_c == 0:
            return np.clip(u_rl[i], p.u_min, p.u_max), {"qp_infeasible": 0.0, "slack": 0.0, "active_constraints": 0}

        def unpack(x):
            return x[:2], x[2:]

        def objective(x):
            u, s = unpack(x)
            val = float(np.sum((u - u_rl[i]) ** 2))
            for idx, item in enumerate(constraints):
                weight = p.slack_penalty_base + p.slack_penalty_priority * item[-1]
                if (not train) and p.eval_hard_filter and item[2] < p.hard_h_threshold:
                    weight *= 100.0
                val += weight * s[idx] * s[idx]
            return val

        cons = []
        for idx, (diff_p, diff_v, h, hdot, u_j_hat, gamma, _priority) in enumerate(constraints):
            def hocbf_fun(x, idx=idx, diff_p=diff_p, diff_v=diff_v, h=h, hdot=hdot, u_j_hat=u_j_hat, gamma=gamma):
                u, s = unpack(x)
                hddot = 2.0 * np.dot(diff_v, diff_v) + 2.0 * np.dot(diff_p, gamma * u - (1.0 - gamma) * u_j_hat)
                return hddot + (p.k1 + p.k2) * hdot + p.k1 * p.k2 * h + s[idx]
            cons.append({"type": "ineq", "fun": hocbf_fun})

        x0 = np.concatenate([np.clip(u_rl[i], p.u_min, p.u_max), np.full(n_c, 1e-3, dtype=np.float32)])
        bounds = [(p.u_min, p.u_max), (p.u_min, p.u_max)] + [(0.0, None)] * n_c
        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 50, "ftol": 1e-6, "disp": False})
        if not result.success:
            return np.clip(u_rl[i], p.u_min, p.u_max), {"qp_infeasible": 1.0, "slack": 0.0, "active_constraints": n_c}
        u, s = unpack(result.x)
        return np.clip(u, p.u_min, p.u_max), {"qp_infeasible": 0.0, "slack": float(np.mean(s) if len(s) else 0.0), "active_constraints": n_c}
