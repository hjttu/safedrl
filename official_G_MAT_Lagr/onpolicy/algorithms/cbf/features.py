import numpy as np


CBF_EDGE_FEATURE_DIM = 11
LARGE_TTC = 1.0e3


def entity_type_id(entity) -> int:
    if "agent" in entity.name:
        return 0
    if "target" in entity.name:
        return 1
    if "dynamic_obstacle" in entity.name:
        return 3
    if "obstacle" in entity.name:
        return 2
    return -1


def build_cbf_edge_matrix(world, d_safe_agent: float, d_safe_obstacle: float) -> np.ndarray:
    """Return [N,N,1+CBF_EDGE_FEATURE_DIM] edge tensor, preserving distance first."""
    entities = world.entities
    n_entities = len(entities)
    edge = np.zeros((n_entities, n_entities, 1 + CBF_EDGE_FEATURE_DIM), dtype=np.float32)
    dists = world.cached_dist_mag
    edge[..., 0] = dists

    for i, ent_i in enumerate(entities):
        p_i = np.asarray(ent_i.state.p_pos, dtype=np.float32)
        v_i = np.asarray(ent_i.state.p_vel, dtype=np.float32)
        for j, ent_j in enumerate(entities):
            if i == j:
                continue
            p_j = np.asarray(ent_j.state.p_pos, dtype=np.float32)
            v_j = np.asarray(ent_j.state.p_vel, dtype=np.float32)
            rel_pos = p_j - p_i
            rel_vel = v_j - v_i
            dist = float(np.linalg.norm(rel_pos))
            is_agent_pair = "agent" in ent_i.name and "agent" in ent_j.name
            d_safe = d_safe_agent if is_agent_pair else d_safe_obstacle
            diff_p = p_i - p_j
            diff_v = v_i - v_j
            h_ij = float(np.dot(diff_p, diff_p) - d_safe * d_safe)
            hdot_ij = float(2.0 * np.dot(diff_p, diff_v))
            closing_speed = float(-np.dot(rel_pos, rel_vel) / (dist + 1e-6))
            ttc = dist / closing_speed if closing_speed > 1e-6 else LARGE_TTC
            edge[i, j, 1:] = np.array(
                [
                    rel_pos[0],
                    rel_pos[1],
                    rel_vel[0],
                    rel_vel[1],
                    dist,
                    h_ij,
                    hdot_ij,
                    min(ttc, LARGE_TTC),
                    float(entity_type_id(ent_j)),
                    float(d_safe),
                    1.0 if is_agent_pair else 0.0,
                ],
                dtype=np.float32,
            )
    return edge


def risk_from_cbf(h: float, hdot: float, ttc: float, d_safe: float) -> float:
    h_scale = max(d_safe * d_safe, 1e-6)
    h_risk = 1.0 / (1.0 + np.exp(4.0 * h / h_scale))
    closing_risk = 1.0 / (1.0 + np.exp(2.0 * hdot / h_scale))
    ttc_risk = np.exp(-max(ttc, 0.0) / 2.0)
    return float(np.clip(0.45 * h_risk + 0.35 * closing_risk + 0.20 * ttc_risk, 0.0, 1.0))


def discrete_actions_to_accel(actions: np.ndarray, n_bins: int = 20) -> np.ndarray:
    mapping = np.linspace(-1.0, 1.0, n_bins, dtype=np.float32)
    actions = np.asarray(actions)
    return np.stack([mapping[actions[..., 0].astype(int)], mapping[actions[..., 1].astype(int)]], axis=-1)


def accel_to_multidiscrete_action(accel: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """Encode continuous normalized acceleration as one-hot MultiDiscrete env action."""
    mapping = np.linspace(-1.0, 1.0, n_bins, dtype=np.float32)
    accel = np.clip(np.asarray(accel, dtype=np.float32), -1.0, 1.0)
    idx = np.abs(accel[..., None] - mapping).argmin(axis=-1)
    out_shape = accel.shape[:-1] + (2 * n_bins,)
    one_hot = np.zeros(out_shape, dtype=np.float32)
    for axis in range(2):
        flat = one_hot[..., axis * n_bins : (axis + 1) * n_bins].reshape(-1, n_bins)
        flat[np.arange(flat.shape[0]), idx[..., axis].reshape(-1)] = 1.0
    return one_hot
