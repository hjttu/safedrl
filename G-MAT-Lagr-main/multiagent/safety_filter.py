import numpy as np

from multiagent.guide_policy import limit_action_inf_norm


EPS = 1e-8


def _arg(args, name, default):
    return getattr(args, name, default)


def _vec2(value):
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 2:
        out = np.zeros(2, dtype=np.float32)
        out[:arr.size] = arr
        return out
    return arr[:2].astype(np.float32)


def _speed(entity):
    if not hasattr(entity, "state") or entity.state.p_vel is None:
        return np.zeros(2, dtype=np.float32)
    return _vec2(entity.state.p_vel)


def _radius(entity):
    return float(getattr(entity, "R", getattr(entity, "size", 0.0)))


def _entity_pos(entity):
    return _vec2(entity.state.p_pos)


def _dt(agent, args):
    world = getattr(agent, "_pgfs_world", getattr(agent, "world", None))
    return float(getattr(world, "dt", _arg(args, "shield_dt", 0.1)))


def _sigmoid(x):
    x = float(np.clip(x, -60.0, 60.0))
    return 1.0 / (1.0 + np.exp(-x))


def get_entities(world):
    return list(world.egos) + list(world.dynamic_obstacles) + list(world.obstacles)


def get_neighbors(agent, world):
    entities = get_entities(world)
    agent_gid = getattr(agent, "global_id", None)

    if getattr(world, "edge_list", None) is None or agent_gid is None:
        return [e for e in entities if e is not agent]

    edge_list = np.asarray(world.edge_list)
    if edge_list.ndim != 2 or edge_list.shape[0] < 2:
        return [e for e in entities if e is not agent]

    neighbor_ids = set()
    for src, dst in edge_list[:2].T:
        if int(src) == int(agent_gid):
            neighbor_ids.add(int(dst))

    return [
        e
        for e in entities
        if e is not agent
        and getattr(e, "global_id", None) is not None
        and int(e.global_id) in neighbor_ids
    ]


def get_goal_pos(agent, world):
    goal = getattr(agent, "goal", None)
    if goal is not None:
        return _vec2(goal)
    return _vec2(world.targets[agent.id].state.p_pos)


def progress_action(agent, world):
    direction = get_goal_pos(agent, world) - _entity_pos(agent)
    norm = float(np.linalg.norm(direction))
    if norm < EPS:
        return np.zeros(2, dtype=np.float32)
    return limit_action_inf_norm(direction / norm, 1.0)


def risk_score(agent, world, neighbors):
    args = getattr(world, "_pgfs_args", getattr(world, "args", None))
    margin_extra = _arg(args, "shield_risk_margin", None)
    if margin_extra is None:
        margin_extra = 0.1
    decay = _arg(args, "risk_decay", None)
    if decay is None:
        decay = 3.0

    risk = 0.0
    p_i = _entity_pos(agent)
    for other in neighbors:
        dist = float(np.linalg.norm(p_i - _entity_pos(other)))
        safe_r = _radius(agent) + _radius(other) + float(margin_extra)
        margin = dist - safe_r
        risk += float(np.exp(-float(decay) * max(margin, 0.0)))
    return float(risk)


def update_priority(agent, world, risk, args):
    dist_to_goal = float(np.linalg.norm(_entity_pos(agent) - get_goal_pos(agent, world)))
    waiting_time = float(getattr(agent, "waiting_time", 0.0))
    deadlock_score = float(getattr(agent, "deadlock_score", 0.0))

    priority = (
        _arg(args, "priority_wait_coef", 1.0) * waiting_time
        - _arg(args, "priority_dist_coef", 0.2) * dist_to_goal
        + _arg(args, "priority_deadlock_coef", 1.0) * deadlock_score
        - _arg(args, "priority_risk_coef", 0.2) * float(risk)
        + 0.01 * float(getattr(agent, "id", 0))
    )
    agent.priority = float(priority)
    return agent.priority


def discrete_cbf_safe(agent, other, action, args):
    dt = _dt(agent, args)
    alpha = float(_arg(args, "cbf_alpha", 1.0))
    p_i = _entity_pos(agent)
    p_j = _entity_pos(other)
    v_i = _speed(agent)
    v_j = _speed(other)
    action = _vec2(action)

    base_margin = float(_arg(args, "cbf_safe_margin", 0.0))
    yield_margin = float(_arg(args, "cbf_yield_margin", 0.1))
    pass_margin = float(_arg(args, "cbf_pass_margin", 0.0))
    asym_margin = float(_arg(args, "asymmetric_yield_margin", 0.0))

    safe_r = _radius(agent) + _radius(other) + base_margin + asym_margin
    other_priority = getattr(other, "priority", None)
    agent_priority = getattr(agent, "priority", None)
    if other_priority is not None and agent_priority is not None and agent_priority < other_priority:
        safe_r += yield_margin
    else:
        safe_r += pass_margin

    p_i_next = p_i + v_i * dt + 0.5 * action * dt * dt
    p_j_next = p_j + v_j * dt

    h_now = float(np.dot(p_i - p_j, p_i - p_j) - safe_r * safe_r)
    h_next = float(np.dot(p_i_next - p_j_next, p_i_next - p_j_next) - safe_r * safe_r)
    return bool(h_next - (1.0 - alpha * dt) * h_now >= -EPS)


def predict_progress(agent, world, action):
    dt = float(getattr(world, "dt", 0.1))
    p_now = _entity_pos(agent)
    goal = get_goal_pos(agent, world)
    p_next = p_now + _speed(agent) * dt + 0.5 * _vec2(action) * dt * dt
    return float(np.linalg.norm(goal - p_now) - np.linalg.norm(goal - p_next))


def tangent_escape_action(agent, world, neighbors, args):
    if not neighbors:
        return progress_action(agent, world)

    p_i = _entity_pos(agent)
    nearest = None
    nearest_margin = np.inf
    for other in neighbors:
        rel = p_i - _entity_pos(other)
        margin = float(np.linalg.norm(rel) - (_radius(agent) + _radius(other)))
        if margin < nearest_margin:
            nearest = other
            nearest_margin = margin

    if nearest is None:
        return progress_action(agent, world)

    rel = p_i - _entity_pos(nearest)
    norm = float(np.linalg.norm(rel))
    if norm < EPS:
        parity_sign = 1.0 if int(getattr(agent, "id", 0)) % 2 == 0 else -1.0
        return np.array([parity_sign, 0.0], dtype=np.float32)

    tangent = np.array([-rel[1], rel[0]], dtype=np.float32) / norm
    parity_sign = 1.0 if int(getattr(agent, "id", 0)) % 2 == 0 else -1.0
    other_priority = getattr(nearest, "priority", None)
    agent_priority = getattr(agent, "priority", None)
    if other_priority is not None and agent_priority is not None:
        sign = parity_sign if agent_priority < other_priority else -parity_sign
    else:
        sign = parity_sign
    return limit_action_inf_norm(sign * tangent, 1.0)


def update_deadlock_state(agent, world, filter_info, args):
    dist = float(np.linalg.norm(get_goal_pos(agent, world) - _entity_pos(agent)))
    speed = float(np.linalg.norm(_speed(agent)))
    last_goal_dist = getattr(agent, "last_goal_dist", dist)
    if last_goal_dist is None:
        last_goal_dist = dist
    progress = float(last_goal_dist - dist)

    stuck_count = int(getattr(agent, "stuck_count", 0))
    waiting_time = float(getattr(agent, "waiting_time", 0.0))
    cbf_active_count = int(getattr(agent, "cbf_active_count", 0))
    shield_total_count = int(getattr(agent, "shield_total_count", 0))
    shield_intervention_count = int(getattr(agent, "shield_intervention_count", 0))
    mask_tightness_sum = float(getattr(agent, "mask_tightness_sum", 0.0))
    fallback_count = int(getattr(agent, "fallback_count", 0))
    deadlock_duration = int(getattr(agent, "deadlock_duration", 0))

    if progress < _arg(args, "stuck_progress_eps", 1e-3) and speed < _arg(args, "stuck_speed_eps", 5e-2):
        stuck_count += 1
        waiting_time += float(getattr(world, "dt", _arg(args, "shield_dt", 0.1)))
    else:
        stuck_count = max(0, stuck_count - 1)
        waiting_time = max(0.0, waiting_time * _arg(args, "waiting_time_decay", 0.8))

    intervened = bool(filter_info.get("intervened", False))
    if intervened:
        cbf_active_count += 1
        shield_intervention_count += 1

    shield_total_count += 1
    mask_tightness = float(filter_info.get("mask_tightness", 0.0))
    mask_tightness_sum += mask_tightness
    if bool(filter_info.get("fallback", False)):
        fallback_count += 1

    window = max(float(_arg(args, "deadlock_window", 20.0)), 1.0)
    deadlock_score = (
        0.5 * min(stuck_count / window, 1.0)
        + 0.3 * min(cbf_active_count / window, 1.0)
        + 0.2 * mask_tightness
    )
    if deadlock_score > _arg(args, "deadlock_threshold", 0.5):
        deadlock_duration += 1

    agent.last_goal_dist = dist
    agent.stuck_count = stuck_count
    agent.waiting_time = waiting_time
    agent.cbf_active_count = cbf_active_count
    agent.deadlock_score = float(deadlock_score)
    agent.deadlock_duration = deadlock_duration
    agent.shield_total_count = shield_total_count
    agent.shield_intervention_count = shield_intervention_count
    agent.mask_tightness_sum = mask_tightness_sum
    agent.fallback_count = fallback_count
    return agent.deadlock_score


def adaptive_mix_ratio(agent, world, risk, deadlock_score, args):
    rho_risk = _sigmoid(_arg(args, "risk_gain", 4.0) * (float(risk) - _arg(args, "risk_threshold", 1.0)))
    rho_deadlock = _sigmoid(
        _arg(args, "deadlock_gain", 8.0)
        * (float(deadlock_score) - _arg(args, "deadlock_threshold", 0.5))
    )
    rho = (
        _arg(args, "rho_base", 0.1)
        + _arg(args, "rho_risk_coef", 0.4) * rho_risk
        + _arg(args, "rho_deadlock_coef", 0.4) * rho_deadlock
    )
    if hasattr(world, "CL_ratio") and hasattr(args, "rho_curriculum_coef"):
        rho += float(args.rho_curriculum_coef) * float(world.CL_ratio)
    return float(np.clip(rho, 0.0, 1.0))


def pg_action_mask(agent, world, a_ref, a_escape, neighbors, deadlock_score, args):
    action_mapping = np.linspace(-1.0, 1.0, 20, dtype=np.float32)
    a_ref = _vec2(a_ref)
    a_escape = _vec2(a_escape)
    total = int(action_mapping.size * action_mapping.size)
    safe_num = 0
    best_action = None
    best_score = np.inf
    in_deadlock = float(deadlock_score) >= _arg(args, "deadlock_threshold", 0.5)
    progress_tolerance = float(_arg(args, "progress_tolerance", 0.0))
    escape_weight_active = float(_arg(args, "escape_weight", 0.4)) if in_deadlock else 0.0

    for ax in action_mapping:
        for ay in action_mapping:
            candidate = np.array([ax, ay], dtype=np.float32)
            if not all(discrete_cbf_safe(agent, other, candidate, args) for other in neighbors):
                continue
            progress = predict_progress(agent, world, candidate)
            if not in_deadlock and progress < -progress_tolerance:
                continue

            safe_num += 1
            score = (
                _arg(args, "action_track_weight", 1.0) * float(np.sum((candidate - a_ref) ** 2))
                + escape_weight_active * float(np.sum((candidate - a_escape) ** 2))
                - _arg(args, "progress_score_weight", 0.2) * progress
            )
            if score < best_score:
                best_score = score
                best_action = candidate

    if best_action is None:
        return None, {
            "intervened": True,
            "mask_tightness": 1.0,
            "fallback": True,
            "safe_action_num": 0,
            "total_action_num": total,
        }

    mask_tightness = 1.0 - float(safe_num) / float(total)
    return best_action, {
        "intervened": bool(np.linalg.norm(best_action - a_ref) > _arg(args, "intervention_eps", 1e-3)),
        "mask_tightness": float(mask_tightness),
        "fallback": False,
        "safe_action_num": int(safe_num),
        "total_action_num": total,
    }


def fallback_action(agent, world, a_escape, args):
    action = (
        _arg(args, "fallback_escape_coef", 1.0) * _vec2(a_escape)
        - _arg(args, "fallback_damping_coef", 0.5) * _speed(agent)
    )
    return limit_action_inf_norm(action, 1.0)


def compute_soft_action_bias(agent, world, args):
    """
    Return logits_bias with shape (2, 20) for MultiDiscrete [ax, ay].
    Negative values reduce logits. Zero means no change.
    """
    action_mapping = np.linspace(-1.0, 1.0, 20, dtype=np.float32)
    neighbors = get_neighbors(agent, world)
    risk = risk_score(agent, world, neighbors)
    update_priority(agent, world, risk, args)

    dt = float(getattr(world, "dt", _arg(args, "shield_dt", 0.1)))
    alpha = float(_arg(args, "cbf_alpha", 1.0))
    progress_tolerance = float(_arg(args, "progress_tolerance", 0.0))
    safety_coef = float(_arg(args, "soft_safety_risk_coef", 1.0))
    progress_coef = float(_arg(args, "soft_progress_risk_coef", 0.2))
    deadlock_coef = float(_arg(args, "soft_deadlock_risk_coef", 0.1))
    current_deadlock_score = float(getattr(agent, "deadlock_score", 0.0))

    p_i = _entity_pos(agent)
    v_i = _speed(agent)
    joint_risk = np.zeros((20, 20), dtype=np.float32)

    for k, ax in enumerate(action_mapping):
        for l, ay in enumerate(action_mapping):
            action = np.array([ax, ay], dtype=np.float32)
            safety_violation = 0.0
            p_i_next = p_i + v_i * dt + 0.5 * action * dt * dt

            for other in neighbors:
                p_j = _entity_pos(other)
                v_j = _speed(other)
                base_margin = float(_arg(args, "cbf_safe_margin", 0.0))
                yield_margin = float(_arg(args, "cbf_yield_margin", 0.1))
                pass_margin = float(_arg(args, "cbf_pass_margin", 0.0))
                asym_margin = float(_arg(args, "asymmetric_yield_margin", 0.0))

                safe_r = _radius(agent) + _radius(other) + base_margin + asym_margin
                other_priority = getattr(other, "priority", None)
                agent_priority = getattr(agent, "priority", None)
                if other_priority is not None and agent_priority is not None and agent_priority < other_priority:
                    safe_r += yield_margin
                else:
                    safe_r += pass_margin

                p_j_next = p_j + v_j * dt
                h_now = float(np.dot(p_i - p_j, p_i - p_j) - safe_r * safe_r)
                h_next = float(np.dot(p_i_next - p_j_next, p_i_next - p_j_next) - safe_r * safe_r)
                cbf_value = h_next - (1.0 - alpha * dt) * h_now
                violation = max(0.0, -cbf_value)
                scale = max(safe_r * safe_r, EPS)
                safety_violation += violation / scale

            progress = predict_progress(agent, world, action)
            progress_risk = max(0.0, -progress - progress_tolerance)
            joint_risk[k, l] = (
                safety_coef * safety_violation
                + progress_coef * progress_risk
                + deadlock_coef * current_deadlock_score
            )

    beta = float(_arg(args, "soft_axis_lme_beta", 5.0))
    beta = beta if abs(beta) > EPS else 1.0
    max_x = np.max(beta * joint_risk, axis=1, keepdims=True)
    risk_x = (np.log(np.mean(np.exp(beta * joint_risk - max_x), axis=1)) + max_x[:, 0]) / beta
    max_y = np.max(beta * joint_risk, axis=0, keepdims=True)
    risk_y = (np.log(np.mean(np.exp(beta * joint_risk - max_y), axis=0)) + max_y[0, :]) / beta

    threshold = float(_arg(args, "soft_risk_threshold", 0.05))
    scale = float(_arg(args, "soft_mask_scale", 1.0))
    temperature = float(_arg(args, "soft_mask_temperature", 3.0))
    max_bias = abs(float(_arg(args, "soft_mask_max_bias", 6.0)))

    def risk_to_bias(axis_risk):
        excess = np.maximum(axis_risk - threshold, 0.0)
        bias = -scale * np.expm1(temperature * excess)
        return np.clip(bias, -max_bias, 0.0)

    logits_bias = np.stack([risk_to_bias(risk_x), risk_to_bias(risk_y)], axis=0)
    logits_bias = np.nan_to_num(logits_bias, nan=0.0, posinf=0.0, neginf=-max_bias).astype(np.float32)

    agent.soft_bias_mean = float(np.mean(np.abs(logits_bias)))
    agent.soft_bias_max = float(np.max(np.abs(logits_bias)))
    agent.soft_danger_ratio = float(np.mean(logits_bias < -1e-6))
    if not hasattr(agent, "shield_info") or agent.shield_info is None:
        agent.shield_info = {}
    agent.shield_info["soft_bias_mean"] = agent.soft_bias_mean
    agent.shield_info["soft_bias_max"] = agent.soft_bias_max
    agent.shield_info["soft_danger_ratio"] = agent.soft_danger_ratio

    return logits_bias


def pg_fs_shield(agent, world, a_rl, a_guide, args):
    world._pgfs_args = args
    agent._pgfs_world = world
    neighbors = get_neighbors(agent, world)
    risk = risk_score(agent, world, neighbors)
    priority = update_priority(agent, world, risk, args)
    old_deadlock = float(getattr(agent, "deadlock_score", 0.0))

    a_rl = _vec2(a_rl)
    a_guide = _vec2(a_guide)
    a_progress = progress_action(agent, world)
    a_escape = tangent_escape_action(agent, world, neighbors, args)
    rho = adaptive_mix_ratio(agent, world, risk, old_deadlock, args)

    guide_ref = _arg(args, "guide_weight", 1.0) * a_guide + _arg(args, "progress_ref_weight", 0.2) * a_progress
    a_ref = (1.0 - rho) * a_rl + rho * guide_ref

    if old_deadlock >= _arg(args, "deadlock_threshold", 0.5):
        escape_ratio = float(np.clip(_arg(args, "escape_ref_ratio", 0.5), 0.0, 1.0))
        a_ref = (1.0 - escape_ratio) * a_ref + escape_ratio * a_escape
    a_ref = limit_action_inf_norm(a_ref, 1.0)

    if _arg(args, "shield_type", "mask") == "mask":
        a_exec, info = pg_action_mask(agent, world, a_ref, a_escape, neighbors, old_deadlock, args)
    else:
        a_exec = a_ref
        info = {
            "intervened": bool(np.linalg.norm(a_exec - a_rl) > _arg(args, "intervention_eps", 1e-3)),
            "mask_tightness": 0.0,
            "fallback": False,
            "safe_action_num": 400,
            "total_action_num": 400,
        }

    if a_exec is None:
        a_exec = fallback_action(agent, world, a_escape, args)
        info["intervened"] = True
        info["fallback"] = True

    deadlock_score = update_deadlock_state(agent, world, info, args)
    info.update(
        {
            "risk": float(risk),
            "deadlock_score": float(deadlock_score),
            "rho": float(rho),
            "priority": float(priority),
            "a_ref": _vec2(a_ref),
            "a_rl": _vec2(a_rl),
            "a_guide": _vec2(a_guide),
            "a_escape": _vec2(a_escape),
        }
    )
    return limit_action_inf_norm(a_exec, 1.0), info
