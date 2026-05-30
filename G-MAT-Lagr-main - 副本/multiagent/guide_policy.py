import numpy as np  
# guide_policy.py

delta_t = 0.1  # time step

def set_JS_curriculum(CL_ratio, gp_type):
    if "formation" in gp_type:
        func_ = 1-CL_ratio
    elif "encirclement" in gp_type:
        # k = 1.0
        # delta = 1-(np.exp(-k*(-1))-np.exp(k*(-1)))/(np.exp(-k*(-1))+np.exp(k*(-1)))
        # x = 2*CL_ratio-1
        # y_mid = (np.exp(-k*x)-np.exp(k*x))/(np.exp(-k*x)+np.exp(k*x))-delta*x**3
        # func_ = (y_mid+1)/2
        func_ = 1-CL_ratio
    elif "navigation" in gp_type:
        func_ = 1-CL_ratio
    return func_

def guide_policy(world, gp_type):
    """Factory function to select the appropriate policy based on the version"""
    if gp_type == "navigation_rvo":
        return guide_policy_navigation_rvo(world)
    elif gp_type == "navigation_cbf":
        return guide_policy_navigation_cbf(world)
    else:
        raise ValueError(f"Unknown policy version: {gp_type}")

def guide_policy_navigation(world):
    egos = world.egos
    dynamic_obstacles = world.dynamic_obstacles
    obstacles = world.obstacles
    targets = world.targets
    num_egos = len(egos)
    U = np.zeros((num_egos, 2, 1))

    edge_list = world.edge_list.tolist()
    edge_num = len(edge_list[1])  # each edge is calculated twice

    k1 = 0.4  # goal coefficient (0.5)
    k_obs = 2.5  # obstacle coefficient (1.5)
    k_b = 1.6  # damping coefficient

    for i, ego in enumerate(egos):
        # Get the neighbors of the ego
        neighbors_id = []  # the neighbor id of all entities, global id
        for j in range(edge_num):
            if int(edge_list[0][j]) == ego.global_id:
                neighbors_id.append(edge_list[1][j])
            if int(edge_list[0][j]) > ego.global_id:
                break
        
        f_goal = k1 * (targets[i].state.p_pos - ego.state.p_pos)

        # print("ego", i, "neighbors_id", neighbors_id)
        neighbors_ego = [e for e in egos if e.global_id in neighbors_id]
        neighbors_dobs = [d for d in dynamic_obstacles if d.global_id in neighbors_id]
        neighbors_obs = [o for o in obstacles if o.global_id in neighbors_id]

        f_obs = np.array([0., 0.])
        for nb_obs in neighbors_obs:
            d_ij = ego.state.p_pos - nb_obs.state.p_pos
            norm_d_ij = np.linalg.norm(d_ij)
            L_min = ego.R + nb_obs.R + nb_obs.delta
            Ls = L_min + 0.7
            if norm_d_ij < Ls:
                f_obs = f_obs + k_obs*(Ls-norm_d_ij)/norm_d_ij*d_ij

        f_dobs = np.array([0., 0.])
        for nb_dobs in neighbors_dobs:
            r_ij = ego.state.p_pos - nb_dobs.state.p_pos
            norm_r_ij = np.linalg.norm(r_ij)
            relative_velocity = ego.state.p_vel - nb_dobs.state.p_vel
            L_min = ego.R + nb_dobs.R + nb_dobs.delta
            Ls = L_min + 0.5  
            if norm_r_ij < Ls:
                relative_speed_in_r_dir = np.dot(relative_velocity, r_ij) / norm_r_ij
                if relative_speed_in_r_dir < 0:
                    f_dobs = f_dobs + k_obs * (Ls - norm_r_ij) / norm_r_ij * r_ij

        f_egos = np.array([0., 0.])
        for nb_ego in neighbors_ego:
            d_ij = ego.state.p_pos - nb_ego.state.p_pos
            norm_d_ij = np.linalg.norm(d_ij)
            L_min = ego.R + nb_ego.R
            Ls = L_min + 0.3
            if norm_d_ij < Ls:
                f_egos = f_egos + k_obs*(Ls-norm_d_ij)/norm_d_ij*d_ij

        u_i = f_goal + f_egos + f_obs + f_dobs - k_b*ego.state.p_vel

        u_i = limit_action_inf_norm(u_i, 1)

        U[i] = u_i.reshape(2,1)

    return U

def guide_policy_navigation_rvo(world):
    egos = world.egos
    dynamic_obstacles = world.dynamic_obstacles
    obstacles = world.obstacles
    num_egos = len(egos)
    U = np.zeros((num_egos, 2, 1))

    edge_list = world.edge_list.tolist()
    edge_num = len(edge_list[1])  # each edge is calculated twice

    k1 = 0.5  # goal coefficient
    k_b = 1.6  # damping coefficient

    for i, ego in enumerate(egos):
        # Get the neighbors of the ego
        neighbors_id = []  # the neighbor id of all entities, global id
        for j in range(edge_num):
            if int(edge_list[0][j]) == ego.global_id:
                neighbors_id.append(edge_list[1][j])
            if int(edge_list[0][j]) > ego.global_id:
                break

        f_goal = k1 * (ego.goal - ego.state.p_pos)

        # print("ego", i, "neighbors_id", neighbors_id)
        neighbors_ego = [e for e in egos if e.global_id in neighbors_id]
        neighbors_dobs = [d for d in dynamic_obstacles if d.global_id in neighbors_id]
        neighbors_obs = [o for o in obstacles if o.global_id in neighbors_id]

        u_i = f_goal - k_b * ego.state.p_vel
        k_a = 1
        k_v = 8
        vel_des = ego.state.p_vel + k_a * u_i
        v_i = RVO(ego, neighbors_ego, neighbors_dobs, neighbors_obs, vel_des)
        u_i = (v_i - ego.state.p_vel)/delta_t
        u_i = limit_action_inf_norm(u_i, 1)
        U[i] = u_i.reshape(2, 1)

    return U

def guide_policy_navigation_cbf(world):

    egos = world.egos
    dynamic_obstacles = world.dynamic_obstacles
    obstacles = world.obstacles
    targets = world.targets
    num_egos = len(egos)
    U = np.zeros((num_egos, 2, 1), dtype=np.float32)

    edge_list = world.edge_list.tolist()
    edge_num = len(edge_list[1])  # each edge is calculated twice

    # ---- parameters (match your CBF runner) ----
    KP = 1.5             # goal attraction gain
    MAX_SPEED = 1.0      # per-axis speed clamp
    CBF_ALPHA = 1.0      # ZCBF alpha
    POCS_ITERS = 2       # projection iterations
    k_v = 1.0           # velocity tracking gain -> convert v_safe to action u

    def clip_speed_box(v, vmax=MAX_SPEED):
        v = np.asarray(v, dtype=np.float32).copy()
        v[0] = np.clip(v[0], -vmax, vmax)
        v[1] = np.clip(v[1], -vmax, vmax)
        return v

    def cbf_project_velocity(v_nom, p_i, vj_list, pj_list, rj_list, Ri,
                             alpha=CBF_ALPHA, iters=POCS_ITERS, vmax=MAX_SPEED):
        v = clip_speed_box(v_nom, vmax)
        for _ in range(iters):
            for pj, vj, Rj in zip(pj_list, vj_list, rj_list):
                p_rel = p_i - pj
                h = float(np.dot(p_rel, p_rel) - (Ri + Rj) ** 2)
                n = 2.0 * p_rel
                c = -alpha * h + 2.0 * float(np.dot(p_rel, vj))  # ZCBF linearized RHS
                lhs = float(np.dot(n, v))
                if lhs < c:
                    nn = float(np.dot(n, n)) + 1e-12
                    v = v + (c - lhs) * (n / nn)                # orthogonal projection
            v = clip_speed_box(v, vmax)
        return v

    # # Pre-assemble obstacle constraint lists
    # pj_obs = [o.state.p_pos.copy() for o in obstacles]
    # vj_obs = [np.zeros(2, dtype=np.float32) for _ in obstacles]  # static obstacles
    # rj_obs = [o.R for o in obstacles]

    # # Dynamic obstacles
    # pj_dobs = [d.state.p_pos.copy() for d in dynamic_obstacles]
    # vj_dobs = [d.state.p_vel.copy() for d in dynamic_obstacles]
    # rj_dobs = [d.R for d in dynamic_obstacles]

    for i, ego in enumerate(egos):
        # Get the neighbors of the ego
        neighbors_id = []  # the neighbor id of all entities, global id
        for j in range(edge_num):
            if int(edge_list[0][j]) == ego.global_id:
                neighbors_id.append(edge_list[1][j])
            if int(edge_list[0][j]) > ego.global_id:
                break

        neighbors_ego = [e for e in egos if e.global_id in neighbors_id]
        neighbors_dobs = [d for d in dynamic_obstacles if d.global_id in neighbors_id]
        neighbors_obs = [o for o in obstacles if o.global_id in neighbors_id]
        
        # Pre-assemble obstacle constraint lists
        pj_obs = [o.state.p_pos.copy() for o in neighbors_obs]
        vj_obs = [np.zeros(2, dtype=np.float32) for _ in neighbors_obs]  # static obstacles
        rj_obs = [o.R for o in neighbors_obs]

        # Dynamic obstacles
        pj_dobs = [d.state.p_pos.copy() for d in neighbors_dobs]
        vj_dobs = [d.state.p_vel.copy() for d in neighbors_dobs]
        rj_dobs = [d.R for d in neighbors_dobs]

        # Build constraint sets: static obs + dynamic obs + all other egos
        pj = list(pj_obs) + list(pj_dobs)
        vj = list(vj_obs) + list(vj_dobs)
        rj = list(rj_obs) + list(rj_dobs)

        for other in neighbors_ego:
            if other is ego:
                continue
            pj.append(other.state.p_pos.copy())
            vj.append(other.state.p_vel.copy())
            rj.append(other.R)

        # Nominal goal-seeking velocity
        goal = targets[i].state.p_pos
        v_nom = KP * (goal - ego.state.p_pos)
        v_nom = clip_speed_box(v_nom, MAX_SPEED)

        # CBF projection -> safe velocity
        v_safe = cbf_project_velocity(v_nom, ego.state.p_pos, vj, pj, rj, ego.R)

        # Convert to action and clip (keep the same interface as other policies)
        u_i = (v_safe - ego.state.p_vel)/delta_t
        u_i = limit_action_inf_norm(u_i, 1)
        U[i] = u_i.reshape(2, 1)

    return U

def limit_action_inf_norm(action, max_limit):
    action = np.float32(action)
    action_ = action
    if abs(action[0]) > abs(action[1]):
        if abs(action[0])>max_limit:
            action_[1] = max_limit*action[1]/abs(action[0])
            action_[0] = max_limit if action[0] > 0 else -max_limit
        else:
            pass
    else:
        if abs(action[1])>max_limit:
            action_[0] = max_limit*action[0]/abs(action[1])
            action_[1] = max_limit if action[1] > 0 else -max_limit
        else:
            pass
    return action_
