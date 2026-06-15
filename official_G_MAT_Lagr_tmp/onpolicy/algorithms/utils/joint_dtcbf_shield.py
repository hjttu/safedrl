import itertools
from typing import Dict, List, Tuple

import torch

from onpolicy.algorithms.utils.distributions import (
    FixedCategorical,
    categorical_mask_value,
)


class DecentralizedPriorityJointDTCBFShield:
    """Priority shield with immediate hard and predictive mixed constraints."""

    def __init__(self, args, action_table, device):
        self.args = args
        self.device = device
        self.action_table = action_table.to(device)
        self.dt = args.cbf_dt
        self.alpha = args.cbf_alpha
        self.max_accel = args.cbf_max_accel
        self.safety_buffer = args.cbf_safety_buffer
        self.max_edge_dist = args.max_edge_dist
        self.max_neighbors = args.max_shield_neighbors
        self.priority_metric = args.priority_metric
        self.backup_action_mode = args.backup_action_mode
        self.graph_feat_type = args.graph_feat_type
        self.horizon = max(int(getattr(args, "dtcbf_horizon", 3)), 1)
        self.predict_mode = getattr(
            args, "dtcbf_predict_mode", "constant_action"
        )
        self.min_margin = float(getattr(args, "dtcbf_min_margin", 0.0))
        self.early_brake_buffer = float(
            getattr(args, "dtcbf_early_brake_buffer", 0.05)
        )
        self.predictive_hard_neighbors = max(
            int(getattr(args, "predictive_hard_neighbors", 1)), 0
        )
        self.min_joint_domain_size = max(
            int(getattr(args, "min_joint_domain_size", 5)), 1
        )
        self.predictive_soft_penalty_coef = float(
            getattr(args, "predictive_soft_penalty_coef", 2.0)
        )
        self.use_soft_predictive_penalty = bool(
            getattr(args, "use_soft_predictive_penalty", True)
        )
        self.recovery_margin_coef = float(
            getattr(args, "recovery_margin_coef", 10.0)
        )
        self.recovery_logit_coef = float(
            getattr(args, "recovery_logit_coef", 1.0)
        )
        self.recovery_progress_coef = float(
            getattr(args, "recovery_progress_coef", 0.1)
        )
        self.use_joint_repair = bool(
            getattr(args, "use_joint_repair", True)
        )
        self.repair_top_k = max(
            int(getattr(args, "joint_repair_top_k", 8)), 1
        )
        self.repair_max_cluster_size = max(
            int(getattr(args, "joint_repair_max_cluster_size", 4)), 2
        )
        self.repair_include_blockers = bool(
            getattr(args, "joint_repair_include_blockers", True)
        )

    def _ego_index(self, agent_id, agent_slot):
        return int(agent_id[agent_slot].reshape(-1)[0].item())

    def _pair_state(self, node_obs_i, ego_index_i, other_index):
        if self.graph_feat_type == "global":
            ego = node_obs_i[ego_index_i]
            other = node_obs_i[other_index]
            rel_pos = ego[0:2] - other[0:2]
            rel_vel = ego[2:4] - other[2:4]
            ego_radius = ego[4]
        else:
            other = node_obs_i[other_index]
            rel_pos = other[0:2]
            rel_vel = other[2:4]
            ego_radius = node_obs_i[ego_index_i, 4]
        return rel_pos, rel_vel, ego_radius, node_obs_i[other_index, 4]

    def _goal_direction(self, node_obs_i, ego_index, agent_slot, num_agents):
        target_index = num_agents + agent_slot
        if target_index >= node_obs_i.shape[0]:
            return torch.zeros(2, device=self.device)
        if int(node_obs_i[target_index, -1].item()) != 1:
            return torch.zeros(2, device=self.device)
        if self.graph_feat_type == "global":
            direction = (
                node_obs_i[target_index, 0:2]
                - node_obs_i[ego_index, 0:2]
            )
        else:
            direction = -node_obs_i[target_index, 0:2]
        return direction / direction.norm().clamp_min(1e-6)

    def multi_step_pairwise_margin_vector(
        self,
        node_obs_i,
        ego_index_i,
        other_index,
        fixed_action_j,
        horizon,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.predict_mode != "constant_action":
            raise ValueError(
                f"Unsupported DTCBF prediction mode: {self.predict_mode}"
            )
        rel_pos, rel_vel, radius_i, radius_j = self._pair_state(
            node_obs_i, ego_index_i, other_index
        )
        clearance = (
            radius_i
            + radius_j
            + self.safety_buffer
            + self.early_brake_buffer
        )
        h_now = rel_pos.square().sum() - clearance.square()
        accel_i = self.action_table * self.max_accel
        accel_j = self.action_table[int(fixed_action_j)] * self.max_accel
        relative_accel = accel_i - accel_j[None]
        margins = []
        for step in range(1, max(int(horizon), 1) + 1):
            elapsed = step * self.dt
            rel_pos_k = (
                rel_pos[None]
                + elapsed * rel_vel[None]
                + 0.5 * elapsed ** 2 * relative_accel
            )
            h_k = rel_pos_k.square().sum(-1) - clearance.square()
            margins.append(h_k - (1.0 - self.alpha) ** step * h_now)
        min_margin = torch.stack(margins).min(dim=0).values
        return min_margin >= self.min_margin, min_margin

    def pairwise_compatibility(
        self, node_obs_i, ego_index_i, other_index, fixed_action_j
    ):
        return self.multi_step_pairwise_margin_vector(
            node_obs_i,
            ego_index_i,
            other_index,
            fixed_action_j,
            self.horizon,
        )

    def _priority(self, node_obs_i, adj_i, ego_index, agent_identifier):
        if self.priority_metric == "agent_id":
            return float(agent_identifier)
        entity_types = node_obs_i[:, -1].long()
        agent_nodes = torch.nonzero(
            entity_types == 0, as_tuple=False
        ).flatten()
        values = []
        for other in agent_nodes.tolist():
            if other == ego_index:
                continue
            distance = float(adj_i[ego_index, other].item())
            if distance <= 0.0 or distance > self.max_edge_dist:
                continue
            rel_pos, rel_vel, radius_i, radius_j = self._pair_state(
                node_obs_i, ego_index, other
            )
            clearance = (
                radius_i
                + radius_j
                + self.safety_buffer
                + self.early_brake_buffer
            )
            h = rel_pos.square().sum() - clearance.square()
            if self.priority_metric == "ttc":
                closing = -(rel_pos * rel_vel).sum()
                value = h / closing.clamp_min(1e-6)
            else:
                value = h
            values.append(float(value.item()))
        return min(values) if values else float("inf")

    def _selected_neighbors(self, i, selected, ego_indices, adj_i):
        ego_i = ego_indices[i]
        neighbors = []
        for j, action_j in selected.items():
            if j == i:
                continue
            other_index = ego_indices[j]
            distance = float(adj_i[ego_i, other_index].item())
            if 0.0 < distance <= self.max_edge_dist:
                neighbors.append((distance, j, action_j))
        neighbors.sort(key=lambda item: item[0])
        return neighbors[: self.max_neighbors]

    def _constraint_state(
        self, i, selected, local_mask_i, node_obs_b, adj_b, ego_indices
    ):
        mask_h1 = local_mask_i.bool().clone()
        blockers = []
        predictive_data = []
        neighbors = self._selected_neighbors(
            i, selected, ego_indices, adj_b[i]
        )
        for distance, j, action_j in neighbors:
            compat_h1, margin_h1 = self.multi_step_pairwise_margin_vector(
                node_obs_b[i],
                ego_indices[i],
                ego_indices[j],
                action_j,
                horizon=1,
            )
            previous_count = int(mask_h1.sum().item())
            next_mask = mask_h1 & compat_h1
            if int(next_mask.sum().item()) < previous_count:
                blockers.append((float(margin_h1.max().item()), distance, j))
            mask_h1 = next_mask

            compat_hh, margin_hh = self.multi_step_pairwise_margin_vector(
                node_obs_b[i],
                ego_indices[i],
                ego_indices[j],
                action_j,
                horizon=self.horizon,
            )
            predictive_data.append(
                {
                    "neighbor": j,
                    "compat": compat_hh,
                    "margin": margin_hh,
                    "risk": -float(margin_hh.max().item()),
                }
            )

        mask_predictive = mask_h1.clone()
        ranked = sorted(
            predictive_data, key=lambda item: item["risk"], reverse=True
        )
        for item in ranked[: self.predictive_hard_neighbors]:
            mask_predictive &= item["compat"]

        predictive_applied = (
            self.predictive_hard_neighbors > 0
            and bool(predictive_data)
            and int(mask_predictive.sum().item()) >= self.min_joint_domain_size
        )
        final_mask = mask_predictive if predictive_applied else mask_h1
        predictive_relaxed = (
            self.predictive_hard_neighbors > 0
            and bool(predictive_data)
            and not predictive_applied
            and int(mask_predictive.sum().item()) < int(mask_h1.sum().item())
        )

        if predictive_data:
            predictive_penalty = torch.stack(
                [
                    torch.relu(-item["margin"])
                    for item in predictive_data
                ]
            ).max(dim=0).values
        else:
            predictive_penalty = torch.zeros_like(local_mask_i, dtype=torch.float32)
        if not self.use_soft_predictive_penalty:
            predictive_penalty.zero_()

        return {
            "mask": final_mask,
            "mask_h1": mask_h1,
            "mask_predictive": mask_predictive,
            "penalty": predictive_penalty,
            "blockers": blockers,
            "edges": len(neighbors),
            "predictive_applied": predictive_applied,
            "predictive_relaxed": predictive_relaxed,
        }

    def _action_weights(self, mask, penalty):
        weights = mask.to(dtype=penalty.dtype)
        if self.use_soft_predictive_penalty:
            log_weight = -self.predictive_soft_penalty_coef * penalty.float()
            weights = weights * torch.exp(log_weight.clamp(min=-9.0))
        return weights

    def _adjusted_logits(self, logits_i, penalty):
        if not self.use_soft_predictive_penalty:
            return logits_i
        return logits_i - self.predictive_soft_penalty_coef * penalty.to(
            dtype=logits_i.dtype
        )

    def _top_actions(
        self, logits_i, mask_i, penalty, required_action=None
    ):
        candidates = torch.nonzero(mask_i, as_tuple=False).flatten()
        if candidates.numel() == 0:
            candidates = torch.arange(logits_i.shape[0], device=self.device)
        adjusted = self._adjusted_logits(logits_i, penalty)
        top_count = min(self.repair_top_k, candidates.numel())
        result = candidates[
            torch.topk(adjusted[candidates], top_count).indices
        ].tolist()
        if required_action is not None and required_action not in result:
            result[-1] = int(required_action)
        return list(dict.fromkeys(int(action) for action in result))

    def _repair_cluster(
        self,
        i,
        blockers,
        selected,
        logits_b,
        local_mask_b,
        node_obs_b,
        adj_b,
        ego_indices,
    ):
        if not self.use_joint_repair or not blockers:
            return None
        blocker_ids = [item[2] for item in sorted(blockers)]
        if not self.repair_include_blockers:
            blocker_ids = blocker_ids[:1]
        cluster = [i]
        for blocker in blocker_ids:
            if blocker not in cluster:
                cluster.append(blocker)
            if len(cluster) >= self.repair_max_cluster_size:
                break
        if len(cluster) < 2:
            return None

        candidate_lists = []
        for agent in cluster:
            state = self._constraint_state(
                agent,
                selected,
                local_mask_b[agent],
                node_obs_b,
                adj_b,
                ego_indices,
            )
            candidate_lists.append(
                self._top_actions(
                    logits_b[agent],
                    state["mask_h1"],
                    state["penalty"],
                    selected.get(agent),
                )
            )

        best_score = None
        best_actions = None
        outside = {
            agent: action
            for agent, action in selected.items()
            if agent not in cluster
        }
        for action_tuple in itertools.product(*candidate_lists):
            trial = dict(outside)
            trial.update(dict(zip(cluster, action_tuple)))
            valid = True
            score = 0.0
            for agent, action in zip(cluster, action_tuple):
                state = self._constraint_state(
                    agent,
                    trial,
                    local_mask_b[agent],
                    node_obs_b,
                    adj_b,
                    ego_indices,
                )
                if not bool(state["mask_h1"][action]):
                    valid = False
                    break
                adjusted = self._adjusted_logits(
                    logits_b[agent], state["penalty"]
                )
                score += float(adjusted[action].item())
            if valid and (best_score is None or score > best_score):
                best_score = score
                best_actions = dict(zip(cluster, action_tuple))
        return best_actions

    def _maxmin_recovery(
        self,
        i,
        selected,
        logits_i,
        local_mask_i,
        node_obs_b,
        adj_b,
        ego_indices,
        num_agents,
    ):
        state = self._constraint_state(
            i,
            selected,
            torch.ones_like(local_mask_i),
            node_obs_b,
            adj_b,
            ego_indices,
        )
        immediate_margins = []
        for _, j, action_j in self._selected_neighbors(
            i, selected, ego_indices, adj_b[i]
        ):
            _, margin_h1 = self.multi_step_pairwise_margin_vector(
                node_obs_b[i],
                ego_indices[i],
                ego_indices[j],
                action_j,
                horizon=1,
            )
            immediate_margins.append(margin_h1)
        if immediate_margins:
            min_margin = torch.stack(immediate_margins).min(dim=0).values
        else:
            min_margin = torch.zeros_like(logits_i)

        adjusted_logits = self._adjusted_logits(logits_i, state["penalty"])
        policy_scores = torch.log_softmax(adjusted_logits.float(), dim=-1)
        goal_direction = self._goal_direction(
            node_obs_b[i], ego_indices[i], i, num_agents
        )
        progress = (self.action_table * goal_direction[None]).sum(-1)
        score = (
            self.recovery_margin_coef * min_margin.float()
            + self.recovery_logit_coef * policy_scores
            + self.recovery_progress_coef * progress.float()
        )
        if local_mask_i.any():
            score = score.masked_fill(~local_mask_i.bool(), -torch.inf)
        return int(score.argmax().item()), state

    def _audit_immediate_safety(
        self, selected, node_obs_b, adj_b, ego_indices
    ):
        agents = sorted(selected)
        unsafe = set()
        for offset, i in enumerate(agents):
            for j in agents[offset + 1:]:
                distance = float(
                    adj_b[i, ego_indices[i], ego_indices[j]].item()
                )
                if distance <= 0.0 or distance > self.max_edge_dist:
                    continue
                compat_h1, _ = self.multi_step_pairwise_margin_vector(
                    node_obs_b[i],
                    ego_indices[i],
                    ego_indices[j],
                    selected[j],
                    horizon=1,
                )
                if not bool(compat_h1[selected[i]]):
                    unsafe.update((i, j))
        return unsafe

    def sample(
        self, logits, local_mask, node_obs, adj, agent_id, deterministic=False
    ):
        batch_size, num_agents, action_count = logits.shape
        actions = torch.zeros(
            batch_size, num_agents, 1, dtype=torch.long, device=self.device
        )
        log_probs = torch.zeros(
            batch_size, num_agents, 1, dtype=logits.dtype, device=self.device
        )
        action_weights = torch.zeros(
            batch_size,
            num_agents,
            action_count,
            dtype=logits.dtype,
            device=self.device,
        )
        local_safe = local_mask.float().mean().item()
        final_domain_sizes: List[int] = []
        no_joint_safe = 0
        recovery_used = 0
        repair_attempts = 0
        repair_successes = 0
        predictive_applied = 0
        predictive_relaxed = 0

        for b in range(batch_size):
            ego_indices = [
                self._ego_index(agent_id[b], i) for i in range(num_agents)
            ]
            priorities = [
                (
                    self._priority(
                        node_obs[b, i],
                        adj[b, i],
                        ego_indices[i],
                        ego_indices[i],
                    ),
                    ego_indices[i],
                    i,
                )
                for i in range(num_agents)
            ]
            order = [item[2] for item in sorted(priorities)]
            selected: Dict[int, int] = {}
            decision_weights: Dict[int, torch.Tensor] = {}

            for i in order:
                state = self._constraint_state(
                    i,
                    selected,
                    local_mask[b, i],
                    node_obs[b],
                    adj[b],
                    ego_indices,
                )
                predictive_applied += int(state["predictive_applied"])
                predictive_relaxed += int(state["predictive_relaxed"])

                if not state["mask"].any():
                    no_joint_safe += 1
                    repaired = None
                    if (
                        self.use_joint_repair
                        and local_mask[b, i].any()
                        and state["blockers"]
                    ):
                        repair_attempts += 1
                        repaired = self._repair_cluster(
                            i,
                            state["blockers"],
                            selected,
                            logits[b],
                            local_mask[b],
                            node_obs[b],
                            adj[b],
                            ego_indices,
                        )
                    if repaired is not None:
                        selected.update(repaired)
                        for repaired_agent in repaired:
                            repaired_state = self._constraint_state(
                                repaired_agent,
                                selected,
                                local_mask[b, repaired_agent],
                                node_obs[b],
                                adj[b],
                                ego_indices,
                            )
                            decision_weights[repaired_agent] = (
                                self._action_weights(
                                    repaired_state["mask"],
                                    repaired_state["penalty"],
                                )
                            )
                        repair_successes += 1
                        continue

                    recovery, recovery_state = self._maxmin_recovery(
                        i,
                        selected,
                        logits[b, i],
                        local_mask[b, i],
                        node_obs[b],
                        adj[b],
                        ego_indices,
                        num_agents,
                    )
                    selected[i] = recovery
                    weights = torch.zeros_like(logits[b, i])
                    weights[recovery] = 1.0
                    decision_weights[i] = weights
                    recovery_used += 1
                    continue

                weights = self._action_weights(
                    state["mask"], state["penalty"]
                ).to(dtype=logits.dtype)
                adjusted_logits = logits[b, i] + torch.log(
                    weights.clamp_min(1e-4)
                )
                adjusted_logits = adjusted_logits.masked_fill(
                    ~state["mask"], categorical_mask_value(adjusted_logits)
                )
                dist = FixedCategorical(logits=adjusted_logits[None])
                action = dist.mode() if deterministic else dist.sample()
                selected[i] = int(action.item())
                decision_weights[i] = weights

            unsafe_agents = self._audit_immediate_safety(
                selected, node_obs[b], adj[b], ego_indices
            )
            for i in range(num_agents):
                action_i = int(selected[i])
                weights_i = decision_weights[i].clone()
                if i in unsafe_agents or weights_i[action_i] <= 0:
                    weights_i.zero_()
                    weights_i[action_i] = 1.0
                action_weights[b, i] = weights_i
                adjusted_logits = logits[b, i] + torch.log(
                    weights_i.clamp_min(1e-4)
                )
                adjusted_logits = adjusted_logits.masked_fill(
                    weights_i <= 0, categorical_mask_value(adjusted_logits)
                )
                dist = FixedCategorical(logits=adjusted_logits[None])
                action = torch.tensor([[action_i]], device=self.device)
                actions[b, i] = action[0]
                log_probs[b, i] = dist.log_probs(action)[0]
                final_domain_sizes.append(
                    int((weights_i > 0).sum().item())
                )

        stats: Dict[str, float] = {
            "local_safe_action_ratio": local_safe,
            "joint_safe_action_ratio": float(
                (action_weights > 0).float().mean().item()
            ),
            "avg_final_domain_size": float(
                sum(final_domain_sizes) / max(len(final_domain_sizes), 1)
            ),
            "min_final_domain_size": float(
                min(final_domain_sizes) if final_domain_sizes else 0
            ),
            "num_no_joint_safe_action": float(no_joint_safe),
            "num_recovery_used": float(recovery_used),
            "num_backup_used": float(recovery_used),
            "num_joint_repair_attempts": float(repair_attempts),
            "num_joint_repair_successes": float(repair_successes),
            "predictive_hard_applied_count": float(predictive_applied),
            "predictive_hard_relaxed_count": float(predictive_relaxed),
        }
        return actions, log_probs, action_weights, stats
