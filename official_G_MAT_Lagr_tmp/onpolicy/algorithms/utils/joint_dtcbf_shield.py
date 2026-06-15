import itertools
from typing import Dict, List, Tuple

import torch

from onpolicy.algorithms.utils.distributions import (
    FixedCategorical,
    categorical_mask_value,
)


class DecentralizedPriorityJointDTCBFShield:
    """Priority shield with predictive DTCBF, local repair, and recovery."""

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
        self.no_safe_action_strategy = args.no_safe_action_strategy
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

    def _agent_velocity(self, node_obs_i, ego_index):
        if self.graph_feat_type == "global":
            return node_obs_i[ego_index, 2:4]
        entity_types = node_obs_i[:, -1].long()
        targets = torch.nonzero(entity_types == 1, as_tuple=False).flatten()
        if targets.numel() > 0:
            return node_obs_i[targets[0], 2:4]
        return torch.zeros(2, dtype=node_obs_i.dtype, device=node_obs_i.device)

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

    def _backup_index(self, node_obs_i, ego_index):
        if self.backup_action_mode == "zero":
            target = torch.zeros(2, device=self.device)
        else:
            velocity = self._agent_velocity(node_obs_i, ego_index)
            target = torch.clamp(
                -velocity / max(self.max_accel * self.dt, 1e-8),
                -1.0,
                1.0,
            )
        distances = (self.action_table - target).square().sum(-1)
        return int(distances.argmin().item())

    def pairwise_compatibility(
        self,
        node_obs_i,
        ego_index_i,
        other_index,
        fixed_action_j,
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
        horizon_margins = []
        for step in range(1, self.horizon + 1):
            elapsed = step * self.dt
            rel_pos_k = (
                rel_pos[None]
                + elapsed * rel_vel[None]
                + 0.5 * elapsed ** 2 * relative_accel
            )
            h_k = rel_pos_k.square().sum(-1) - clearance.square()
            decay = (1.0 - self.alpha) ** step
            horizon_margins.append(h_k - decay * h_now)
        margins = torch.stack(horizon_margins).min(dim=0).values
        return margins >= self.min_margin, margins

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
            other_index = ego_indices[j]
            distance = float(adj_i[ego_i, other_index].item())
            if 0.0 < distance <= self.max_edge_dist:
                neighbors.append((distance, j, action_j))
        neighbors.sort(key=lambda item: item[0])
        return neighbors[: self.max_neighbors]

    def _conditional_mask(
        self, i, selected, local_mask_i, node_obs_b, adj_b, ego_indices
    ):
        mask_i = local_mask_i.bool().clone()
        action_count = mask_i.shape[0]
        combined_margin = torch.full(
            (action_count,),
            torch.finfo(node_obs_b.dtype).max,
            dtype=node_obs_b.dtype,
            device=self.device,
        )
        blockers = []
        neighbors = self._selected_neighbors(
            i, {j: a for j, a in selected.items() if j != i},
            ego_indices, adj_b[i]
        )
        for distance, j, action_j in neighbors:
            compat, margins = self.pairwise_compatibility(
                node_obs_b[i], ego_indices[i], ego_indices[j], action_j
            )
            previous_count = int(mask_i.sum().item())
            next_mask = mask_i & compat
            if int(next_mask.sum().item()) < previous_count:
                blockers.append((float(margins.max().item()), distance, j))
            mask_i = next_mask
            combined_margin = torch.minimum(combined_margin, margins)
        return mask_i, combined_margin, blockers, len(neighbors)

    def _top_actions(self, logits_i, local_mask_i, required_action=None):
        candidates = torch.nonzero(local_mask_i, as_tuple=False).flatten()
        if candidates.numel() == 0:
            candidates = torch.arange(logits_i.shape[0], device=self.device)
        scores = logits_i[candidates]
        top_count = min(self.repair_top_k, candidates.numel())
        result = candidates[torch.topk(scores, top_count).indices].tolist()
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
            candidate_lists.append(
                self._top_actions(
                    logits_b[agent],
                    local_mask_b[agent].bool(),
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
            min_margin = float("inf")
            for agent, action in zip(cluster, action_tuple):
                mask, margins, _, _ = self._conditional_mask(
                    agent,
                    trial,
                    local_mask_b[agent],
                    node_obs_b,
                    adj_b,
                    ego_indices,
                )
                if not bool(mask[action]):
                    valid = False
                    break
                if not torch.isinf(margins).all():
                    min_margin = min(
                        min_margin, float(margins[action].item())
                    )
            if not valid:
                continue
            logit_score = sum(
                float(logits_b[agent, action].item())
                for agent, action in zip(cluster, action_tuple)
            )
            margin_score = 0.0 if min_margin == float("inf") else min_margin
            score = (
                self.recovery_margin_coef * margin_score
                + self.recovery_logit_coef * logit_score
            )
            if best_score is None or score > best_score:
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
        _, margins, _, _ = self._conditional_mask(
            i, selected, torch.ones_like(local_mask_i),
            node_obs_b, adj_b, ego_indices
        )
        finite_pairwise = ~torch.isinf(margins)
        if not finite_pairwise.any():
            margins = torch.zeros_like(logits_i)
        policy_scores = torch.log_softmax(logits_i.float(), dim=-1)
        goal_direction = self._goal_direction(
            node_obs_b[i], ego_indices[i], i, num_agents
        )
        progress = (self.action_table * goal_direction[None]).sum(-1)
        score = (
            self.recovery_margin_coef * margins.float()
            + self.recovery_logit_coef * policy_scores
            + self.recovery_progress_coef * progress.float()
        )
        if local_mask_i.any():
            score = score.masked_fill(~local_mask_i.bool(), -torch.inf)
        return int(score.argmax().item()), float(margins.max().item())

    def _finalize_distributions(
        self,
        selected,
        logits_b,
        decision_masks,
        certified,
    ):
        num_agents, action_count = logits_b.shape
        actions = torch.zeros(
            num_agents, 1, dtype=torch.long, device=self.device
        )
        log_probs = torch.zeros(
            num_agents, 1, dtype=logits_b.dtype, device=self.device
        )
        final_masks = torch.zeros(
            num_agents, action_count, dtype=torch.bool, device=self.device
        )
        for i in range(num_agents):
            action_i = int(selected[i])
            mask_i = decision_masks[i].clone()
            if not bool(mask_i[action_i]):
                mask_i.zero_()
                mask_i[action_i] = True
                certified[i] = False
            final_masks[i] = mask_i
            masked_logits = logits_b[i].masked_fill(
                ~mask_i, categorical_mask_value(logits_b[i])
            )
            dist = FixedCategorical(logits=masked_logits[None])
            action = torch.tensor([[action_i]], device=self.device)
            actions[i] = action[0]
            log_probs[i] = dist.log_probs(action)[0]
        return actions, log_probs, final_masks

    def _audit_selected_actions(
        self, selected, node_obs_b, adj_b, ego_indices, certified
    ):
        agents = sorted(selected)
        for offset, i in enumerate(agents):
            for j in agents[offset + 1:]:
                distance = float(
                    adj_b[i, ego_indices[i], ego_indices[j]].item()
                )
                if distance <= 0.0 or distance > self.max_edge_dist:
                    continue
                compat, _ = self.pairwise_compatibility(
                    node_obs_b[i],
                    ego_indices[i],
                    ego_indices[j],
                    selected[j],
                )
                if not bool(compat[selected[i]]):
                    certified[i] = False
                    certified[j] = False

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
        final_masks = torch.zeros(
            batch_size, num_agents, action_count,
            dtype=torch.bool, device=self.device
        )
        local_safe = local_mask.float().mean().item()
        final_domain_sizes: List[int] = []
        pairwise_minima: List[float] = []
        no_joint_safe = 0
        recovery_used = 0
        repair_attempts = 0
        repair_successes = 0
        uncertified_actions = 0
        pairwise_edges = 0

        for b in range(batch_size):
            ego_indices = [
                self._ego_index(agent_id[b], i) for i in range(num_agents)
            ]
            priorities = [
                (
                    self._priority(
                        node_obs[b, i], adj[b, i],
                        ego_indices[i], ego_indices[i]
                    ),
                    ego_indices[i],
                    i,
                )
                for i in range(num_agents)
            ]
            order = [item[2] for item in sorted(priorities)]
            selected: Dict[int, int] = {}
            decision_masks: Dict[int, torch.Tensor] = {}
            certified = {i: True for i in range(num_agents)}

            for i in order:
                mask_i, margins, blockers, edges = self._conditional_mask(
                    i, selected, local_mask[b, i],
                    node_obs[b], adj[b], ego_indices
                )
                pairwise_edges += edges
                if not torch.isinf(margins).all():
                    pairwise_minima.append(float(margins.min().item()))

                if not mask_i.any():
                    no_joint_safe += 1
                    repaired = None
                    if (
                        self.use_joint_repair
                        and local_mask[b, i].any()
                        and blockers
                    ):
                        repair_attempts += 1
                        repaired = self._repair_cluster(
                            i, blockers, selected, logits[b], local_mask[b],
                            node_obs[b], adj[b], ego_indices
                        )
                    if repaired is not None:
                        selected.update(repaired)
                        for repaired_agent in repaired:
                            repaired_mask, _, _, _ = self._conditional_mask(
                                repaired_agent,
                                selected,
                                local_mask[b, repaired_agent],
                                node_obs[b],
                                adj[b],
                                ego_indices,
                            )
                            decision_masks[repaired_agent] = repaired_mask
                        repair_successes += 1
                        continue
                    if self.no_safe_action_strategy == "terminate":
                        raise RuntimeError(
                            "Joint DTCBF shield found no feasible action."
                        )
                    recovery, recovery_margin = self._maxmin_recovery(
                        i, selected, logits[b, i], local_mask[b, i],
                        node_obs[b], adj[b], ego_indices, num_agents
                    )
                    selected[i] = recovery
                    recovery_mask = torch.zeros_like(local_mask[b, i])
                    recovery_mask[recovery] = True
                    decision_masks[i] = recovery_mask
                    certified[i] = False
                    recovery_used += 1
                    pairwise_minima.append(recovery_margin)
                    continue

                masked_logits = logits[b, i].masked_fill(
                    ~mask_i, categorical_mask_value(logits[b, i])
                )
                dist = FixedCategorical(logits=masked_logits[None])
                action = dist.mode() if deterministic else dist.sample()
                selected[i] = int(action.item())
                decision_masks[i] = mask_i

            self._audit_selected_actions(
                selected, node_obs[b], adj[b], ego_indices, certified
            )
            actions[b], log_probs[b], final_masks[b] = (
                self._finalize_distributions(
                    selected, logits[b], decision_masks, certified
                )
            )
            uncertified_actions += sum(not value for value in certified.values())
            final_domain_sizes.extend(
                final_masks[b].sum(dim=-1).cpu().tolist()
            )

        stats: Dict[str, float] = {
            "local_safe_action_ratio": local_safe,
            "joint_safe_action_ratio": float(final_masks.float().mean().item()),
            "avg_final_domain_size": float(
                sum(final_domain_sizes) / max(len(final_domain_sizes), 1)
            ),
            "min_final_domain_size": float(
                min(final_domain_sizes) if final_domain_sizes else 0
            ),
            "num_no_joint_safe_action": float(no_joint_safe),
            "num_recovery_used": float(recovery_used),
            "num_backup_used": float(recovery_used),
            "num_least_unsafe_used": 0.0,
            "num_uncertified_actions": float(uncertified_actions),
            "num_joint_repair_attempts": float(repair_attempts),
            "num_joint_repair_successes": float(repair_successes),
            "joint_repair_success_rate": float(
                repair_successes / max(repair_attempts, 1)
            ),
            "min_pairwise_dtcbf_margin": float(
                min(pairwise_minima) if pairwise_minima else 0.0
            ),
            "num_pairwise_edges": float(pairwise_edges),
        }
        return actions, log_probs, final_masks, stats
