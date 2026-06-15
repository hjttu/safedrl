from typing import Dict, Tuple

import torch

from onpolicy.algorithms.utils.distributions import FixedCategorical


class DecentralizedPriorityJointDTCBFShield:
    """Sequential one-hop joint-action shield with decentralized priorities."""

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

    def _ego_index(self, agent_id, agent_slot):
        value = int(agent_id[agent_slot].reshape(-1)[0].item())
        return value

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
        other_radius = node_obs_i[other_index, 4]
        return rel_pos, rel_vel, ego_radius, other_radius

    def _agent_velocity(self, node_obs_i, ego_index):
        if self.graph_feat_type == "global":
            return node_obs_i[ego_index, 2:4]
        entity_types = node_obs_i[:, -1].long()
        targets = torch.nonzero(entity_types == 1, as_tuple=False).flatten()
        if targets.numel() > 0:
            return node_obs_i[targets[0], 2:4]
        return torch.zeros(2, dtype=node_obs_i.dtype, device=node_obs_i.device)

    def _backup_index(self, node_obs_i, ego_index):
        if self.backup_action_mode == "zero":
            target = torch.zeros(2, device=self.device)
        else:
            velocity = self._agent_velocity(node_obs_i, ego_index)
            target = torch.clamp(
                -velocity / max(self.max_accel * self.dt, 1e-8), -1.0, 1.0
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
        rel_pos, rel_vel, radius_i, radius_j = self._pair_state(
            node_obs_i, ego_index_i, other_index
        )
        clearance = radius_i + radius_j + self.safety_buffer
        h_now = rel_pos.square().sum() - clearance.square()
        accel_i = self.action_table * self.max_accel
        accel_j = self.action_table[int(fixed_action_j)] * self.max_accel
        rel_pos_next = (
            rel_pos[None]
            + self.dt * rel_vel[None]
            + 0.5 * self.dt ** 2 * (accel_i - accel_j[None])
        )
        h_next = rel_pos_next.square().sum(-1) - clearance.square()
        margins = h_next - (1.0 - self.alpha) * h_now
        return margins >= 0.0, margins

    def _priority(self, node_obs_i, adj_i, ego_index, agent_identifier):
        if self.priority_metric == "agent_id":
            return float(agent_identifier)
        entity_types = node_obs_i[:, -1].long()
        agent_nodes = torch.nonzero(entity_types == 0, as_tuple=False).flatten()
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
            clearance = radius_i + radius_j + self.safety_buffer
            h = rel_pos.square().sum() - clearance.square()
            if self.priority_metric == "ttc":
                closing = -(rel_pos * rel_vel).sum()
                value = h / closing.clamp_min(1e-6)
            else:
                value = h
            values.append(float(value.item()))
        return min(values) if values else float("inf")

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
            batch_size,
            num_agents,
            action_count,
            dtype=torch.bool,
            device=self.device,
        )
        local_safe = local_mask.float().mean().item()
        final_domain_sizes = []
        pairwise_minima = []
        no_joint_safe = 0
        backup_used = 0
        least_unsafe_used = 0
        pairwise_edges = 0

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
            selected = {}

            for i in order:
                ego_i = ego_indices[i]
                mask_i = local_mask[b, i].bool().clone()
                combined_margin = torch.full(
                    (action_count,),
                    torch.finfo(logits.dtype).max,
                    dtype=logits.dtype,
                    device=self.device,
                )
                neighbors = []
                for j, action_j in selected.items():
                    other_index = ego_indices[j]
                    distance = float(adj[b, i, ego_i, other_index].item())
                    if 0.0 < distance <= self.max_edge_dist:
                        neighbors.append((distance, j, action_j))
                neighbors.sort(key=lambda item: item[0])
                for _, j, action_j in neighbors[: self.max_neighbors]:
                    compat, margins = self.pairwise_compatibility(
                        node_obs[b, i], ego_i, ego_indices[j], action_j
                    )
                    mask_i &= compat
                    combined_margin = torch.minimum(combined_margin, margins)
                    pairwise_minima.append(float(margins.min().item()))
                    pairwise_edges += 1

                if not mask_i.any():
                    no_joint_safe += 1
                    if self.no_safe_action_strategy == "terminate":
                        raise RuntimeError(
                            "Joint DTCBF shield found no feasible action."
                        )
                    if self.no_safe_action_strategy == "least_unsafe":
                        if torch.isinf(combined_margin).all():
                            fallback = int(logits[b, i].argmax().item())
                        else:
                            fallback = int(combined_margin.argmax().item())
                        least_unsafe_used += 1
                    else:
                        fallback = self._backup_index(
                            node_obs[b, i], ego_i
                        )
                        backup_used += 1
                    mask_i[fallback] = True

                final_masks[b, i] = mask_i
                final_domain_sizes.append(int(mask_i.sum().item()))
                masked_logits = logits[b, i].masked_fill(
                    ~mask_i, torch.finfo(logits.dtype).min
                )
                dist = FixedCategorical(logits=masked_logits[None])
                action = dist.mode() if deterministic else dist.sample()
                actions[b, i] = action[0]
                log_probs[b, i] = dist.log_probs(action)[0]
                selected[i] = int(action.item())

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
            "num_backup_used": float(backup_used),
            "num_least_unsafe_used": float(least_unsafe_used),
            "min_pairwise_dtcbf_margin": float(
                min(pairwise_minima) if pairwise_minima else 0.0
            ),
            "num_pairwise_edges": float(pairwise_edges),
        }
        return actions, log_probs, final_masks, stats
