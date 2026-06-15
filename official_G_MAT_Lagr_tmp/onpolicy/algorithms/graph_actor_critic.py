import argparse
import math
import warnings
from typing import Tuple, List

import gym
import torch
from torch import Tensor
import torch.nn as nn
from onpolicy.algorithms.utils.util import init, check
# from onpolicy.algorithms.utils.gnn_transformer import GNNBase
# from onpolicy.algorithms.utils.gnn import GNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.lstm import LSTMLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.distributions import (
    FixedCategorical,
    categorical_mask_value,
)
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space
from multiagent.action_table import build_action_table_np


def minibatchGenerator(obs: Tensor, node_obs: Tensor, adj: Tensor, agent_id: Tensor, max_batch_size: int):
    """
    Split a big batch into smaller batches.
    """
    num_minibatches = obs.shape[0] // max_batch_size + 1
    for i in range(num_minibatches):
        yield (
            obs[i * max_batch_size : (i + 1) * max_batch_size],
            node_obs[i * max_batch_size : (i + 1) * max_batch_size],
            adj[i * max_batch_size : (i + 1) * max_batch_size],
            agent_id[i * max_batch_size : (i + 1) * max_batch_size],
        )


class GR_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    args: argparse.Namespace
        Arguments containing relevant model information.
    obs_space: (gym.Space)
        Observation space.
    node_obs_space: (gym.Space)
        Node observation space
    edge_obs_space: (gym.Space)
        Edge dimension in graphs
    action_space: (gym.Space)
        Action space.
    device: (torch.device)
        Specifies the device to run on (cpu/gpu).
    split_batch: (bool)
        Whether to split a big-batch into multiple
        smaller ones to speed up forward pass.
    max_batch_size: (int)
        Maximum batch size to use.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        obs_space: gym.Space,
        node_obs_space: gym.Space,
        edge_obs_space: gym.Space,
        action_space: gym.Space,
        device=torch.device("cpu"),
        split_batch: bool = False,
        max_batch_size: int = 32,
    ) -> None:
        super(GR_Actor, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.use_att_gnn = args.use_att_gnn
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_lstm = args.use_lstm
        self._recurrent_N = args.recurrent_N
        self.split_batch = split_batch
        self.max_batch_size = max_batch_size
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.use_graph_cbf_shield = (
            args.use_graph_cbf_shield
            and action_space.__class__.__name__ == "Discrete"
        )
        self.use_local_dtcbf_shield = args.use_local_dtcbf_shield
        self.use_joint_dtcbf_shield = args.use_joint_dtcbf_shield
        self.cbf_alpha = args.cbf_alpha
        self.cbf_dt = args.cbf_dt
        self.cbf_max_accel = args.cbf_max_accel
        self.cbf_max_speed = args.cbf_max_speed
        self.cbf_horizon = args.cbf_horizon
        self.cbf_include_obstacles = args.cbf_include_obstacles
        self.cbf_include_agents_in_local_mask = args.cbf_include_agents_in_local_mask
        self.cbf_safety_buffer = args.cbf_safety_buffer
        self.safety_score_coef = args.safety_score_coef
        self.no_safe_action_strategy = args.no_safe_action_strategy
        self.backup_action_mode = args.backup_action_mode
        self.guide_decay_steepness = args.guide_decay_steepness
        self.guide_weight = 1.0
        self.graph_feat_type = args.graph_feat_type

        obs_shape = get_shape_from_obs_space(obs_space)  # returns (6,)
        node_obs_shape = get_shape_from_obs_space(node_obs_space)[1]  # returns (num_nodes, num_node_feats), get 6 from (13, 6)
        edge_dim = get_shape_from_obs_space(edge_obs_space)[0]  # returns (1, )
        # print(edge_dim)
        if self.use_att_gnn:
            from onpolicy.algorithms.utils.gnn import GNNBase
        else:
            from onpolicy.algorithms.utils.gnn_transformer import GNNBase
        self.gnn_base = GNNBase(args, node_obs_shape, edge_dim, args.actor_graph_aggr)
        gnn_out_dim = self.gnn_base.out_dim  # output shape from gnns
        mlp_base_in_dim = gnn_out_dim + obs_shape[0]
        self.base = MLPBase(args, obs_shape=None, override_obs_dim=mlp_base_in_dim)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            if self._use_lstm:
                self.rnn = LSTMLayer(
                    self.hidden_size,  # input size
                    self.hidden_size, # output size 
                    self._recurrent_N,
                    self._use_orthogonal,
                )
            else:
                self.rnn = RNNLayer(
                    self.hidden_size,
                    self.hidden_size,
                    self._recurrent_N,
                    self._use_orthogonal,
                )

        self.act = ACTLayer(
            action_space, self.hidden_size, self._use_orthogonal, self._gain
        )
        if self.use_graph_cbf_shield:
            action_side = int(round(math.sqrt(action_space.n)))
            if action_side * action_side != action_space.n:
                raise ValueError("Graph-CBF shield requires a square 2-D action table")
            if action_side % 2 == 0:
                warnings.warn(
                    "Even action_grid_size has no exact zero action; an odd grid "
                    "is recommended for DTCBF backup control.",
                    RuntimeWarning,
                )
            action_table = torch.as_tensor(build_action_table_np(action_side))
            self.register_buffer("action_table", action_table)
            self.action_encoder = nn.Sequential(
                nn.Linear(2, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
            self.graph_cbf_head = nn.Sequential(
                nn.Linear(gnn_out_dim + self.hidden_size + 4, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 1),
            )

        self.to(device)

    def set_training_progress(self, progress: float) -> None:
        progress = min(max(float(progress), 0.0), 1.0)
        self.guide_weight = 1.0 / (
            1.0 + math.exp(self.guide_decay_steepness * (progress - 0.5))
        )

    def _local_dtcbf_action_features(self, obs, node_obs, adj, agent_id):
        """Build one-step local DTCBF masks using the environment dynamics."""
        batch_size, num_nodes, _ = node_obs.shape
        action_count = self.action_table.shape[0]
        ego_index = agent_id.long().view(batch_size, -1)[:, 0]
        batch_index = torch.arange(batch_size, device=node_obs.device)
        has_radius_feature = node_obs.shape[-1] >= 6

        if self.graph_feat_type == "global":
            ego_nodes = node_obs[batch_index, ego_index]
            rel_pos = ego_nodes[:, None, 0:2] - node_obs[:, :, 0:2]
            rel_vel = ego_nodes[:, None, 2:4] - node_obs[:, :, 2:4]
            ego_radius = (
                ego_nodes[:, 4]
                if has_radius_feature
                else ego_nodes.new_full((batch_size,), 0.1)
            )
        else:
            rel_pos = node_obs[:, :, 0:2]
            rel_vel = node_obs[:, :, 2:4]
            ego_radius = (
                node_obs[batch_index, ego_index, 4]
                if has_radius_feature
                else node_obs.new_full((batch_size,), 0.1)
            )

        entity_radius = (
            node_obs[:, :, 4]
            if has_radius_feature
            else node_obs.new_full((batch_size, num_nodes), 0.1)
        )
        entity_type = node_obs[:, :, -1].long()
        connected = (adj > 0) & (adj <= self.args.max_edge_dist)
        connected = connected[batch_index, ego_index]
        not_self = (
            torch.arange(num_nodes, device=node_obs.device)[None, :]
            != ego_index[:, None]
        )
        obstacle_mask = (entity_type == 2) | (entity_type == 3)
        agent_mask = entity_type == 0
        constrained_types = torch.zeros_like(entity_type, dtype=torch.bool)
        if self.cbf_include_obstacles:
            constrained_types |= obstacle_mask
        if self.cbf_include_agents_in_local_mask:
            constrained_types |= agent_mask
        constraint_mask = connected & not_self & constrained_types

        clearance = ego_radius[:, None] + entity_radius + self.cbf_safety_buffer
        h_now = rel_pos.square().sum(-1) - clearance.square()
        acceleration = self.action_table * self.cbf_max_accel
        ego_velocity = obs[:, 2:4]
        entity_velocity = ego_velocity[:, None] - rel_vel
        predicted_ego_velocity = ego_velocity[:, None].expand(
            -1, action_count, -1
        )
        predicted_rel_pos = rel_pos[:, None].expand(
            -1, action_count, -1, -1
        )
        predicted_h = h_now[:, None].expand(-1, action_count, -1)
        horizon_margins = []
        for _ in range(max(int(self.cbf_horizon), 1)):
            next_ego_velocity = (
                predicted_ego_velocity + self.cbf_dt * acceleration[None]
            )
            speed = next_ego_velocity.norm(dim=-1, keepdim=True)
            speed_scale = torch.clamp(
                self.cbf_max_speed / speed.clamp_min(1e-8), max=1.0
            )
            next_ego_velocity = next_ego_velocity * speed_scale
            effective_acceleration = (
                next_ego_velocity - predicted_ego_velocity
            ) / self.cbf_dt
            predicted_rel_velocity = (
                predicted_ego_velocity[:, :, None]
                - entity_velocity[:, None]
            )
            next_rel_pos = (
                predicted_rel_pos
                + self.cbf_dt * predicted_rel_velocity
                + 0.5
                * self.cbf_dt ** 2
                * effective_acceleration[:, :, None]
            )
            h_next = (
                next_rel_pos.square().sum(-1) - clearance[:, None].square()
            )
            horizon_margins.append(
                h_next - (1.0 - self.cbf_alpha) * predicted_h
            )
            predicted_ego_velocity = next_ego_velocity
            predicted_rel_pos = next_rel_pos
            predicted_h = h_next
        margins = torch.stack(horizon_margins).min(dim=0).values
        inf = torch.finfo(margins.dtype).max
        masked_margins = margins.masked_fill(~constraint_mask[:, None], inf)
        min_margin = masked_margins.min(dim=-1).values
        no_constraints = ~constraint_mask.any(dim=-1)
        min_margin[no_constraints] = 1.0

        hard_mask = min_margin >= 0.0
        if not self.use_local_dtcbf_shield:
            hard_mask = torch.ones_like(hard_mask)
        h_min = h_now.masked_fill(~constraint_mask, inf).min(dim=-1).values
        h_min[no_constraints] = 1.0
        violations = torch.relu(-margins)
        neighbor_risk = violations.masked_fill(
            ~(constraint_mask & agent_mask)[:, None], 0.0
        ).max(dim=-1).values
        obstacle_risk = violations.masked_fill(
            ~(constraint_mask & obstacle_mask)[:, None], 0.0
        ).max(dim=-1).values

        cbf_features = torch.stack(
            [
                min_margin,
                h_min[:, None].expand(-1, action_count),
                neighbor_risk,
                obstacle_risk,
            ],
            dim=-1,
        )
        return min_margin, hard_mask, cbf_features

    def _cbf_action_features(self, obs, node_obs, adj, agent_id):
        """Backward-compatible alias for the local discrete-time CBF."""
        return self._local_dtcbf_action_features(obs, node_obs, adj, agent_id)

    def _backup_action_indices(self, obs):
        if self.backup_action_mode == "zero":
            target = torch.zeros_like(obs[:, 2:4])
        else:
            target = torch.clamp(
                -obs[:, 2:4] / max(self.cbf_max_accel * self.cbf_dt, 1e-8),
                -1.0,
                1.0,
            )
        distances = (self.action_table[None] - target[:, None]).square().sum(-1)
        return distances.argmin(dim=-1)

    def _make_nonempty_mask(self, mask, margins, obs):
        resolved = mask.clone()
        infeasible = ~resolved.any(dim=-1)
        if not infeasible.any():
            return resolved
        if self.no_safe_action_strategy == "terminate":
            raise RuntimeError("Local DTCBF shield found no feasible action.")
        if self.no_safe_action_strategy == "least_unsafe":
            fallback = margins.argmax(dim=-1)
        else:
            fallback = self._backup_action_indices(obs)
        resolved[infeasible] = False
        resolved[infeasible, fallback[infeasible]] = True
        return resolved

    def _shielded_distribution(
        self, actor_features, graph_emb, obs, node_obs, adj, agent_id,
        available_actions=None,
    ):
        base_logits = self.act.action_out.linear(actor_features)
        margins, hard_mask, cbf_features = self._local_dtcbf_action_features(
            obs, node_obs, adj, agent_id
        )
        batch_size, action_count = margins.shape
        action_emb = self.action_encoder(self.action_table)
        action_emb = action_emb[None].expand(batch_size, -1, -1)
        graph_context = graph_emb[:, None].expand(-1, action_count, -1)
        safety_input = torch.cat([graph_context, action_emb, cbf_features], dim=-1)
        safety_score = self.graph_cbf_head(safety_input).squeeze(-1)
        if self.use_joint_dtcbf_shield and available_actions is not None:
            final_mask = available_actions.bool()
        else:
            final_mask = hard_mask
            if available_actions is not None:
                final_mask = final_mask & available_actions.bool()
            final_mask = self._make_nonempty_mask(final_mask, margins, obs)
        logits = base_logits + self.safety_score_coef * safety_score
        logits = logits.masked_fill(
            ~final_mask, categorical_mask_value(logits)
        )
        return FixedCategorical(logits=logits), margins, hard_mask, safety_score

    def _extract_actor_features(
        self, obs, node_obs, adj, agent_id, rnn_states, masks
    ):
        if self.split_batch and obs.shape[0] > self.max_batch_size:
            actor_features = []
            graph_embeddings = []
            for batch in minibatchGenerator(
                obs, node_obs, adj, agent_id, self.max_batch_size
            ):
                obs_batch, node_obs_batch, adj_batch, agent_id_batch = batch
                if node_obs_batch.shape[0] > 0:
                    graph_batch = self.gnn_base(
                        node_obs_batch, adj_batch, agent_id_batch
                    )
                    actor_features.append(
                        self.base(torch.cat([obs_batch, graph_batch], dim=1))
                    )
                    graph_embeddings.append(graph_batch)
            actor_features = torch.cat(actor_features, dim=0)
            graph_emb = torch.cat(graph_embeddings, dim=0)
        else:
            graph_emb = self.gnn_base(node_obs, adj, agent_id)
            actor_features = self.base(torch.cat([obs, graph_emb], dim=1))
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        return actor_features, graph_emb, rnn_states

    def get_logits_and_local_masks(
        self, obs, node_obs, adj, agent_id, rnn_states, masks
    ):
        obs = check(obs).to(**self.tpdv)
        node_obs = check(node_obs).to(**self.tpdv)
        adj = check(adj).to(**self.tpdv)
        agent_id = check(agent_id).to(**self.tpdv).long()
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        actor_features, graph_emb, rnn_states = self._extract_actor_features(
            obs, node_obs, adj, agent_id, rnn_states, masks
        )
        base_logits = self.act.action_out.linear(actor_features)
        margins, local_mask, cbf_features = self._local_dtcbf_action_features(
            obs, node_obs, adj, agent_id
        )
        action_count = margins.shape[-1]
        action_emb = self.action_encoder(self.action_table)[None].expand(
            margins.shape[0], -1, -1
        )
        graph_context = graph_emb[:, None].expand(-1, action_count, -1)
        safety_input = torch.cat(
            [graph_context, action_emb, cbf_features], dim=-1
        )
        safety_score = self.graph_cbf_head(safety_input).squeeze(-1)
        logits = base_logits + self.safety_score_coef * safety_score
        return logits, local_mask, margins, safety_score, rnn_states

    def _safe_guide_actions(self, obs, hard_mask):
        goal_error = obs[:, 4:6]
        velocity = obs[:, 2:4]
        guide_u = torch.clamp(1.5 * goal_error - 1.6 * velocity, -1.0, 1.0)
        distances = (self.action_table[None] - guide_u[:, None]).square().sum(-1)
        distances = distances.masked_fill(~hard_mask, torch.finfo(distances.dtype).max)
        return distances.argmin(dim=-1, keepdim=True)

    def forward(
        self,
        obs,
        node_obs,
        adj,
        agent_id,
        rnn_states,
        masks,
        available_actions=None,
        deterministic=False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute actions from the given inputs.
        obs: (np.ndarray / torch.Tensor)
            Observation inputs into network.
        node_obs (np.ndarray / torch.Tensor):
            Local agent graph node features to the actor.
        adj (np.ndarray / torch.Tensor):
            Adjacency matrix for the graph
        agent_id (np.ndarray / torch.Tensor)
            The agent id to which the observation belongs to
        rnn_states: (np.ndarray / torch.Tensor)
            If RNN network, hidden states for RNN.
        masks: (np.ndarray / torch.Tensor)
            Mask tensor denoting if hidden states
            should be reinitialized to zeros.
        available_actions: (np.ndarray / torch.Tensor)
            Denotes which actions are available to agent
            (if None, all actions available)
        deterministic: (bool)
            Whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor)
            Actions to take.
        :return action_log_probs: (torch.Tensor)
            Log probabilities of taken actions.
        :return rnn_states: (torch.Tensor)
            Updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        node_obs = check(node_obs).to(**self.tpdv)
        adj = check(adj).to(**self.tpdv)
        agent_id = check(agent_id).to(**self.tpdv).long()
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features, graph_emb, rnn_states = self._extract_actor_features(
            obs, node_obs, adj, agent_id, rnn_states, masks
        )

        if self.use_graph_cbf_shield:
            dist, _, _, _ = self._shielded_distribution(
                actor_features, graph_emb, obs, node_obs, adj, agent_id,
                available_actions,
            )
            actions = dist.mode() if deterministic else dist.sample()
            action_log_probs = dist.log_probs(actions)
        else:
            actions, action_log_probs = self.act(
                actor_features, available_actions, deterministic
            )

        return (actions, action_log_probs, rnn_states)

    def evaluate_actions(
        self,
        obs,
        node_obs,
        adj,
        agent_id,
        rnn_states,
        action,
        masks,
        available_actions=None,
        active_masks=None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute log probability and entropy of given actions.
        obs: (torch.Tensor)
            Observation inputs into network.
        node_obs (torch.Tensor):
            Local agent graph node features to the actor.
        adj (torch.Tensor):
            Adjacency matrix for the graph.
        agent_id (np.ndarray / torch.Tensor)
            The agent id to which the observation belongs to
        action: (torch.Tensor)
            Actions whose entropy and log probability to evaluate.
        rnn_states: (torch.Tensor)
            If RNN network, hidden states for RNN.
        masks: (torch.Tensor)
            Mask tensor denoting if hidden states
            should be reinitialized to zeros.
        available_actions: (torch.Tensor)
            Denotes which actions are available to agent
            (if None, all actions available)
        active_masks: (torch.Tensor)
            Denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor)
            Log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor)
            Action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        node_obs = check(node_obs).to(**self.tpdv)
        adj = check(adj).to(**self.tpdv)
        agent_id = check(agent_id).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        # if batch size is big, split into smaller batches, forward pass and then concatenate
        if (self.split_batch) and (obs.shape[0] > self.max_batch_size):
            # print(f'eval Actor obs: {obs.shape[0]}')
            batchGenerator = minibatchGenerator(obs, node_obs, adj, agent_id, self.max_batch_size)
            actor_features = []
            graph_embeddings = []
            for batch in batchGenerator:
                obs_batch, node_obs_batch, adj_batch, agent_id_batch = batch
                if node_obs_batch.shape[0] > 0:    
                    nbd_feats_batch = self.gnn_base(node_obs_batch, adj_batch, agent_id_batch)
                    act_feats_batch = torch.cat([obs_batch, nbd_feats_batch], dim=1)
                    actor_feats_batch = self.base(act_feats_batch)
                    actor_features.append(actor_feats_batch)
                    graph_embeddings.append(nbd_feats_batch)
            actor_features = torch.cat(actor_features, dim=0)
            graph_emb = torch.cat(graph_embeddings, dim=0)
        else:
            nbd_features = self.gnn_base(node_obs, adj, agent_id)
            graph_emb = nbd_features
            actor_features = torch.cat([obs, nbd_features], dim=1)
            actor_features = self.base(actor_features)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self.use_graph_cbf_shield:
            dist, margins, hard_mask, safety_score = self._shielded_distribution(
                actor_features, graph_emb, obs, node_obs, adj, agent_id,
                available_actions,
            )
            action_log_probs = dist.log_probs(action)
            entropy = dist.entropy()
            if active_masks is not None and self._use_policy_active_masks:
                dist_entropy = (
                    entropy * active_masks.squeeze(-1)
                ).sum() / active_masks.sum().clamp_min(1.0)
            else:
                dist_entropy = entropy.mean()
            local_loss_mask = self._make_nonempty_mask(hard_mask, margins, obs)
            local_logits = (
                self.act.action_out.linear(actor_features)
                + self.safety_score_coef * safety_score
            )
            local_logits = local_logits.masked_fill(
                ~local_loss_mask, categorical_mask_value(local_logits)
            )
            local_dist = FixedCategorical(logits=local_logits)
            guide_actions = self._safe_guide_actions(obs, local_loss_mask)
            guide_loss = -local_dist.log_probs(guide_actions).mean()
            safe_float = local_loss_mask.float()
            margin_min = margins.masked_fill(
                ~local_loss_mask, torch.finfo(margins.dtype).max
            ).min(-1, keepdim=True).values
            margin_max = margins.masked_fill(
                ~local_loss_mask, torch.finfo(margins.dtype).min
            ).max(-1, keepdim=True).values
            rank_target = (margins - margin_min) / (margin_max - margin_min).clamp_min(1e-6)
            rank_target = torch.where(
                local_loss_mask, rank_target, torch.zeros_like(rank_target)
            )
            safety_rank_loss = (
                (safety_score - rank_target).square() * safe_float
            ).sum() / safe_float.sum().clamp_min(1.0)
        else:
            action_log_probs, dist_entropy = self.act.evaluate_actions(
                actor_features,
                action,
                available_actions,
                active_masks=active_masks if self._use_policy_active_masks else None,
            )
            guide_loss = actor_features.new_zeros(())
            safety_rank_loss = actor_features.new_zeros(())

        return (action_log_probs, dist_entropy, guide_loss, safety_rank_loss)


class GR_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions
    given centralized input (MAPPO) or local observations (IPPO).
    args: (argparse.Namespace)
        Arguments containing relevant model information.
    cent_obs_space: (gym.Space)
        (centralized) observation space.
    node_obs_space: (gym.Space)
        node observation space.
    edge_obs_space: (gym.Space)
        edge observation space.
    device: (torch.device)
        Specifies the device to run on (cpu/gpu).
    split_batch: (bool)
        Whether to split a big-batch into multiple
        smaller ones to speed up forward pass.
    max_batch_size: (int)
        Maximum batch size to use.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        cent_obs_space: gym.Space,
        node_obs_space: gym.Space,
        edge_obs_space: gym.Space,
        device=torch.device("cpu"),
        split_batch: bool = False,
        max_batch_size: int = 32,
    ) -> None:
        super(GR_Critic, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.use_att_gnn = args.use_att_gnn
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_lstm = args.use_lstm
        self._use_popart = args.use_popart
        self.split_batch = split_batch
        self.max_batch_size = max_batch_size
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][
            self._use_orthogonal
        ]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        node_obs_shape = get_shape_from_obs_space(node_obs_space)[
            1
        ]  # (num_nodes, num_node_feats)
        edge_dim = get_shape_from_obs_space(edge_obs_space)[0]  # (edge_dim,)

        if self.use_att_gnn:
            from onpolicy.algorithms.utils.gnn import GNNBase
        else:
            from onpolicy.algorithms.utils.gnn_transformer import GNNBase
        self.gnn_base = GNNBase(args, node_obs_shape, edge_dim, args.critic_graph_aggr)
        gnn_out_dim = self.gnn_base.out_dim
        # if node aggregation, then concatenate aggregated node features for all agents
        # otherwise, the aggregation is done for the whole graph
        if args.critic_graph_aggr == "node":
            gnn_out_dim *= args.num_agents
        mlp_base_in_dim = gnn_out_dim
        if self.args.use_cent_obs:
            mlp_base_in_dim += cent_obs_shape[0]

        self.base = MLPBase(args, cent_obs_shape, override_obs_dim=mlp_base_in_dim)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            if self._use_lstm:
                self.rnn = LSTMLayer(
                    self.hidden_size,
                    self.hidden_size,
                    self._recurrent_N,
                    self._use_orthogonal,
                )
            else:
                self.rnn = RNNLayer(
                    self.hidden_size,
                    self.hidden_size,
                    self._recurrent_N,
                    self._use_orthogonal,
                )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(
        self, cent_obs, node_obs, adj, agent_id, rnn_states, masks
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute actions from the given inputs.
        cent_obs: (np.ndarray / torch.Tensor)
            Observation inputs into network.
        node_obs (np.ndarray):
            Local agent graph node features to the actor.
        adj (np.ndarray):
            Adjacency matrix for the graph.
        agent_id (np.ndarray / torch.Tensor)
            The agent id to which the observation belongs to
        rnn_states: (np.ndarray / torch.Tensor)
            If RNN network, hidden states for RNN.
        masks: (np.ndarray / torch.Tensor)
            Mask tensor denoting if RNN states
            should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        node_obs = check(node_obs).to(**self.tpdv)
        adj = check(adj).to(**self.tpdv)
        agent_id = check(agent_id).to(**self.tpdv).long()
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        # if batch size is big, split into smaller batches, forward pass and then concatenate
        if (self.split_batch) and (cent_obs.shape[0] > self.max_batch_size):
            # print(f'Cent obs: {cent_obs.shape[0]}')
            batchGenerator = minibatchGenerator(cent_obs, node_obs, adj, agent_id, self.max_batch_size)
            critic_features = []
            for batch in batchGenerator:
                obs_batch, node_obs_batch, adj_batch, agent_id_batch = batch
                if node_obs_batch.shape[0] > 0:
                    # print("node_obs_batch: ", node_obs_batch.shape)
                    nbd_feats_batch = self.gnn_base(node_obs_batch, adj_batch, agent_id_batch)
                    if self.args.use_cent_obs:
                        act_feats_batch = torch.cat([obs_batch, nbd_feats_batch], dim=1)
                    else: 
                        act_feats_batch = nbd_feats_batch
                    critic_feats_batch = self.base(act_feats_batch)
                    critic_features.append(critic_feats_batch)
            critic_features = torch.cat(critic_features, dim=0)
        else:
            nbd_features = self.gnn_base(node_obs, adj, agent_id)  # CHECK from where are these agent_ids coming
            if self.args.use_cent_obs:
                critic_features = torch.cat([cent_obs, nbd_features], dim=1)  # NOTE can remove concatenation with cent_obs and just use graph_feats
            else:
                critic_features = nbd_features
            critic_features = self.base(critic_features)  # Cent obs here

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return (values, rnn_states)
