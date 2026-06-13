import torch
import gym
import argparse
import numpy as np
from numpy import ndarray as arr
from typing import Optional, Tuple, Generator
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space, get_shape_from_act_space


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


class GSReplayBuffer(object):
    """
    Buffer to store training data. For graph-based environments
    args: (argparse.Namespace)
        arguments containing relevant model, policy, and env information.
    num_agents: (int)
        number of agents in the env.
    num_entities: (int)
        number of entities in the env. This will be used for the `edge_list`
        size and `node_feats`
    obs_space: (gym.Space)
        observation space of agents.
    cent_obs_space: (gym.Space)
        centralized observation space of agents.
    node_obs_space: (gym.Space)
        node observation space of agents.
    agent_id_space: (gym.Space)
        observation space of agent ids.
    share_agent_id_space: (gym.Space)
        centralised observation space of agent ids.
    adj_space: (gym.Space)
        observation space of adjacency matrix.
    act_space: (gym.Space)
        action space for agents.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        num_agents: int,
        obs_space: gym.Space,
        cent_obs_space: gym.Space,
        node_obs_space: gym.Space,
        agent_id_space: gym.Space,
        share_agent_id_space: gym.Space,
        adj_space: gym.Space,
        act_space: gym.Space,
    ):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = 2 * args.hidden_size if args.use_lstm else args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits

        # get shapes of observations
        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(cent_obs_space)
        node_obs_shape = get_shape_from_obs_space(node_obs_space)
        agent_id_shape = get_shape_from_obs_space(agent_id_space)
        if args.use_centralized_V:
            share_agent_id_shape = get_shape_from_obs_space(share_agent_id_space)
        else:
            share_agent_id_shape = get_shape_from_obs_space(agent_id_space)
        adj_shape = get_shape_from_obs_space(adj_space)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]

        # Observation and graph-related storage
        self.share_obs = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                num_agents,
                *share_obs_shape,
            ),
            dtype=np.float32,
        )
        self.obs = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape),
            dtype=np.float32,
        )
        self.node_obs = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                num_agents,
                *node_obs_shape,
            ),
            dtype=np.float32,
        )
        self.adj = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, *adj_shape),
            dtype=np.float32,
        )
        self.agent_id = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                num_agents,
                *agent_id_shape,
            ),
            dtype=np.int32,
        )
        self.share_agent_id = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                num_agents,
                *share_agent_id_shape,
            ),
            dtype=np.int32,
        )

        # RNN states
        self.rnn_states = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        self.rnn_states_critic = np.zeros_like(self.rnn_states)
        self.rnn_states_cost = np.zeros_like(self.rnn_states)  # Added for cost critic RNN states

        # Value and return storage
        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1),
            dtype=np.float32,
        )
        self.returns = np.zeros_like(self.value_preds)
        self.cost_preds = np.zeros_like(self.value_preds)  # Added for cost value predictions
        self.cost_returns = np.zeros_like(self.returns)  # Added for cost returns

        # Action-related storage
        if act_space.__class__.__name__ == "Discrete":
            self.available_actions = np.ones(
                (
                    self.episode_length + 1,
                    self.n_rollout_threads,
                    num_agents,
                    act_space.n,
                ),
                dtype=np.float32,
            )
        elif act_space.__class__.__name__ == "MultiDiscrete": # our MAS control
            self.available_actions = np.ones(
                (
                    self.episode_length + 1,
                    self.n_rollout_threads,
                    num_agents,
                    act_space.num_discrete_space,
                ),
                dtype=np.float32,
            )
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape),
            dtype=np.float32,
        )
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape),
            dtype=np.float32,
        )
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1),
            dtype=np.float32,
        )
        self.costs = np.zeros_like(self.rewards)  # Added for cost storage

        # Masks
        self.masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1),
            dtype=np.float32,
        )
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        # # Average episode costs
        self.aver_episode_costs = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1),
            dtype=np.float32,
        )  # Added for average episode costs. Copy format of returns
        # self.aver_episode_costs = np.zeros((self.n_rollout_threads, num_agents),
        #     dtype=np.float32,
        # )  # Added for average episode costs

        self.factor = None
        self.step = 0

    def update_factor(self, factor):
        self.factor = factor.copy()

    def return_aver_insert(self, aver_episode_costs):
        self.aver_episode_costs = aver_episode_costs.copy()

    def insert(
        self,
        share_obs: np.ndarray,
        obs: np.ndarray,
        node_obs: np.ndarray,
        adj: np.ndarray,
        agent_id: np.ndarray,
        share_agent_id: np.ndarray,
        rnn_states_actor: np.ndarray,
        rnn_states_critic: np.ndarray,
        actions: np.ndarray,
        action_log_probs: np.ndarray,
        value_preds: np.ndarray,
        rewards: np.ndarray,
        masks: np.ndarray,
        bad_masks: np.ndarray = None,
        active_masks: np.ndarray = None,
        available_actions: np.ndarray = None,
        costs: np.ndarray = None,
        cost_preds: np.ndarray = None,
        rnn_states_cost: np.ndarray = None,
    ) -> None:
        """
        Insert data into the replay buffer, including safety-related data.
        """
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.node_obs[self.step + 1] = node_obs.copy()
        self.adj[self.step + 1] = adj.copy()
        self.agent_id[self.step + 1] = agent_id.copy()
        self.share_agent_id[self.step + 1] = share_agent_id.copy()
        self.rnn_states[self.step + 1] = rnn_states_actor.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()
        if costs is not None:
            self.costs[self.step] = costs.copy()
        if cost_preds is not None:
            self.cost_preds[self.step] = cost_preds.copy()
        if rnn_states_cost is not None:
            self.rnn_states_cost[self.step + 1] = rnn_states_cost.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self) -> None:
        """Copy last timestep data to first index. Called after update to model."""
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.node_obs[0] = self.node_obs[-1].copy()
        self.adj[0] = self.adj[-1].copy()
        self.agent_id[0] = self.agent_id[-1].copy()
        self.share_agent_id[0] = self.share_agent_id[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.rnn_states_cost[0] = self.rnn_states_cost[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def compute_returns(
        self, next_value: arr, value_normalizer: Optional[PopArt] = None
    ) -> None:
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        next_value: (np.ndarray)
            value predictions for the step after the last episode step.
        value_normalizer: (PopArt)
            If not None, PopArt value normalizer instance.
        """
        if self._use_proper_time_limits:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        # step + 1
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * value_normalizer.denormalize(self.value_preds[step + 1])
                            * self.masks[step + 1]
                            - value_normalizer.denormalize(self.value_preds[step])
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * gae * self.masks[step + 1]
                        )
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * self.value_preds[step + 1]
                            * self.masks[step + 1]
                            - self.value_preds[step]
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        self.returns[step] = (
                            self.returns[step + 1] * self.gamma * self.masks[step + 1]
                            + self.rewards[step]
                        ) * self.bad_masks[step + 1] + (
                            1 - self.bad_masks[step + 1]
                        ) * value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:
                        self.returns[step] = (
                            self.returns[step + 1] * self.gamma * self.masks[step + 1]
                            + self.rewards[step]
                        ) * self.bad_masks[step + 1] + (
                            1 - self.bad_masks[step + 1]
                        ) * self.value_preds[
                            step
                        ]
        else:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * value_normalizer.denormalize(self.value_preds[step + 1])
                            * self.masks[step + 1]
                            - value_normalizer.denormalize(self.value_preds[step])
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        self.returns[step] = gae + value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * self.value_preds[step + 1]
                            * self.masks[step + 1]
                            - self.value_preds[step]
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = (
                        self.returns[step + 1] * self.gamma * self.masks[step + 1]
                        + self.rewards[step]
                    )

    def compute_cost_returns(
        self, next_cost: np.ndarray, value_normalizer: Optional[PopArt] = None
    ) -> None:
        """
        Compute cost returns either as discounted sum of costs, or using GAE.
        
        Args:
            next_cost (np.ndarray): Cost predictions for the step after the last episode step.
            value_normalizer (PopArt, optional): If not None, PopArt value normalizer instance.
        """
        if self._use_proper_time_limits:
            if self._use_gae:
                self.cost_preds[-1] = next_cost
                gae = 0
                for step in reversed(range(self.costs.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = (
                            self.costs[step]
                            + self.gamma
                            * value_normalizer.denormalize(self.cost_preds[step + 1])
                            * self.masks[step + 1]
                            - value_normalizer.denormalize(self.cost_preds[step])
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * gae * self.masks[step + 1]
                        )
                        gae = gae * self.bad_masks[step + 1]
                        self.cost_returns[step] = gae + value_normalizer.denormalize(
                            self.cost_preds[step]
                        )
                    else:
                        delta = (
                            self.costs[step]
                            + self.gamma
                            * self.cost_preds[step + 1]
                            * self.masks[step + 1]
                            - self.cost_preds[step]
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        gae = gae * self.bad_masks[step + 1]
                        self.cost_returns[step] = gae + self.cost_preds[step]
            else:
                self.cost_returns[-1] = next_cost
                for step in reversed(range(self.costs.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        self.cost_returns[step] = (
                            self.cost_returns[step + 1] * self.gamma * self.masks[step + 1]
                            + self.costs[step]
                        ) * self.bad_masks[step + 1] + (
                            1 - self.bad_masks[step + 1]
                        ) * value_normalizer.denormalize(
                            self.cost_preds[step]
                        )
                    else:
                        self.cost_returns[step] = (
                            self.cost_returns[step + 1] * self.gamma * self.masks[step + 1]
                            + self.costs[step]
                        ) * self.bad_masks[step + 1] + (
                            1 - self.bad_masks[step + 1]
                        ) * self.cost_preds[
                            step
                        ]
        else:
            if self._use_gae:
                self.cost_preds[-1] = next_cost
                # print("next_cost in buffer is: ", next_cost)  # DEBUG
                gae = 0
                for step in reversed(range(self.costs.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = (
                            self.costs[step]
                            + self.gamma
                            * value_normalizer.denormalize(self.cost_preds[step + 1])
                            * self.masks[step + 1]
                            - value_normalizer.denormalize(self.cost_preds[step])
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        self.cost_returns[step] = gae + value_normalizer.denormalize(
                            self.cost_preds[step]
                        )
                        # print("cost_returns[{}] is: {}, vc-denorm:{}".format(step, self.cost_returns[step], value_normalizer.denormalize(self.cost_preds[step])))
                    else:
                        delta = (
                            self.costs[step]
                            + self.gamma
                            * self.cost_preds[step + 1]
                            * self.masks[step + 1]
                            - self.cost_preds[step]
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        self.cost_returns[step] = gae + self.cost_preds[step]
            else:
                self.cost_returns[-1] = next_cost
                for step in reversed(range(self.costs.shape[0])):
                    self.cost_returns[step] = (
                        self.cost_returns[step + 1] * self.gamma * self.masks[step + 1]
                        + self.costs[step]
                    )

    def compute_average_episode_costs(self):
        """
        Compute average episode costs for each agent across all rollout threads.
        """
        # Sum costs over the episode length for each agent in each rollout thread
        total_costs = np.sum(self.costs, axis=0)
        # Compute average costs by dividing by the number of steps
        for step in reversed(range(self.aver_episode_costs.shape[0])):
            self.aver_episode_costs[step] = total_costs

    def feed_forward_generator(
        self,
        advantages: np.ndarray,
        num_mini_batch: Optional[int] = None,
        mini_batch_size: Optional[int] = None,
        cost_adv: Optional[np.ndarray] = None,
    ) -> Generator[
        Tuple[
            np.ndarray,  # share_obs_batch
            np.ndarray,  # obs_batch
            np.ndarray,  # node_obs_batch
            np.ndarray,  # adj_batch
            np.ndarray,  # agent_id_batch
            np.ndarray,  # share_agent_id_batch
            np.ndarray,  # rnn_states_batch
            np.ndarray,  # rnn_states_critic_batch
            np.ndarray,  # actions_batch
            np.ndarray,  # value_preds_batch
            np.ndarray,  # return_batch
            np.ndarray,  # masks_batch
            np.ndarray,  # active_masks_batch
            np.ndarray,  # old_action_log_probs_batch
            np.ndarray,  # adv_targ
            np.ndarray,  # available_actions_batch
            Optional[np.ndarray],  # factor_batch
            np.ndarray,  # cost_preds_batch
            np.ndarray,  # cost_return_batch
            np.ndarray,  # rnn_states_cost_batch
            np.ndarray,  # cost_adv_targ
            np.ndarray,  # aver_episode_costs
        ],
        None,
        None,
    ]:
        """
        Yield training data for MLP policies, including safety-related data.
        
        Args:
            advantages (np.ndarray): Advantage estimates.
            num_mini_batch (int, optional): Number of minibatches to split the batch into.
            mini_batch_size (int, optional): Number of samples in each minibatch.
            cost_adv (np.ndarray, optional): Cost advantage estimates.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                f"PPO requires the number of processes ({n_rollout_threads}) "
                f"* number of steps ({episode_length}) * number of agents "
                f"({num_agents}) = {n_rollout_threads*episode_length*num_agents} "
                "to be greater than or equal to the number of "
                f"PPO mini batches ({num_mini_batch})."
            )
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(num_mini_batch)
        ]

        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[3:])
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[3:])
        node_obs = self.node_obs[:-1].reshape(-1, *self.node_obs.shape[3:])
        adj = self.adj[:-1].reshape(-1, *self.adj.shape[3:])
        agent_id = self.agent_id[:-1].reshape(-1, *self.agent_id.shape[3:])
        share_agent_id = self.share_agent_id[:-1].reshape(
            -1, *self.share_agent_id.shape[3:]
        )
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(
            -1, *self.rnn_states_critic.shape[3:]
        )
        rnn_states_cost = self.rnn_states_cost[:-1].reshape(
            -1, *self.rnn_states_cost.shape[3:]
        )
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(
                -1, self.available_actions.shape[-1]
            )
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        cost_preds = self.cost_preds[:-1].reshape(-1, 1)
        cost_returns = self.cost_returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(
            -1, self.action_log_probs.shape[-1]
        )
        advantages = advantages.reshape(-1, 1)
        if cost_adv is not None:
            cost_adv = cost_adv.reshape(-1, 1)
        if self.factor is not None:
            factor = self.factor.reshape(-1, 1)
        # aver_episode_costs = self.aver_episode_costs[:-1].reshape(-1, *self.aver_episode_costs.shape[3:])
        aver_episode_costs = self.aver_episode_costs[:-1].reshape(-1, 1)

        for indices in sampler:
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            node_obs_batch = node_obs[indices]
            adj_batch = adj[indices]
            agent_id_batch = agent_id[indices]
            share_agent_id_batch = share_agent_id[indices]
            rnn_states_batch = rnn_states[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            rnn_states_cost_batch = rnn_states_cost[indices]
            actions_batch = actions[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            cost_preds_batch = cost_preds[indices]
            cost_return_batch = cost_returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]
            if cost_adv is None:
                cost_adv_targ = None
            else:
                cost_adv_targ = cost_adv[indices]
            if self.factor is None:
                factor_batch = None
            else:
                factor_batch = factor[indices]

            yield (
                share_obs_batch,
                obs_batch,
                node_obs_batch,
                adj_batch,
                agent_id_batch,
                share_agent_id_batch,
                rnn_states_batch,
                rnn_states_critic_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                active_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                available_actions_batch,
                factor_batch,
                cost_preds_batch,
                cost_return_batch,
                rnn_states_cost_batch,
                cost_adv_targ,
                aver_episode_costs,
            )

    def naive_recurrent_generator(
        self, advantages: np.ndarray, num_mini_batch: int, cost_adv: Optional[np.ndarray] = None
    ) -> Generator[
        Tuple[
            np.ndarray,  # share_obs_batch
            np.ndarray,  # obs_batch
            np.ndarray,  # node_obs_batch
            np.ndarray,  # adj_batch
            np.ndarray,  # agent_id_batch
            np.ndarray,  # share_agent_id_batch
            np.ndarray,  # rnn_states_batch
            np.ndarray,  # rnn_states_critic_batch
            np.ndarray,  # actions_batch
            np.ndarray,  # value_preds_batch
            np.ndarray,  # return_batch
            np.ndarray,  # masks_batch
            np.ndarray,  # active_masks_batch
            np.ndarray,  # old_action_log_probs_batch
            np.ndarray,  # adv_targ
            np.ndarray,  # available_actions_batch
            Optional[np.ndarray],  # factor_batch
            np.ndarray,  # cost_preds_batch
            np.ndarray,  # cost_return_batch
            np.ndarray,  # rnn_states_cost_batch
            np.ndarray,  # cost_adv_targ
            np.ndarray,  # aver_episode_costs
        ],
        None,
        None,
    ]:
        """
        Yield training data for non-chunked RNN training, including safety-related data.
        
        Args:
            advantages (np.ndarray): Advantage estimates.
            num_mini_batch (int): Number of minibatches to split the batch into.
            cost_adv (np.ndarray, optional): Cost advantage estimates.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * num_agents
        assert n_rollout_threads * num_agents >= num_mini_batch, (
            "PPO requires the number of processes ({})* number of agents ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(
                n_rollout_threads, num_agents, num_mini_batch
            )
        )
        num_envs_per_batch = batch_size // num_mini_batch
        perm = torch.randperm(batch_size).numpy()

        share_obs = self.share_obs.reshape(-1, batch_size, *self.share_obs.shape[3:])
        obs = self.obs.reshape(-1, batch_size, *self.obs.shape[3:])
        node_obs = self.node_obs.reshape(-1, batch_size, *self.node_obs.shape[3:])
        adj = self.adj.reshape(-1, batch_size, *self.adj.shape[3:])
        agent_id = self.agent_id.reshape(-1, batch_size, *self.agent_id.shape[3:])
        share_agent_id = self.share_agent_id.reshape(
            -1, batch_size, *self.share_agent_id.shape[3:]
        )
        rnn_states = self.rnn_states.reshape(-1, batch_size, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic.reshape(
            -1, batch_size, *self.rnn_states_critic.shape[3:]
        )
        rnn_states_cost = self.rnn_states_cost.reshape(
            -1, batch_size, *self.rnn_states_cost.shape[3:]
        )
        actions = self.actions.reshape(-1, batch_size, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions.reshape(
                -1, batch_size, self.available_actions.shape[-1]
            )
        value_preds = self.value_preds.reshape(-1, batch_size, 1)
        returns = self.returns.reshape(-1, batch_size, 1)
        cost_preds = self.cost_preds.reshape(-1, batch_size, 1)
        cost_returns = self.cost_returns.reshape(-1, batch_size, 1)
        masks = self.masks.reshape(-1, batch_size, 1)
        active_masks = self.active_masks.reshape(-1, batch_size, 1)
        action_log_probs = self.action_log_probs.reshape(
            -1, batch_size, self.action_log_probs.shape[-1]
        )
        advantages = advantages.reshape(-1, batch_size, 1)
        if cost_adv is not None:
            cost_adv = cost_adv.reshape(-1, batch_size, 1)
        if self.factor is not None:
            factor = self.factor.reshape(-1, batch_size, 1)
        # aver_episode_costs = self.aver_episode_costs.reshape(-1, batch_size, *self.aver_episode_costs.shape[3:])
        aver_episode_costs = self.aver_episode_costs.reshape(-1, batch_size, 1)
                                                             
        for start_ind in range(0, batch_size, num_envs_per_batch):
            share_obs_batch = []
            obs_batch = []
            node_obs_batch = []
            adj_batch = []
            agent_id_batch = []
            share_agent_id_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            rnn_states_cost_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            cost_preds_batch = []
            cost_return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            cost_adv_targ = []
            factor_batch = []
            aver_episode_costs_batch = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                share_obs_batch.append(share_obs[:-1, ind])
                obs_batch.append(obs[:-1, ind])
                node_obs_batch.append(node_obs[:-1, ind])
                adj_batch.append(adj[:-1, ind])
                agent_id_batch.append(agent_id[:-1, ind])
                share_agent_id_batch.append(share_agent_id[:-1, ind])
                rnn_states_batch.append(rnn_states[0:1, ind])
                rnn_states_critic_batch.append(rnn_states_critic[0:1, ind])
                rnn_states_cost_batch.append(rnn_states_cost[0:1, ind])
                actions_batch.append(actions[:, ind])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[:-1, ind])
                value_preds_batch.append(value_preds[:-1, ind])
                return_batch.append(returns[:-1, ind])
                cost_preds_batch.append(cost_preds[:-1, ind])
                cost_return_batch.append(cost_returns[:-1, ind])
                masks_batch.append(masks[:-1, ind])
                active_masks_batch.append(active_masks[:-1, ind])
                old_action_log_probs_batch.append(action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])
                if cost_adv is not None:
                    cost_adv_targ.append(cost_adv[:, ind])
                if self.factor is not None:
                    factor_batch.append(factor[:, ind])
                aver_episode_costs_batch.append(aver_episode_costs[:-1, ind])

            T, N = self.episode_length, num_envs_per_batch
            share_obs_batch = np.stack(share_obs_batch, 1)
            obs_batch = np.stack(obs_batch, 1)
            node_obs_batch = np.stack(node_obs_batch, 1)
            adj_batch = np.stack(adj_batch, 1)
            agent_id_batch = np.stack(agent_id_batch, 1)
            share_agent_id_batch = np.stack(share_agent_id_batch, 1)
            actions_batch = np.stack(actions_batch, 1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, 1)
            value_preds_batch = np.stack(value_preds_batch, 1)
            return_batch = np.stack(return_batch, 1)
            cost_preds_batch = np.stack(cost_preds_batch, 1)
            cost_return_batch = np.stack(cost_return_batch, 1)
            masks_batch = np.stack(masks_batch, 1)
            active_masks_batch = np.stack(active_masks_batch, 1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, 1)
            adv_targ = np.stack(adv_targ, 1)
            if cost_adv is not None:
                cost_adv_targ = np.stack(cost_adv_targ, 1)
            if self.factor is not None:
                factor_batch = np.stack(factor_batch, 1)
            aver_episode_costs_batch = np.stack(aver_episode_costs_batch, 1)

            rnn_states_batch = np.stack(rnn_states_batch).reshape(
                N, *self.rnn_states.shape[3:]
            )
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(
                N, *self.rnn_states_critic.shape[3:]
            )
            rnn_states_cost_batch = np.stack(rnn_states_cost_batch).reshape(
                N, *self.rnn_states_cost.shape[3:]
            )

            share_obs_batch = _flatten(T, N, share_obs_batch)
            obs_batch = _flatten(T, N, obs_batch)
            node_obs_batch = _flatten(T, N, node_obs_batch)
            adj_batch = _flatten(T, N, adj_batch)
            agent_id_batch = _flatten(T, N, agent_id_batch)
            share_agent_id_batch = _flatten(T, N, share_agent_id_batch)
            actions_batch = _flatten(T, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(T, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(T, N, value_preds_batch)
            return_batch = _flatten(T, N, return_batch)
            cost_preds_batch = _flatten(T, N, cost_preds_batch)
            cost_return_batch = _flatten(T, N, cost_return_batch)
            masks_batch = _flatten(T, N, masks_batch)
            active_masks_batch = _flatten(T, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(T, N, old_action_log_probs_batch)
            adv_targ = _flatten(T, N, adv_targ)
            if cost_adv is not None:
                cost_adv_targ = _flatten(T, N, cost_adv_targ)
            else:
                cost_adv_targ = None
            if self.factor is not None:
                factor_batch = _flatten(T, N, factor_batch)
            else:
                factor_batch = None
            aver_episode_costs_batch = _flatten(T, N, aver_episode_costs_batch)

            yield (
                share_obs_batch,
                obs_batch,
                node_obs_batch,
                adj_batch,
                agent_id_batch,
                share_agent_id_batch,
                rnn_states_batch,
                rnn_states_critic_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                active_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                available_actions_batch,
                factor_batch,
                cost_preds_batch,
                cost_return_batch,
                rnn_states_cost_batch,
                cost_adv_targ,
                aver_episode_costs_batch,
            )

    def recurrent_generator(
        self, advantages: np.ndarray, num_mini_batch: int, data_chunk_length: int, cost_adv: Optional[np.ndarray] = None
    ) -> Generator[
        Tuple[
            np.ndarray,  # share_obs_batch
            np.ndarray,  # obs_batch
            np.ndarray,  # node_obs_batch
            np.ndarray,  # adj_batch
            np.ndarray,  # agent_id_batch
            np.ndarray,  # share_agent_id_batch
            np.ndarray,  # rnn_states_batch
            np.ndarray,  # rnn_states_critic_batch
            np.ndarray,  # actions_batch
            np.ndarray,  # value_preds_batch
            np.ndarray,  # return_batch
            np.ndarray,  # masks_batch
            np.ndarray,  # active_masks_batch
            np.ndarray,  # old_action_log_probs_batch
            np.ndarray,  # adv_targ
            np.ndarray,  # available_actions_batch
            Optional[np.ndarray],  # factor_batch
            np.ndarray,  # cost_preds_batch
            np.ndarray,  # cost_return_batch
            np.ndarray,  # rnn_states_cost_batch
            np.ndarray,  # cost_adv_targ
            np.ndarray,  # aver_episode_costs_batch
        ],
        None,
        None,
    ]:
        """
        Yield training data for chunked RNN training, including safety-related data.
        
        Args:
            advantages (np.ndarray): Advantage estimates.
            num_mini_batch (int): Number of minibatches to split the batch into.
            data_chunk_length (int): Length of sequence chunks with which to train RNN.
            cost_adv (np.ndarray, optional): Cost advantage estimates.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents
        data_chunks = batch_size // data_chunk_length  # [C=r*T*M/L]
        mini_batch_size = data_chunks // num_mini_batch

        rand = torch.randperm(data_chunks).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(num_mini_batch)
        ]

        if len(self.share_obs.shape) > 4:
            share_obs = (
                self.share_obs[:-1]
                .transpose(1, 2, 0, 3, 4, 5)
                .reshape(-1, *self.share_obs.shape[3:])
            )
            obs = (
                self.obs[:-1]
                .transpose(1, 2, 0, 3, 4, 5)
                .reshape(-1, *self.obs.shape[3:])
            )
        else:
            share_obs = _cast(self.share_obs[:-1])
            obs = _cast(self.obs[:-1])

        node_obs = (
            self.node_obs[:-1]
            .transpose(1, 2, 0, 3, 4)
            .reshape(-1, *self.node_obs.shape[3:])
        )
        adj = self.adj[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.adj.shape[3:])
        agent_id = _cast(self.agent_id[:-1])
        share_agent_id = _cast(self.share_agent_id[:-1])
        actions = _cast(self.actions)
        action_log_probs = _cast(self.action_log_probs)
        advantages = _cast(advantages)
        value_preds = _cast(self.value_preds[:-1])
        returns = _cast(self.returns[:-1])
        cost_preds = _cast(self.cost_preds[:-1])
        cost_returns = _cast(self.cost_returns[:-1])
        masks = _cast(self.masks[:-1])
        active_masks = _cast(self.active_masks[:-1])
        rnn_states = (
            self.rnn_states[:-1]
            .transpose(1, 2, 0, 3, 4)
            .reshape(-1, *self.rnn_states.shape[3:])
        )
        rnn_states_critic = (
            self.rnn_states_critic[:-1]
            .transpose(1, 2, 0, 3, 4)
            .reshape(-1, *self.rnn_states_critic.shape[3:])
        )
        rnn_states_cost = (
            self.rnn_states_cost[:-1]
            .transpose(1, 2, 0, 3, 4)
            .reshape(-1, *self.rnn_states_cost.shape[3:])
        )
        if self.factor is not None:
            factor = _cast(self.factor)
        if self.available_actions is not None:
            available_actions = _cast(self.available_actions[:-1])
        aver_episode_costs = _cast(self.aver_episode_costs[:-1])
        if cost_adv is not None:
            cost_adv = _cast(cost_adv)

        for indices in sampler:
            share_obs_batch = []
            obs_batch = []
            node_obs_batch = []
            adj_batch = []
            agent_id_batch = []
            share_agent_id_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            rnn_states_cost_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            cost_preds_batch = []
            cost_return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            cost_adv_targ = []
            factor_batch = []
            aver_episode_costs_batch = []

            for index in indices:
                ind = index * data_chunk_length
                share_obs_batch.append(share_obs[ind : ind + data_chunk_length])
                obs_batch.append(obs[ind : ind + data_chunk_length])
                node_obs_batch.append(node_obs[ind : ind + data_chunk_length])
                adj_batch.append(adj[ind : ind + data_chunk_length])
                agent_id_batch.append(agent_id[ind : ind + data_chunk_length])
                share_agent_id_batch.append(share_agent_id[ind : ind + data_chunk_length])
                actions_batch.append(actions[ind : ind + data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[ind : ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind : ind + data_chunk_length])
                return_batch.append(returns[ind : ind + data_chunk_length])
                cost_preds_batch.append(cost_preds[ind : ind + data_chunk_length])
                cost_return_batch.append(cost_returns[ind : ind + data_chunk_length])
                masks_batch.append(masks[ind : ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind : ind + data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind : ind + data_chunk_length])
                adv_targ.append(advantages[ind : ind + data_chunk_length])
                if cost_adv is not None:
                    cost_adv_targ.append(cost_adv[ind : ind + data_chunk_length])
                if self.factor is not None:
                    factor_batch.append(factor[ind : ind + data_chunk_length])
                aver_episode_costs_batch.append(aver_episode_costs[ind : ind + data_chunk_length])
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])
                rnn_states_cost_batch.append(rnn_states_cost[ind])

            L, N = data_chunk_length, mini_batch_size
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            obs_batch = np.stack(obs_batch, axis=1)
            node_obs_batch = np.stack(node_obs_batch, axis=1)
            adj_batch = np.stack(adj_batch, axis=1)
            agent_id_batch = np.stack(agent_id_batch, axis=1)
            share_agent_id_batch = np.stack(share_agent_id_batch, axis=1)
            actions_batch = np.stack(actions_batch, axis=1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            cost_preds_batch = np.stack(cost_preds_batch, axis=1)
            cost_return_batch = np.stack(cost_return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            adv_targ = np.stack(adv_targ, axis=1)
            if cost_adv is not None:
                cost_adv_targ = np.stack(cost_adv_targ, axis=1)
            else:
                cost_adv_targ = None
            if self.factor is not None:
                factor_batch = np.stack(factor_batch, axis=1)
            else:
                factor_batch = None
            aver_episode_costs_batch = np.stack(aver_episode_costs_batch, axis=1)

            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])
            rnn_states_cost_batch = np.stack(rnn_states_cost_batch).reshape(N, *self.rnn_states_cost.shape[3:])

            share_obs_batch = _flatten(L, N, share_obs_batch)
            obs_batch = _flatten(L, N, obs_batch)
            node_obs_batch = _flatten(L, N, node_obs_batch)
            adj_batch = _flatten(L, N, adj_batch)
            agent_id_batch = _flatten(L, N, agent_id_batch)
            share_agent_id_batch = _flatten(L, N, share_agent_id_batch)
            actions_batch = _flatten(L, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            cost_preds_batch = _flatten(L, N, cost_preds_batch)
            cost_return_batch = _flatten(L, N, cost_return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)
            if cost_adv_targ is not None:
                cost_adv_targ = _flatten(L, N, cost_adv_targ)
            if factor_batch is not None:
                factor_batch = _flatten(L, N, factor_batch)
            aver_episode_costs_batch = _flatten(L, N, aver_episode_costs_batch)

            yield (
                share_obs_batch,
                obs_batch,
                node_obs_batch,
                adj_batch,
                agent_id_batch,
                share_agent_id_batch,
                rnn_states_batch,
                rnn_states_critic_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                active_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                available_actions_batch,
                factor_batch,
                cost_preds_batch,
                cost_return_batch,
                rnn_states_cost_batch,
                cost_adv_targ,
                aver_episode_costs_batch,
            )

def create_generator():
    mylist = range(3)
    for i in mylist:
        yield i * i
