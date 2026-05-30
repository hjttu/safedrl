import time
import numpy as np
import argparse
from typing import Tuple
import torch
from torch import Tensor
import torch.nn as nn
from onpolicy.algorithms.graph_lagr_MAPPOPolicy import GS_MAPPOPolicy
from onpolicy.utils.gs_buffer import GSReplayBuffer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_grad_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check
import torch.jit as jit
import torch.cuda.amp as amp

class GS_MAPPO():
    """
        Trainer class for Graph Safe MAPPO to update policies.
        args: (argparse.Namespace)  
            Arguments containing relevant model, policy, and env information.
        policy: (GR_MAPPO_Policy) 
            Policy to update.
        device: (torch.device) 
            Specifies the device to run on (cpu/gpu).
    """
    def __init__(self, 
                args: argparse.Namespace, 
                policy: GS_MAPPOPolicy,
                device=torch.device("cpu")) -> None:
        """
        Initialize trainer for MAPPO with Lagrangian optimization and graph support.
        
        Args:
            args: (argparse.Namespace) Arguments containing model, policy, and env information.
            policy: (GS_MAPPOPolicy) Policy to update, supporting graph inputs and cost critic.
            device: (torch.device) Specifies the device to run on (cpu/gpu).
            attempt_feasible_recovery: (bool) Handle cases where x=0 is infeasible but problem is feasible.
            attempt_infeasible_recovery: (bool) Handle entirely infeasible optimization problems.
            revert_to_last_safe_point: (bool) Reset to last safe point if optimization is infeasible.
            delta_bound: (float) Trust region constraint bound.
            safety_bound: (float) Safety constraint bound.
            _backtrack_ratio: (float) Backtracking ratio for line search.
            _max_backtracks: (int) Maximum number of backtracking steps.
            _constraint_name_1: (str) Name of trust region constraint.
            _constraint_name_2: (str) Name of safety region constraint.
            linesearch_infeasible_recovery: (bool) Use line search for infeasible recovery.
            accept_violation: (bool) Accept constraint violations during optimization.
        """
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.cost_value_loss_coef = args.cost_value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta
        self.gamma = args.gamma

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self.scaler = amp.GradScaler() 
        assert (self._use_popart and self._use_valuenorm) == False, (
            "self._use_popart and self._use_valuenorm cannot be set True simultaneously"
        )

        # Lagrangian and safety parameters
        self.lagrangian_coef = args.lagrangian_coef_rate
        self.lamda_lagr = args.lamda_lagr
        self.safety_bound = args.safety_bound
        self.lamda_scale = args.lamda_scale

        # Value and cost normalizers
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
            # self.cost_normalizer = self.policy.cost_critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
            # self.cost_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None
            # self.cost_normalizer = None

    def cal_value_loss(self, 
                    values:Tensor, 
                    value_preds_batch:Tensor, 
                    return_batch:Tensor, 
                    active_masks_batch:Tensor) -> Tensor:
        """
            Calculate value function loss.
            values: (torch.Tensor) 
                value function predictions.
            value_preds_batch: (torch.Tensor) 
                "old" value  predictions from data batch (used for value clip loss)
            return_batch: (torch.Tensor) 
                reward to go returns.
            active_masks_batch: (torch.Tensor) 
                denotes if agent is active or dead at a given timesep.

            :return value_loss: (torch.Tensor) 
                value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - 
                            value_preds_batch).clamp(-self.clip_param,
                                                    self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - \
                            value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - \
                            values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss
    
    def cal_cost_v_loss(self, 
                        values:Tensor, 
                        value_preds_batch:Tensor, 
                        return_batch:Tensor, 
                        active_masks_batch:Tensor) -> Tensor:
            """
                Calculate value function loss.
                values: (torch.Tensor) 
                    value function predictions.
                value_preds_batch: (torch.Tensor) 
                    "old" value  predictions from data batch (used for value clip loss)
                return_batch: (torch.Tensor) 
                    reward to go returns.
                active_masks_batch: (torch.Tensor) 
                    denotes if agent is active or dead at a given timesep.

                :return value_loss: (torch.Tensor) 
                    value function loss.
            """
            value_pred_clipped = value_preds_batch + (values - 
                                value_preds_batch).clamp(-self.clip_param,
                                                        self.clip_param)

            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

            if self._use_huber_loss:
                value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
                value_loss_original = huber_loss(error_original, self.huber_delta)
            else:
                value_loss_clipped = mse_loss(error_clipped)
                value_loss_original = mse_loss(error_original)

            if self._use_clipped_value_loss:
                value_loss = torch.max(value_loss_original, value_loss_clipped)
            else:
                value_loss = value_loss_original

            if self._use_value_active_masks:
                value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
            else:
                value_loss = value_loss.mean()

            return value_loss
    
    @torch.cuda.amp.autocast()
    def ppo_update(self, 
                sample: Tuple, 
                update_actor: bool = True) -> Tuple[torch.Tensor, torch.Tensor, 
                                                    torch.Tensor, torch.Tensor, 
                                                    torch.Tensor, torch.Tensor, 
                                                    torch.Tensor, torch.Tensor]:
        """
        Update actor and critic networks with safety and Lagrangian optimization.
        
        Args:
            sample: (Tuple) Contains data batch with which to update networks.
            update_actor: (bool) Whether to update actor network.

        Returns:
            value_loss: (torch.Tensor) Value function loss.
            critic_grad_norm: (torch.Tensor) Gradient norm from critic update.
            policy_loss: (torch.Tensor) Actor (policy) loss value.
            dist_entropy: (torch.Tensor) Action entropies.
            actor_grad_norm: (torch.Tensor) Gradient norm from actor update.
            imp_weights: (torch.Tensor) Importance sampling weights.
            cost_loss: (torch.Tensor) Cost value function loss.
            cost_grad_norm: (torch.Tensor) Gradient norm from cost critic update.
        """
        share_obs_batch, obs_batch, node_obs_batch, adj_batch, agent_id_batch, \
        share_agent_id_batch, rnn_states_batch, rnn_states_critic_batch, \
        actions_batch, value_preds_batch, return_batch, masks_batch, \
        active_masks_batch, old_action_log_probs_batch, adv_targ, \
        available_actions_batch, factor_batch, cost_preds_batch, \
        cost_returns_batch, rnn_states_cost_batch, cost_adv_targ, \
        aver_episode_costs = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        factor_batch = check(factor_batch).to(**self.tpdv)
        cost_preds_batch = check(cost_preds_batch).to(**self.tpdv)
        cost_returns_batch = check(cost_returns_batch).to(**self.tpdv)
        cost_adv_targ = check(cost_adv_targ).to(**self.tpdv)
        aver_episode_costs = check(aver_episode_costs).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy, cost_values = self.policy.evaluate_actions(
            share_obs_batch,
            obs_batch,
            node_obs_batch,
            adj_batch,
            agent_id_batch,
            share_agent_id_batch,
            rnn_states_batch, 
            rnn_states_critic_batch, 
            actions_batch, 
            masks_batch, 
            available_actions_batch,
            active_masks_batch,
            rnn_states_cost_batch
        )

        # Actor update with Lagrangian hybrid advantage
        adv_targ_hybrid = adv_targ - self.lamda_lagr * cost_adv_targ* self.lamda_scale
        # adv_targ_hybrid = adv_targ
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = imp_weights * adv_targ_hybrid
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ_hybrid

        if self._use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()
        if update_actor:
            self.scaler.scale((policy_loss - dist_entropy * self.entropy_coef)).backward()
        
        self.scaler.unscale_(self.policy.actor_optimizer)
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_grad_norm(self.policy.actor.parameters())

        self.scaler.step(self.policy.actor_optimizer)

        # Reward critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)
        self.policy.critic_optimizer.zero_grad()
        self.scaler.scale(value_loss * self.value_loss_coef).backward()
        self.scaler.unscale_(self.policy.critic_optimizer)
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_grad_norm(self.policy.critic.parameters())
        self.scaler.step(self.policy.critic_optimizer)
        
        # print("value is: ", values)
        # print("value_preds_batch is: ", value_preds_batch)
        # print("return_batch is: ", return_batch)

        # print("*****************************************")
        # print("cost_values is: ", cost_values)
        # print("cost_preds_batch is: ", cost_preds_batch)
        # print("cost_returns_batch is: ", cost_returns_batch)
        # Cost critic update
        cost_loss = self.cal_value_loss(cost_values, cost_preds_batch, cost_returns_batch, active_masks_batch)
        self.policy.cost_optimizer.zero_grad()
        self.scaler.scale(cost_loss * self.cost_value_loss_coef).backward()
        self.scaler.unscale_(self.policy.cost_optimizer)
        if self._use_max_grad_norm:
            cost_grad_norm = nn.utils.clip_grad_norm_(self.policy.cost_critic.parameters(), self.max_grad_norm)
        else:
            cost_grad_norm = get_grad_norm(self.policy.cost_critic.parameters())
        self.scaler.step(self.policy.cost_optimizer)

        # Update Lagrangian coefficient
        if aver_episode_costs is not None:
            # delta_lamda_lagr = -(( cost_returns_batch.mean() - self.safety_bound) * (1 - self.gamma) + (imp_weights * cost_adv_targ)).mean().detach()
            # delta_lamda_lagr = -(( aver_episode_costs.mean() - self.safety_bound) + (imp_weights * cost_adv_targ)).mean().detach()
            delta_lamda_lagr = -((aver_episode_costs.mean() - self.safety_bound)* (1 - self.gamma)).mean().detach()
            R_ReLU = torch.nn.ReLU()
            self.lamda_lagr = R_ReLU(self.lamda_lagr - (delta_lamda_lagr * self.lagrangian_coef))
            self.lamda_lagr = torch.clamp(self.lamda_lagr, 0.0, 1.0)

            # print("the average episode costs is: {}, the cost_returns_batch is: {}".format(aver_episode_costs.mean(), cost_returns_batch.mean()))
            # print("the value_adv_targ is: {}, the cost_adv_targ is: {}".format(adv_targ.mean(), cost_adv_targ.mean()))

        self.scaler.update()

        return (value_loss, critic_grad_norm, policy_loss, dist_entropy, 
                actor_grad_norm, imp_weights, cost_loss, cost_grad_norm, aver_episode_costs, cost_adv_targ, adv_targ)

    def train(self, 
            buffer: GSReplayBuffer, 
            update_actor: bool = True):
        """
        Perform a training update using minibatch GD with safety and Lagrangian optimization.
        
        Args:
            buffer: (GSReplayBuffer) Buffer containing training data, including graph and cost data.
            update_actor: (bool) Whether to update actor network.

        Returns:
            train_info: (dict) Contains information regarding training update (e.g., loss, grad norms, etc).
        """
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
            cost_adv = buffer.cost_returns[:-1] - self.value_normalizer.denormalize(buffer.cost_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
            cost_adv = buffer.cost_returns[:-1] - buffer.cost_preds[:-1]  # lower safer. negative is better

        # cost_adv = buffer.cost_returns[:-1] - buffer.cost_preds[:-1]

        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        cost_adv_copy = cost_adv.copy()
        cost_adv_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_cost_adv = np.nanmean(cost_adv_copy)
        std_cost_adv = np.nanstd(cost_adv_copy)
        cost_adv = (cost_adv - mean_cost_adv) / (std_cost_adv + 1e-5)

        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['cost_loss'] = 0
        train_info['cost_grad_norm'] = 0
        train_info['lamda_lagr'] = 0
        # train_info['avg_epi_costs'] = 0
        train_info['cost_adv_targ'] = 0
        train_info['adv_targ'] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length, cost_adv=cost_adv)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch, cost_adv=cost_adv)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch, mini_batch_size=self.data_chunks//self.num_mini_batch, cost_adv=cost_adv)

            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, cost_loss, cost_grad_norm, aver_episode_costs, cost_adv_targ, adv_targ = self.ppo_update(sample, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()
                train_info['cost_loss'] += cost_loss.item()
                train_info['cost_grad_norm'] += cost_grad_norm
                train_info['lamda_lagr'] += self.lamda_lagr.item()
                # train_info['avg_epi_costs'] += aver_episode_costs.mean()
                train_info['cost_adv_targ'] += cost_adv_targ.mean()
                train_info['adv_targ'] += adv_targ.mean()


        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        """Convert networks to training mode"""
        self.policy.actor.train()
        self.policy.critic.train()
        self.policy.cost_critic.train()

    def prep_rollout(self):
        """Convert networks to eval mode"""
        self.policy.actor.eval()
        self.policy.critic.eval()
        self.policy.cost_critic.eval()