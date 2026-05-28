import time
import numpy as np
from numpy import ndarray as arr
from typing import Tuple
import torch
from onpolicy.runner.shared.base_lagr_runner import Runner
import wandb
import imageio
from onpolicy import global_var as glv
import csv
from onpolicy.algorithms.cbf.features import (
    accel_to_multidiscrete_action,
    discrete_actions_to_accel,
    risk_from_cbf,
)
from onpolicy.algorithms.cbf.hocbf_filter import HOCBFSafetyFilter, HOCBFParams
from onpolicy.algorithms.cbf.discrete_action_mask import CBFActionMaskConfig, CBFDiscreteActionMask
from onpolicy.algorithms.cbf.temporal_responsibility_memory import TemporalResponsibilityMemory


def _t2n(x):
    return x.detach().cpu().numpy()

def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

class GSMPERunner(Runner):
    """
    Runner class to perform training, evaluation and data
    collection for the MPEs. See parent class for details
    """

    dt = 0.1

    def __init__(self, config):
        super(GSMPERunner, self).__init__(config)
        self.save_data = self.all_args.save_data
        self.use_train_render = self.all_args.use_train_render
        self.no_imageshow = self.all_args.no_imageshow
        self.reward_file_name = self.all_args.reward_file_name
        self.cost_file_name = self.all_args.cost_file_name
        self.use_cbf_filter = getattr(self.all_args, "use_cbf_filter", False)
        self.use_cbf_action_mask = getattr(self.all_args, "use_cbf_action_mask", False)
        self.use_continuous_cbf_filter = getattr(self.all_args, "use_continuous_cbf_filter", False)
        self.trm = TemporalResponsibilityMemory(
            beta_min=getattr(self.all_args, "beta_min", 0.2),
            beta_max=getattr(self.all_args, "beta_max", 0.95),
            reset_steps=getattr(self.all_args, "memory_reset_steps", 20),
            reset_h=getattr(self.all_args, "memory_reset_h", 1.0),
        )
        self.cbf_filter = HOCBFSafetyFilter(
            HOCBFParams(
                k1=getattr(self.all_args, "cbf_k1", 2.0),
                k2=getattr(self.all_args, "cbf_k2", 2.0),
                slack_penalty_base=getattr(self.all_args, "slack_penalty_base", 100.0),
                slack_penalty_priority=getattr(self.all_args, "slack_penalty_priority", 500.0),
                eval_hard_filter=getattr(self.all_args, "eval_hard_filter", True),
            )
        )
        self.action_n_x = 20
        self.action_n_y = 20
        if self.envs.action_space[0].__class__.__name__ == "MultiDiscrete":
            action_dims = self.envs.action_space[0].high - self.envs.action_space[0].low + 1
            self.action_n_x = int(action_dims[0])
            self.action_n_y = int(action_dims[1])
        self.action_masker = CBFDiscreteActionMask(
            CBFActionMaskConfig(
                n_x=self.action_n_x,
                n_y=self.action_n_y,
                k1=getattr(self.all_args, "cbf_k1", 2.0),
                k2=getattr(self.all_args, "cbf_k2", 2.0),
                h_keep=getattr(self.all_args, "h_keep", 0.05),
                tau_ttc=getattr(self.all_args, "tau_ttc", 1.0),
                d_safe_agent=getattr(self.all_args, "d_safe_agent", 0.25),
                d_safe_obstacle=getattr(self.all_args, "d_safe_obstacle", 0.30),
                lambda_soft_mask=getattr(self.all_args, "lambda_soft_mask", 1.0),
                action_mask_hard=getattr(self.all_args, "action_mask_hard", True),
                action_mask_soft_penalty=getattr(self.all_args, "action_mask_soft_penalty", True),
                neighbor_action_mode=getattr(self.all_args, "neighbor_action_mode", "zero"),
                empty_mask_fallback=getattr(self.all_args, "empty_mask_fallback", "min_violation"),
                guide_tau=getattr(self.all_args, "guide_tau", 0.5),
                semi_hard_mask=getattr(self.all_args, "semi_hard_mask", True),
                guide_fallback_topk=getattr(self.all_args, "guide_fallback_topk", 5),
            )
        )
        self.last_actions_norm = np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype=np.float32)
        self.current_total_num_steps = 0
        self.last_pg_cbf_stats = {
            "guide_alpha": 0.0,
            "cbf_beta": 0.0,
            "h_keep_current": getattr(self.all_args, "h_keep_init", -0.05),
            "hard_mask_enabled": 0.0,
            "min_valid_action_ratio": getattr(self.all_args, "min_valid_action_ratio_init", 0.7),
        }
        if self.use_train_render:
            print("render the image while training")

    def _pg_cbf_schedule(self, total_num_steps: int):
        progress = min(1.0, float(total_num_steps) / max(float(self.num_env_steps), 1.0))
        guide_decay_ratio = max(float(getattr(self.all_args, "guide_decay_ratio", 0.6)), 1e-6)
        warmup_ratio = float(getattr(self.all_args, "cbf_warmup_ratio", 0.2))
        warmup_denom = max(1.0 - warmup_ratio, 1e-6)
        alpha_t = float(getattr(self.all_args, "guide_alpha_init", 1.0)) * max(0.0, 1.0 - progress / guide_decay_ratio)
        beta_t = float(getattr(self.all_args, "cbf_beta_init", 0.02)) + (
            float(getattr(self.all_args, "cbf_beta_final", 0.5)) - float(getattr(self.all_args, "cbf_beta_init", 0.02))
        ) * _smoothstep(max(0.0, (progress - warmup_ratio) / warmup_denom))
        h_keep_t = float(getattr(self.all_args, "h_keep_init", -0.05)) + (
            float(getattr(self.all_args, "h_keep_final", 0.05)) - float(getattr(self.all_args, "h_keep_init", -0.05))
        ) * _smoothstep(progress)
        hard_enabled = bool(getattr(self.all_args, "action_mask_hard", True)) and progress >= float(getattr(self.all_args, "hard_mask_start_ratio", 0.7))
        min_valid_ratio = float(getattr(self.all_args, "min_valid_action_ratio_init", 0.7)) + (
            float(getattr(self.all_args, "min_valid_action_ratio_final", 0.3)) - float(getattr(self.all_args, "min_valid_action_ratio_init", 0.7))
        ) * _smoothstep(progress)
        return progress, alpha_t, beta_t, h_keep_t, hard_enabled, min_valid_ratio

    def run(self):
        if self.save_data:
            #csv
            print('save training data')
            file = open(self.reward_file_name+'.csv', 'w', encoding='utf-8', newline="")
            writer = csv.writer(file)
            writer.writerow(['step', 'average', 'min', 'max', 'std'])
            file.close()

            file1 = open(self.cost_file_name+'.csv', 'w', encoding='utf-8', newline="")
            writer1 = csv.writer(file1)
            writer1.writerow(['step', 'average', 'min', 'max', 'std'])
            file1.close()

        self.warmup()

        start = time.time()
        episodes = (
            int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        )

        # This is where the episodes are actually run.
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            CL_ratio = episode/episodes
            glv.set_value('CL_ratio', CL_ratio)  #curriculum learning
            self.envs.set_CL(glv.get_value('CL_ratio'))  # env_wrapper
            for step in range(self.episode_length):
                self.current_total_num_steps = (
                    episode * self.episode_length * self.n_rollout_threads
                    + step * self.n_rollout_threads
                )
                # print("step:", step)
                # Sample actions
                (   values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                    actions_rl_cont,
                    actions_safe_cont,
                    cbf_diag,
                    action_mask_diag,
                    action_mask,
                    joint_actions,
                    cost_preds,
                    rnn_states_cost,
                ) = self.collect(step)

                # print("rnn_states:", rnn_states)

                # Obs reward and next obs
                obs, agent_id, node_obs, adj, rewards, costs, dones, infos = self.envs.step(actions_env)
                # print("reward is {} costs is {}:".format(rewards, costs))  # DEBUG
                data = (
                    obs,
                    agent_id,
                    node_obs,
                    adj,
                    rewards,
                    costs,
                    dones,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_rl_cont,
                    actions_safe_cont,
                    cbf_diag,
                    action_mask_diag,
                    action_mask,
                    joint_actions,
                    cost_preds,
                    rnn_states_cost
                )

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (
                (episode + 1) * self.episode_length * self.n_rollout_threads
            )

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()

                env_infos = self.process_infos(infos)

                avg_ep_rew = np.mean(self.buffer.rewards) * self.episode_length
                train_infos["average_episode_rewards"] = avg_ep_rew
                avg_ep_cost = np.mean(self.buffer.costs) * self.episode_length
                train_infos["average_episode_costs"] = avg_ep_cost
                if self.use_cbf_filter:
                    diag = self.buffer.cbf_diag
                    train_infos["filter_intervention_rate"] = float(np.mean(diag[..., 0] > 1e-5))
                    train_infos["average_intervention_norm"] = float(np.mean(diag[..., 0]))
                    train_infos["QP_infeasible_rate"] = float(np.mean(diag[..., 3]))
                    train_infos["active_constraints"] = float(np.mean(diag[..., 2]))
                    train_infos["cbf_slack"] = float(np.mean(diag[..., 1]))
                    train_infos["responsibility_switch_frequency"] = float(np.mean(diag[..., 5]))
                    du = np.diff(self.buffer.actions_safe_cont, axis=0)
                    train_infos["oscillation_index"] = float(np.mean(np.sum(du * du, axis=-1))) if du.size else 0.0
                    dist_adj = self.buffer.adj[..., 0] if self.buffer.adj.ndim == 6 else self.buffer.adj
                    masked = dist_adj[(dist_adj > 0) & (dist_adj < self.all_args.max_edge_dist)]
                    train_infos["min_inter_agent_distance"] = float(np.min(masked)) if masked.size else 0.0
                if self.use_cbf_action_mask:
                    mdiag = self.buffer.action_mask_diag
                    train_infos["guide_alpha"] = self.last_pg_cbf_stats["guide_alpha"]
                    train_infos["cbf_beta"] = self.last_pg_cbf_stats["cbf_beta"]
                    train_infos["h_keep_current"] = self.last_pg_cbf_stats["h_keep_current"]
                    train_infos["hard_mask_enabled"] = self.last_pg_cbf_stats["hard_mask_enabled"]
                    train_infos["safe_action_ratio"] = float(np.mean(mdiag[..., 0]))
                    train_infos["valid_action_ratio"] = float(np.mean(mdiag[..., 0]))
                    train_infos["empty_mask_rate"] = float(np.mean(mdiag[..., 1]))
                    train_infos["average_num_safe_actions"] = float(np.mean(mdiag[..., 2]))
                    train_infos["min_num_safe_actions"] = float(np.min(mdiag[..., 3]))
                    train_infos["action_mask_entropy"] = float(np.mean(mdiag[..., 4]))
                    train_infos["hard_violation_min"] = float(np.min(mdiag[..., 5]))
                    train_infos["soft_risk_penalty_mean"] = float(np.mean(mdiag[..., 6]))
                    train_infos["risk_penalty_mean"] = float(np.mean(mdiag[..., 6]))
                    train_infos["fallback_action_rate"] = float(np.mean(mdiag[..., 7]))
                    train_infos["guide_fallback_rate"] = float(np.mean(mdiag[..., 7]))
                    if self.buffer.available_actions is not None and self.buffer.available_actions.shape[-1] == 3 * self.action_n_x * self.action_n_y:
                        _, risk_part, guide_part = np.split(self.buffer.available_actions[:-1], 3, axis=-1)
                        train_infos["guide_logits_mean"] = float(np.mean(guide_part))
                        train_infos["risk_penalty_weighted_mean"] = float(np.mean(risk_part))
                    else:
                        train_infos["guide_logits_mean"] = 0.0
      
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}, CL {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start)),
                                format(glv.get_value('CL_ratio'), '.3f')))
                print("average episode rewards is {}".format(avg_ep_rew))
                print("average episode costs is {}".format(avg_ep_cost))

                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

                r = self.buffer.rewards.mean(2).sum(axis=(0, 2))
                c = self.buffer.costs.mean(2).sum(axis=(0, 2))
                Average_r, Min_r, Max_r, Std_r = np.mean(r), np.min(r), np.max(r), np.std(r)
                Average_c, Min_c, Max_c, Std_c = np.mean(c), np.min(c), np.max(c), np.std(c)

                if self.save_data:
                    file = open(self.reward_file_name+'.csv', 'a', encoding='utf-8', newline="")
                    writer = csv.writer(file)
                    writer.writerow([total_num_steps, Average_r, Min_r, Max_r, Std_r])
                    file.close()

                    file1 = open(self.cost_file_name+'.csv', 'a', encoding='utf-8', newline="")
                    writer1 = csv.writer(file1)
                    writer1.writerow([total_num_steps, Average_c, Min_c, Max_c, Std_c])
                    file1.close()

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, agent_id, node_obs, adj = self.envs.reset()

        # replay buffer
        if self.use_centralized_V:
            # (n_rollout_threads, n_agents, feats) -> (n_rollout_threads, n_agents*feats)
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            # (n_rollout_threads, n_agents*feats) -> (n_rollout_threads, n_agents, n_agents*feats)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
            # (n_rollout_threads, n_agents, 1) -> (n_rollout_threads, n_agents*1)
            share_agent_id = agent_id.reshape(self.n_rollout_threads, -1)
            # (n_rollout_threads, n_agents*1) -> (n_rollout_threads, n_agents, n_agents*1)
            share_agent_id = np.expand_dims(share_agent_id, 1).repeat(
                self.num_agents, axis=1
            )
        else:
            share_obs = obs
            share_agent_id = agent_id

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.node_obs[0] = node_obs.copy()
        self.buffer.adj[0] = adj.copy()
        self.buffer.agent_id[0] = agent_id.copy()
        self.buffer.share_agent_id[0] = share_agent_id.copy()

    @torch.no_grad()
    def collect(self, step: int) -> Tuple[arr, arr, arr, arr, arr, arr, arr, arr]:
        self.trainer.prep_rollout()
        action_mask = None
        action_mask_diag = np.zeros((self.n_rollout_threads, self.num_agents, 8), dtype=np.float32)
        if self.use_cbf_action_mask:
            _, alpha_t, beta_t, h_keep_t, hard_enabled, min_valid_ratio = self._pg_cbf_schedule(
                self.current_total_num_steps
            )
            guide_actions_norm = self._get_guide_actions_norm() if getattr(self.all_args, "use_cbf_guide", False) else None
            mask, risk_penalty, guide_logits, action_mask_diag = self.action_masker.build_batch(
                self.buffer.node_obs[step],
                self.buffer.adj[step],
                agent_max_accel=0.5,
                last_actions_norm=self.last_actions_norm,
                guide_actions_norm=guide_actions_norm,
                hard_enabled=hard_enabled,
                h_keep=h_keep_t,
                semi_hard_mask=getattr(self.all_args, "semi_hard_mask", True),
                min_valid_action_ratio=min_valid_ratio,
                guide_fallback_topk=getattr(self.all_args, "guide_fallback_topk", 5),
            )
            if getattr(self.all_args, "action_mask_soft_penalty", True):
                action_mask = np.concatenate([mask, beta_t * risk_penalty, alpha_t * guide_logits], axis=-1)
            else:
                action_mask = np.concatenate([mask, np.zeros_like(risk_penalty), alpha_t * guide_logits], axis=-1)
            self.last_pg_cbf_stats = {
                "guide_alpha": float(alpha_t),
                "cbf_beta": float(beta_t),
                "h_keep_current": float(h_keep_t),
                "hard_mask_enabled": float(hard_enabled),
                "min_valid_action_ratio": float(min_valid_ratio),
            }

        (
            value,
            action,
            action_log_prob,
            rnn_states,
            rnn_states_critic,
            cost_preds,
            rnn_states_cost,
        ) = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.node_obs[step]),
            np.concatenate(self.buffer.adj[step]),
            np.concatenate(self.buffer.agent_id[step]),
            np.concatenate(self.buffer.share_agent_id[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
            np.concatenate(self.buffer.rnn_states_cost[step]),
            available_actions=None if action_mask is None else np.concatenate(action_mask),
        )
        # print("cost_preds:", cost_preds)  # DEBUG
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        cost_preds = np.array(np.split(_t2n(cost_preds), self.n_rollout_threads))
        rnn_states_cost = np.array(np.split(_t2n(rnn_states_cost), self.n_rollout_threads))
        
        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == "MultiDiscrete":
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[
                    actions[:, :, i]
                ]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == "Discrete":
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            raise NotImplementedError

        action_bins = (self.action_n_x, self.action_n_y)
        actions_rl_cont = discrete_actions_to_accel(actions, n_bins=action_bins)
        actions_safe_cont = actions_rl_cont.copy()
        self.last_actions_norm = actions_rl_cont.copy()
        joint_actions = (actions[:, :, 0:1] * self.action_n_y + actions[:, :, 1:2]).astype(np.int32)
        cbf_diag = np.zeros((self.n_rollout_threads, self.num_agents, 6), dtype=np.float32)
        if self.use_cbf_filter and self.use_continuous_cbf_filter:
            priorities_t, gammas_t = self.trainer.policy.safety_edge_scores(
                np.concatenate(self.buffer.adj[step])
            )
            priorities = np.array(np.split(_t2n(priorities_t), self.n_rollout_threads))
            gammas_hat = np.array(np.split(_t2n(gammas_t), self.n_rollout_threads))
            for env_id in range(self.n_rollout_threads):
                node_obs_env = self.buffer.node_obs[step, env_id, 0]
                adj_env = self.buffer.adj[step, env_id, 0]
                priority_env = priorities[env_id, 0]
                gamma_env = gammas_hat[env_id, 0].copy()
                visible_keys = []
                risks = []
                gamma_delta = 0.0
                gamma_count = 0
                if adj_env.ndim == 3 and adj_env.shape[-1] > 9:
                    for i in range(self.num_agents):
                        for j in range(adj_env.shape[0]):
                            if i == j or adj_env[i, j, 0] <= 0 or adj_env[i, j, 0] >= self.all_args.max_edge_dist:
                                continue
                            entity_type = int(round(adj_env[i, j, 9]))
                            if entity_type not in (0, 2, 3):
                                continue
                            h_ij, hdot_ij, ttc_ij, d_safe = adj_env[i, j, 6], adj_env[i, j, 7], adj_env[i, j, 8], adj_env[i, j, 10]
                            risk = risk_from_cbf(h_ij, hdot_ij, ttc_ij, d_safe)
                            risks.append(risk)
                            if getattr(self.all_args, "use_trm", False) and entity_type == 0 and j < self.num_agents:
                                key = self.trm.agent_pair_key(env_id, i, j)
                                prev = self.trm.memory.get(key, None)
                                old_gamma = 0.5 if prev is None else prev.value
                                gamma_ij, gamma_ji = self.trm.update_agent_pair(env_id, i, j, gamma_env[i, j], risk, h_ij)
                                gamma_env[i, j] = gamma_ij
                                gamma_env[j, i] = gamma_ji
                                visible_keys.append(key)
                                gamma_delta += abs(gamma_ij - old_gamma)
                                gamma_count += 1
                            elif entity_type != 0:
                                gamma_env[i, j] = 1.0
                    if getattr(self.all_args, "use_trm", False):
                        self.trm.decay_unseen(visible_keys)
                filtered, info = self.cbf_filter.filter_env(
                    env_id,
                    node_obs_env,
                    actions_rl_cont[env_id],
                    priority_env,
                    gamma_env,
                    train=True,
                )
                actions_safe_cont[env_id] = filtered
                cbf_diag[env_id, :, 0] = np.linalg.norm(filtered - actions_rl_cont[env_id], axis=-1)
                cbf_diag[env_id, :, 1] = info["slack"]
                cbf_diag[env_id, :, 2] = info["active_constraints"]
                cbf_diag[env_id, :, 3] = info["qp_infeasible"]
                cbf_diag[env_id, :, 4] = float(np.mean(risks) if risks else 0.0)
                cbf_diag[env_id, :, 5] = gamma_delta / max(gamma_count, 1)
            actions_env = accel_to_multidiscrete_action(actions_safe_cont, n_bins=action_bins)

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions_env,
            actions_rl_cont,
            actions_safe_cont,
            cbf_diag,
            action_mask_diag,
            action_mask,
            joint_actions,
            cost_preds,
            rnn_states_cost,
        )

    def _get_guide_actions_norm(self):
        if not hasattr(self.envs, "get_guide_actions"):
            return np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype=np.float32)
        guide = np.asarray(self.envs.get_guide_actions(), dtype=np.float32)
        if guide.ndim == 4 and guide.shape[-1] == 1:
            guide = guide[..., 0]
        if guide.shape != (self.n_rollout_threads, self.num_agents, 2):
            guide = guide.reshape(self.n_rollout_threads, self.num_agents, 2)
        return np.clip(guide, -1.0, 1.0).astype(np.float32)

    def insert(self, data):
        (
            obs,
            agent_id,
            node_obs,
            adj,
            rewards,
            costs,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions_rl_cont,
            actions_safe_cont,
            cbf_diag,
            action_mask_diag,
            action_mask,
            joint_actions,
            cost_preds,
            rnn_states_cost,
        ) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32,
        )
        rnn_states_cost[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_cost.shape[3:]),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        # if centralized critic, then shared_obs is concatenation of obs from all agents
        if self.use_centralized_V:
            # TODO stack agent_id as well for agent specific information
            # (n_rollout_threads, n_agents, feats) -> (n_rollout_threads, n_agents*feats)
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            # (n_rollout_threads, n_agents*feats) -> (n_rollout_threads, n_agents, n_agents*feats)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
            # (n_rollout_threads, n_agents, 1) -> (n_rollout_threads, n_agents*1)
            share_agent_id = agent_id.reshape(self.n_rollout_threads, -1)
            # (n_rollout_threads, n_agents*1) -> (n_rollout_threads, n_agents, n_agents*1)
            share_agent_id = np.expand_dims(share_agent_id, 1).repeat(
                self.num_agents, axis=1
            )
        else:
            share_obs = obs
            share_agent_id = agent_id

        self.buffer.insert(
            share_obs,
            obs,
            node_obs,
            adj,
            agent_id,
            share_agent_id,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
            costs=costs,
            cost_preds=cost_preds,
            rnn_states_cost=rnn_states_cost,
            actions_rl_cont=actions_rl_cont,
            actions_safe_cont=actions_safe_cont,
            cbf_diag=cbf_diag,
            action_mask_diag=action_mask_diag,
            available_actions=action_mask,
            joint_actions=joint_actions,
        )

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data. added cost returns"""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(
            np.concatenate(self.buffer.share_obs[-1]),
            np.concatenate(self.buffer.node_obs[-1]),
            np.concatenate(self.buffer.adj[-1]),
            np.concatenate(self.buffer.share_agent_id[-1]),
            np.concatenate(self.buffer.rnn_states_critic[-1]),
            np.concatenate(self.buffer.masks[-1]),
        )
        # print("next_values:", next_values.shape) # [5,1]
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))

        next_costs = self.trainer.policy.get_cost_values(
            np.concatenate(self.buffer.share_obs[-1]),
            np.concatenate(self.buffer.node_obs[-1]),
            np.concatenate(self.buffer.adj[-1]),
            np.concatenate(self.buffer.share_agent_id[-1]),
            np.concatenate(self.buffer.rnn_states_cost[-1]),
            np.concatenate(self.buffer.masks[-1]),
        )
        next_costs = np.array(np.split(_t2n(next_costs), self.n_rollout_threads))
        # print("next_costs:", next_costs)  # DEBUG

        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
        self.buffer.compute_cost_returns(next_costs, self.trainer.value_normalizer)
        self.buffer.compute_average_episode_costs()

    @torch.no_grad()
    def eval(self, total_num_steps: int):
        """Evaluate policies with data from eval environments, including costs."""
        eval_episode = 0
        eval_episode_rewards = []
        eval_episode_costs = []
        one_episode_rewards = []
        one_episode_costs = []

        for eval_i in range(self.n_eval_rollout_threads):
            one_episode_rewards.append([])
            eval_episode_rewards.append([])
            one_episode_costs.append([])
            eval_episode_costs.append([])

        eval_obs, eval_agent_id, eval_node_obs, eval_adj = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
        )

        while True:
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_node_obs),
                np.concatenate(eval_adj),
                np.concatenate(eval_agent_id),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True,
            )
            eval_actions = np.array(
                np.split(_t2n(eval_action), self.n_eval_rollout_threads)
            )
            eval_rnn_states = np.array(
                np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads)
            )

            if self.eval_envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(
                        self.eval_envs.action_space[0].high[i] + 1
                    )[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate(
                            (eval_actions_env, eval_uc_actions_env), axis=2
                        )
            elif self.eval_envs.action_space[0].__class__.__name__ == "Discrete":
                eval_actions_env = np.squeeze(
                    np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2
                )
            else:
                raise NotImplementedError


            # Observe reward, cost, and next obs
            (
                eval_obs,
                eval_agent_id,
                eval_node_obs,
                eval_adj,
                eval_rewards,
                eval_costs,
                eval_dones,
                eval_infos,
            ) = self.eval_envs.step(eval_actions_env)

            for eval_i in range(self.n_eval_rollout_threads):
                one_episode_rewards[eval_i].append(eval_rewards[eval_i])
                one_episode_costs[eval_i].append(eval_costs[eval_i])

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = 0.0
            eval_masks = np.ones(
                (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
            )
            eval_masks[eval_dones_env == True] = 0.0

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards[eval_i].append(np.sum(one_episode_rewards[eval_i], axis=0))
                    eval_episode_costs[eval_i].append(np.sum(one_episode_costs[eval_i], axis=0))
                    one_episode_rewards[eval_i] = []
                    one_episode_costs[eval_i] = []

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.concatenate(eval_episode_rewards)
                eval_episode_costs = np.concatenate(eval_episode_costs)
                eval_env_infos = {
                    'eval_average_episode_rewards': eval_episode_rewards,
                    'eval_max_episode_rewards': [np.max(eval_episode_rewards)],
                    'eval_average_episode_costs': eval_episode_costs,
                    'eval_max_episode_costs': [np.max(eval_episode_costs)]
                }
                self.log_env(eval_env_infos, total_num_steps)
                print("eval average episode rewards of agent: " + str(np.mean(eval_episode_rewards)))
                print("eval average episode costs of agent: " + str(np.mean(eval_episode_costs)))
                break

    @torch.no_grad()
    def render(self, get_metrics: bool = False):
        """
        Visualize the env.
        get_metrics: bool (default=False)
            if True, just return the metrics of the env and don't render.
        """
        envs = self.envs

        all_frames = []
        rewards_arr, costs_arr, success_rates_arr, num_collisions_arr, frac_episode_arr = (
            [],
            [],
            [],
            [],
            [],
        )

        for episode in range(self.all_args.render_episodes):
            obs, agent_id, node_obs, adj = envs.reset()
            if not get_metrics:
                if self.all_args.save_gifs:
                    image = envs.render("rgb_array")[0][0]
                    all_frames.append(image)
                else:
                    envs.render("human")

            rnn_states = np.zeros(
                (
                    self.n_rollout_threads,
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            masks = np.ones(
                (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
            )

            episode_rewards = []
            episode_costs = []

            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(
                    np.concatenate(obs),
                    np.concatenate(node_obs),
                    np.concatenate(adj),
                    np.concatenate(agent_id),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    deterministic=True,
                )
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(
                    np.split(_t2n(rnn_states), self.n_rollout_threads)
                )

                if envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i] + 1)[
                            actions[:, :, i]
                        ]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate(
                                (actions_env, uc_actions_env), axis=2
                            )
                elif envs.action_space[0].__class__.__name__ == "Discrete":
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, agent_id, node_obs, adj, rewards, costs, dones, infos = envs.step(
                    actions_env
                )
                episode_rewards.append(rewards)
                episode_costs.append(costs)

                # print("rewards is {} costs is {}".format(rewards, costs))  # DEBUG

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones(
                    (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
                )
                masks[dones == True] = np.zeros(
                    ((dones == True).sum(), 1), dtype=np.float32
                )

                if not get_metrics:
                    if self.all_args.save_gifs:
                        image = envs.render("rgb_array")[0][0]
                        all_frames.append(image)
                        calc_end = time.time()
                        elapsed = calc_end - calc_start
                        if elapsed < self.all_args.ifi:
                            time.sleep(self.all_args.ifi - elapsed)
                    else:
                        envs.render("human")

            env_infos = self.process_infos(infos)
            # print('_'*50)
            num_collisions = self.get_collisions(env_infos)
            frac, success = self.get_fraction_episodes(env_infos)
            rewards_arr.append(np.mean(np.sum(np.array(episode_rewards), axis=0)))
            frac_episode_arr.append(np.mean(frac))
            success_rates_arr.append(success)
            num_collisions_arr.append(num_collisions)
            costs_arr.append(np.mean(np.sum(np.array(episode_costs), axis=0)))

            # print(np.mean(frac), success)
            print("Average episode rewards is: {}, costs is: {}".format(rewards_arr[-1], costs_arr[-1]))

        # print(rewards_arr)
        # print(frac_episode_arr)
        # print(success_rates_arr)
        # print(num_collisions_arr)

        if not get_metrics:
            if self.all_args.save_gifs:
                print("saving gif to:", str(self.gif_dir) + "/render.gif")
                imageio.mimsave(
                    str(self.gif_dir) + "/render.gif",
                    all_frames,
                    duration=self.all_args.ifi,
                )
