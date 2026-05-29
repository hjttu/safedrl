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


def _t2n(x):
    return x.detach().cpu().numpy()

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
        if self.use_train_render:
            print("render the image while training")

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
                # print("step:", step)
                # Sample actions
                (   values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
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

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions_env,
            cost_preds,
            rnn_states_cost,
        )

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

            eval_rnn_states[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones(
                (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
            )
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), 1), dtype=np.float32
            )

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
