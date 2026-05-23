from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .buffer import RolloutBuffer, collate_graphs, generalized_advantage_estimate
from .env import MultiUAVNavEnv
from .models import GraphActorCritic


@dataclass
class TrainConfig:
    total_updates: int = 200
    rollout_steps: int = 128
    ppo_epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    actor_lr: float = 3e-4
    lambda_lr: float = 1e-2
    cost_limit: float = 0.05
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class GMATransLagrTrainer:
    def __init__(self, env: MultiUAVNavEnv, model: GraphActorCritic, cfg: TrainConfig | None = None):
        self.env = env
        self.model = model
        self.cfg = cfg or TrainConfig()
        self.device = torch.device(self.cfg.device)
        self.model.to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.actor_lr)
        self.lagrange = torch.zeros(env.cfg.n_agents, device=self.device)

    def train(self):
        obs = self.env.reset()
        history = []
        for update in range(1, self.cfg.total_updates + 1):
            rollout, obs, stats = self.collect_rollout(obs)
            loss_info = self.update(rollout)
            row = {"update": update, **stats, **loss_info, "lambda": self.lagrange.mean().item()}
            history.append(row)
            print(
                f"update={update:04d} reward={row['reward']:.2f} cost={row['cost']:.3f} "
                f"lambda={row['lambda']:.3f} loss={row['loss']:.3f}"
            )
        return history

    @torch.no_grad()
    def collect_rollout(self, obs):
        buffer = RolloutBuffer()
        ep_reward = 0.0
        ep_cost = 0.0
        n_agents = self.env.cfg.n_agents

        for _ in range(self.cfg.rollout_steps):
            nodes, edge, mask, self_state = collate_graphs([obs], self.device)
            action, logp, _, graph_emb = self.model.act(nodes, edge, mask, self_state)
            pooled = graph_emb.view(1, n_agents, -1).mean(dim=1)
            value_r = self.model.reward_critic(pooled).repeat(n_agents)
            value_c = self.model.cost_critic(pooled).repeat(n_agents)

            next_obs, reward, cost, terminated, truncated, _ = self.env.step(action.cpu().numpy())
            done = np.full(n_agents, terminated or truncated, dtype=np.float32)
            buffer.add(
                obs,
                action.cpu().numpy(),
                logp.cpu().numpy(),
                reward,
                cost,
                value_r.cpu().numpy(),
                value_c.cpu().numpy(),
                done,
            )
            ep_reward += float(reward.mean())
            ep_cost += float(cost.mean())
            obs = self.env.reset() if terminated or truncated else next_obs

        stats = {"reward": ep_reward / self.cfg.rollout_steps, "cost": ep_cost / self.cfg.rollout_steps}
        return buffer, obs, stats

    def update(self, rollout: RolloutBuffer):
        data = rollout.tensors(self.device)
        n_steps, n_agents = data["actions"].shape
        adv_r, ret_r = generalized_advantage_estimate(
            data["rewards"], data["values_r"], data["dones"], self.cfg.gamma, self.cfg.gae_lambda
        )
        adv_c, ret_c = generalized_advantage_estimate(
            data["costs"], data["values_c"], data["dones"], self.cfg.gamma, self.cfg.gae_lambda
        )
        self._update_lagrange(data["log_probs"], adv_c, data["values_c"])

        lagr = self.lagrange.view(1, n_agents).detach()
        adv_hyb = adv_r - lagr * adv_c
        adv_hyb = (adv_hyb - adv_hyb.mean()) / adv_hyb.std().clamp_min(1e-6)

        nodes, edge, mask, self_state = collate_graphs(rollout.obs, self.device)
        flat_actions = data["actions"].reshape(-1)
        flat_old_logp = data["log_probs"].reshape(-1)
        flat_adv = adv_hyb.reshape(-1)
        flat_ret_r = ret_r.reshape(-1)
        flat_ret_c = ret_c.reshape(-1)
        n_total = flat_actions.numel()
        last_loss = torch.tensor(0.0, device=self.device)

        for _ in range(self.cfg.ppo_epochs):
            perm = torch.randperm(n_total, device=self.device)
            for start in range(0, n_total, self.cfg.minibatch_size):
                idx = perm[start : start + self.cfg.minibatch_size]
                new_logp_all, entropy_all, value_r_all, value_c_all = self.model.evaluate_actions(
                    nodes, edge, mask, self_state, flat_actions, n_agents=n_agents
                )
                new_logp = new_logp_all[idx]
                entropy = entropy_all[idx]
                value_r = value_r_all[idx]
                value_c = value_c_all[idx]
                ratio = (new_logp - flat_old_logp[idx]).exp()
                pg1 = ratio * flat_adv[idx]
                pg2 = torch.clamp(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef) * flat_adv[idx]
                actor_loss = -torch.min(pg1, pg2).mean()
                value_loss = nn.functional.mse_loss(value_r, flat_ret_r[idx]) + nn.functional.mse_loss(
                    value_c, flat_ret_c[idx]
                )
                entropy_loss = -entropy.mean()
                loss = actor_loss + self.cfg.value_coef * value_loss + self.cfg.entropy_coef * entropy_loss

                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optim.step()
                last_loss = loss.detach()

        return {
            "loss": float(last_loss.item()),
            "return_r": float(ret_r.mean().item()),
            "return_c": float(ret_c.mean().item()),
        }

    @torch.no_grad()
    def _update_lagrange(self, old_logp: torch.Tensor, adv_c: torch.Tensor, value_c: torch.Tensor) -> None:
        del old_logp
        violation = (value_c - self.cfg.cost_limit).mean(dim=0)
        delta_lambda = -(violation * (1.0 - self.cfg.gamma) + adv_c.mean(dim=0))
        self.lagrange.sub_(self.cfg.lambda_lr * delta_lambda)
        self.lagrange.clamp_(min=0.0)
