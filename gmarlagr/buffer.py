from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class RolloutBuffer:
    obs: list[list[dict[str, np.ndarray]]] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    log_probs: list[np.ndarray] = field(default_factory=list)
    rewards: list[np.ndarray] = field(default_factory=list)
    costs: list[np.ndarray] = field(default_factory=list)
    values_r: list[np.ndarray] = field(default_factory=list)
    values_c: list[np.ndarray] = field(default_factory=list)
    dones: list[np.ndarray] = field(default_factory=list)

    def add(self, obs, action, log_prob, reward, cost, value_r, value_c, done):
        self.obs.append(obs)
        self.actions.append(np.asarray(action))
        self.log_probs.append(np.asarray(log_prob))
        self.rewards.append(np.asarray(reward, dtype=np.float32))
        self.costs.append(np.asarray(cost, dtype=np.float32))
        self.values_r.append(np.asarray(value_r, dtype=np.float32))
        self.values_c.append(np.asarray(value_c, dtype=np.float32))
        self.dones.append(np.asarray(done, dtype=np.float32))

    def tensors(self, device: torch.device):
        return {
            "actions": torch.as_tensor(np.asarray(self.actions), device=device).long(),
            "log_probs": torch.as_tensor(np.asarray(self.log_probs), device=device).float(),
            "rewards": torch.as_tensor(np.asarray(self.rewards), device=device).float(),
            "costs": torch.as_tensor(np.asarray(self.costs), device=device).float(),
            "values_r": torch.as_tensor(np.asarray(self.values_r), device=device).float(),
            "values_c": torch.as_tensor(np.asarray(self.values_c), device=device).float(),
            "dones": torch.as_tensor(np.asarray(self.dones), device=device).float(),
        }

    def __len__(self) -> int:
        return len(self.actions)


def collate_graphs(obs_seq: list[list[dict[str, np.ndarray]]], device: torch.device):
    flat = [agent_obs for step_obs in obs_seq for agent_obs in step_obs]
    max_nodes = max(item["nodes"].shape[0] for item in flat)
    node_dim = flat[0]["nodes"].shape[-1]
    nodes = np.zeros((len(flat), max_nodes, node_dim), dtype=np.float32)
    edge = np.zeros((len(flat), max_nodes, 1), dtype=np.float32)
    mask = np.zeros((len(flat), max_nodes), dtype=bool)
    self_state = np.zeros((len(flat), flat[0]["self_state"].shape[-1]), dtype=np.float32)
    for i, item in enumerate(flat):
        n = item["nodes"].shape[0]
        nodes[i, :n] = item["nodes"]
        edge[i, :n] = item["edge_dist"]
        mask[i, :n] = True
        self_state[i] = item["self_state"]
    return (
        torch.as_tensor(nodes, device=device),
        torch.as_tensor(edge, device=device),
        torch.as_tensor(mask, device=device),
        torch.as_tensor(self_state, device=device),
    )


def generalized_advantage_estimate(reward, value, done, gamma: float, lam: float):
    adv = torch.zeros_like(reward)
    last_gae = torch.zeros(reward.shape[1], device=reward.device)
    next_value = torch.zeros(reward.shape[1], device=reward.device)
    for t in reversed(range(reward.shape[0])):
        nonterminal = 1.0 - done[t]
        delta = reward[t] + gamma * next_value * nonterminal - value[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
        next_value = value[t]
    returns = adv + value
    return adv, returns
