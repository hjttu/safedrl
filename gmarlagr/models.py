from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(x.dtype)
    return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class GraphTransformerEncoder(nn.Module):
    """Transformer-style local graph encoder with edge-distance bias."""

    def __init__(
        self,
        node_dim: int = 6,
        embed_dim: int = 128,
        edge_dim: int = 1,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_proj = nn.Sequential(nn.Linear(node_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        self.edge_proj = nn.Linear(edge_dim, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.Tanh())

    def forward(self, nodes: torch.Tensor, edge_dist: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.node_proj(nodes) + self.edge_proj(edge_dist)
        h = self.transformer(h, src_key_padding_mask=~mask)
        return self.out(masked_mean(h, mask))


class Actor(nn.Module):
    def __init__(self, graph_dim: int, self_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(graph_dim + self_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_actions),
        )

    def distribution(self, graph_emb: torch.Tensor, self_state: torch.Tensor) -> Categorical:
        logits = self.net(torch.cat([self_state, graph_emb], dim=-1))
        return Categorical(logits=logits)


class Critic(nn.Module):
    def __init__(self, graph_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pooled_graph_emb: torch.Tensor) -> torch.Tensor:
        return self.net(pooled_graph_emb).squeeze(-1)


class GraphActorCritic(nn.Module):
    """Shared decentralized actor plus centralized reward/cost critics."""

    def __init__(
        self,
        n_actions: int,
        node_dim: int = 6,
        self_dim: int = 7,
        embed_dim: int = 128,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.encoder = GraphTransformerEncoder(node_dim=node_dim, embed_dim=embed_dim)
        self.actor = Actor(embed_dim, self_dim, n_actions, hidden_dim)
        self.reward_critic = Critic(embed_dim, hidden_dim)
        self.cost_critic = Critic(embed_dim, hidden_dim)

    def encode(self, nodes: torch.Tensor, edge_dist: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(nodes, edge_dist, mask)

    def act(self, nodes, edge_dist, mask, self_state):
        graph_emb = self.encode(nodes, edge_dist, mask)
        dist = self.actor.distribution(graph_emb, self_state)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), graph_emb

    def evaluate_actions(self, nodes, edge_dist, mask, self_state, actions, n_agents: int):
        graph_emb = self.encode(nodes, edge_dist, mask)
        dist = self.actor.distribution(graph_emb, self_state)
        pooled = graph_emb.view(-1, n_agents, graph_emb.shape[-1]).mean(dim=1)
        values_r = self.reward_critic(pooled).repeat_interleave(n_agents)
        values_c = self.cost_critic(pooled).repeat_interleave(n_agents)
        return dist.log_prob(actions), dist.entropy(), values_r, values_c
