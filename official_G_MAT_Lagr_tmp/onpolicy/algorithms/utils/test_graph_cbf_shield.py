import gym
import numpy as np
import torch

from onpolicy.algorithms.graph_actor_critic import GR_Actor
from onpolicy.config import get_config, graph_config


def make_args():
    parser = get_config()
    parser = graph_config([], parser)[1]
    args = parser.parse_args([])
    args.use_recurrent_policy = False
    args.use_naive_recurrent_policy = False
    args.graph_feat_type = "global"
    args.actor_graph_aggr = "node"
    args.num_agents = 2
    return args


def make_actor():
    args = make_args()
    return GR_Actor(
        args,
        gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
        gym.spaces.Box(-np.inf, np.inf, shape=(4, 6), dtype=np.float32),
        gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
        gym.spaces.Discrete(400),
    )


def test_hard_mask_and_safe_guide_projection():
    actor = make_actor()
    obs = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    node_obs = torch.tensor(
        [[
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.0],
            [0.25, 0.0, 0.0, 0.0, 0.1, 2.0],
            [1.0, 0.0, 0.0, 0.0, 0.1, 1.0],
            [-1.0, 0.0, 0.0, 0.0, 0.1, 0.0],
        ]]
    )
    adj = torch.tensor(
        [[[0.0, 0.25, 1.0, 1.0],
          [0.25, 0.0, 0.75, 1.25],
          [1.0, 0.75, 0.0, 2.0],
          [1.0, 1.25, 2.0, 0.0]]]
    )
    agent_id = torch.tensor([[0]])

    margins, hard_mask, _ = actor._cbf_action_features(
        obs, node_obs, adj, agent_id
    )
    assert hard_mask.any()
    assert (~hard_mask).any()
    guide_action = actor._safe_guide_actions(obs, hard_mask)
    assert hard_mask.gather(1, guide_action).all()
    assert torch.all(margins[hard_mask] >= 0.0)


def test_unsafe_actions_receive_zero_probability():
    actor = make_actor()
    batch = 1
    obs = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    node_obs = torch.tensor(
        [[
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.0],
            [0.25, 0.0, 0.0, 0.0, 0.1, 2.0],
            [1.0, 0.0, 0.0, 0.0, 0.1, 1.0],
            [-1.0, 0.0, 0.0, 0.0, 0.1, 0.0],
        ]]
    )
    adj = torch.tensor(
        [[[0.0, 0.25, 1.0, 1.0],
          [0.25, 0.0, 0.75, 1.25],
          [1.0, 0.75, 0.0, 2.0],
          [1.0, 1.25, 2.0, 0.0]]]
    )
    agent_id = torch.tensor([[0]])
    actor_features = torch.zeros(batch, actor.hidden_size)
    graph_emb = torch.zeros(batch, actor.gnn_base.out_dim)

    dist, _, hard_mask, safety_score = actor._shielded_distribution(
        actor_features, graph_emb, obs, node_obs, adj, agent_id
    )
    assert torch.all(dist.probs[~hard_mask] == 0.0)
    loss = safety_score.square().mean()
    loss.backward()
    assert actor.graph_cbf_head[-1].weight.grad is not None
