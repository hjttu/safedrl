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


def make_actor(action_grid_size=21, use_joint=False):
    args = make_args()
    args.action_grid_size = action_grid_size
    args.use_joint_dtcbf_shield = use_joint
    return GR_Actor(
        args,
        gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
        gym.spaces.Box(-np.inf, np.inf, shape=(4, 6), dtype=np.float32),
        gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
        gym.spaces.Discrete(action_grid_size ** 2),
    )


def test_hard_mask_and_safe_guide_projection():
    actor = make_actor(21)
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

    margins, hard_mask, _ = actor._local_dtcbf_action_features(
        obs, node_obs, adj, agent_id
    )
    assert hard_mask.any()
    assert (~hard_mask).any()
    guide_action = actor._safe_guide_actions(obs, hard_mask)
    assert hard_mask.gather(1, guide_action).all()
    assert torch.all(margins[hard_mask] >= 0.0)
    toward = (
        actor.action_table - torch.tensor([1.0, 0.0])
    ).square().sum(-1).argmin()
    away = (
        actor.action_table - torch.tensor([-1.0, 0.0])
    ).square().sum(-1).argmin()
    assert not bool(hard_mask[0, toward])
    assert bool(hard_mask[0, away])


def test_unsafe_actions_receive_zero_probability():
    actor = make_actor(21)
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


def test_final_mask_reproduces_rollout_log_probability():
    actor = make_actor(9, use_joint=True)
    obs = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    node_obs = torch.tensor(
        [[
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.1, 2.0],
            [1.0, 0.0, 0.0, 0.0, 0.1, 1.0],
            [-1.0, 0.0, 0.0, 0.0, 0.1, 0.0],
        ]]
    )
    adj = torch.tensor(
        [[[0.0, 1.0, 1.0, 1.0],
          [1.0, 0.0, 0.0, 0.0],
          [1.0, 0.0, 0.0, 0.0],
          [1.0, 0.0, 0.0, 0.0]]]
    )
    agent_id = torch.tensor([[0]])
    rnn_states = torch.zeros(1, actor._recurrent_N, actor.hidden_size)
    masks = torch.ones(1, 1)
    final_mask = torch.zeros(1, actor.action_table.shape[0])
    final_mask[:, [0, 10, 20]] = 1.0

    actions, rollout_log_probs, _ = actor.forward(
        obs, node_obs, adj, agent_id, rnn_states, masks,
        available_actions=final_mask, deterministic=True
    )
    update_log_probs, _, _, _ = actor.evaluate_actions(
        obs, node_obs, adj, agent_id, rnn_states, actions, masks,
        available_actions=final_mask
    )
    assert torch.allclose(rollout_log_probs, update_log_probs, atol=1e-6)
