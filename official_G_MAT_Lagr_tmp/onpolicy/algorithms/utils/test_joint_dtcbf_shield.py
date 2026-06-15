from types import SimpleNamespace

import numpy as np
import torch

from multiagent.action_table import (
    build_action_table_np,
    decode_action_index,
)
from onpolicy.algorithms.utils.joint_dtcbf_shield import (
    DecentralizedPriorityJointDTCBFShield,
)


def make_args():
    return SimpleNamespace(
        cbf_dt=0.1,
        cbf_alpha=1.0,
        cbf_max_accel=0.5,
        cbf_safety_buffer=0.05,
        max_edge_dist=1.0,
        max_shield_neighbors=8,
        priority_metric="min_h",
        no_safe_action_strategy="backup",
        backup_action_mode="zero",
        graph_feat_type="relative",
    )


def test_action_table_decode_matches_actor_order():
    for grid_size in [9, 20, 21]:
        table = build_action_table_np(grid_size)
        for index in [0, grid_size // 2, grid_size ** 2 - 1]:
            np.testing.assert_allclose(
                decode_action_index(index, grid_size), table[index]
            )


def make_pair_observations():
    node_obs = torch.zeros(1, 2, 2, 6)
    node_obs[0, 0, 0] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.1, 0.0])
    node_obs[0, 0, 1] = torch.tensor([-0.25, 0.0, 0.0, 0.0, 0.1, 0.0])
    node_obs[0, 1, 0] = torch.tensor([0.25, 0.0, 0.0, 0.0, 0.1, 0.0])
    node_obs[0, 1, 1] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.1, 0.0])
    adj = torch.tensor(
        [[[[0.0, 0.25], [0.25, 0.0]], [[0.0, 0.25], [0.25, 0.0]]]]
    )
    agent_id = torch.tensor([[[0], [1]]])
    return node_obs, adj, agent_id


def action_index(table, action):
    target = torch.tensor(action, dtype=table.dtype)
    return int((table - target).square().sum(-1).argmin().item())


def test_pairwise_compatibility_rejects_mutual_acceleration():
    table = torch.as_tensor(build_action_table_np(9))
    shield = DecentralizedPriorityJointDTCBFShield(
        make_args(), table, torch.device("cpu")
    )
    node_obs, _, _ = make_pair_observations()
    toward_from_agent0 = action_index(table, [1.0, 0.0])
    toward_from_agent1 = action_index(table, [-1.0, 0.0])
    away_from_agent1 = action_index(table, [1.0, 0.0])
    compat, _ = shield.pairwise_compatibility(
        node_obs[0, 1], 1, 0, toward_from_agent0
    )
    assert not bool(compat[toward_from_agent1])
    assert bool(compat[away_from_agent1])


def test_joint_shield_masks_incompatible_action_and_samples_safe_pair():
    table = torch.as_tensor(build_action_table_np(9))
    shield = DecentralizedPriorityJointDTCBFShield(
        make_args(), table, torch.device("cpu")
    )
    node_obs, adj, agent_id = make_pair_observations()
    action_count = table.shape[0]
    logits = torch.zeros(1, 2, action_count)
    local_mask = torch.ones_like(logits, dtype=torch.bool)
    action0 = action_index(table, [1.0, 0.0])
    action1_unsafe = action_index(table, [-1.0, 0.0])
    logits[0, 0, action0] = 10.0
    logits[0, 1, action1_unsafe] = 10.0

    actions, log_probs, final_masks, stats = shield.sample(
        logits, local_mask, node_obs, adj, agent_id, deterministic=True
    )
    assert final_masks.all(dim=-1).logical_not().any()
    assert final_masks.any(dim=-1).all()
    assert not bool(final_masks[0, 1, action1_unsafe])
    assert torch.isfinite(log_probs).all()
    assert stats["num_pairwise_edges"] == 1.0

    compat, margins = shield.pairwise_compatibility(
        node_obs[0, 1], 1, 0, int(actions[0, 0].item())
    )
    chosen = int(actions[0, 1].item())
    assert bool(compat[chosen])
    assert float(margins[chosen]) >= 0.0
