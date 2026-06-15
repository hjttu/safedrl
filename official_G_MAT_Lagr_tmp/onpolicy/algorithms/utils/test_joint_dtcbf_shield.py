from types import SimpleNamespace
from types import MethodType

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
        dtcbf_horizon=3,
        dtcbf_predict_mode="constant_action",
        dtcbf_min_margin=0.0,
        dtcbf_early_brake_buffer=0.05,
        predictive_hard_neighbors=1,
        min_joint_domain_size=5,
        predictive_soft_penalty_coef=2.0,
        use_soft_predictive_penalty=True,
        recovery_margin_coef=10.0,
        recovery_logit_coef=1.0,
        recovery_progress_coef=0.1,
        use_joint_repair=True,
        joint_repair_top_k=8,
        joint_repair_max_cluster_size=4,
        joint_repair_include_blockers=True,
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
    node_obs[0, 0, 1] = torch.tensor([-0.32, 0.0, 0.0, 0.0, 0.1, 0.0])
    node_obs[0, 1, 0] = torch.tensor([0.32, 0.0, 0.0, 0.0, 0.1, 0.0])
    node_obs[0, 1, 1] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.1, 0.0])
    adj = torch.tensor(
        [[[[0.0, 0.32], [0.32, 0.0]], [[0.0, 0.32], [0.32, 0.0]]]]
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
    node_obs[0, 0, 1, 0] = -0.30
    node_obs[0, 1, 0, 0] = 0.30
    toward_from_agent0 = action_index(table, [1.0, 0.0])
    toward_from_agent1 = action_index(table, [-1.0, 0.0])
    away_from_agent1 = action_index(table, [1.0, 0.0])
    compat, _ = shield.multi_step_pairwise_margin_vector(
        node_obs[0, 1], 1, 0, toward_from_agent0, horizon=1
    )
    assert not bool(compat[toward_from_agent1])
    assert bool(compat[away_from_agent1])


def test_joint_shield_masks_incompatible_action_and_samples_safe_pair():
    table = torch.as_tensor(build_action_table_np(9))
    shield = DecentralizedPriorityJointDTCBFShield(
        make_args(), table, torch.device("cpu")
    )
    node_obs, adj, agent_id = make_pair_observations()
    node_obs[0, 0, 1, 0] = -0.30
    node_obs[0, 1, 0, 0] = 0.30
    adj.fill_(0.30)
    adj[:, :, 0, 0] = 0.0
    adj[:, :, 1, 1] = 0.0
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
    assert (final_masks > 0).any(dim=-1).all()
    assert torch.isfinite(log_probs).all()
    state = shield._constraint_state(
        1,
        {0: action0},
        local_mask[0, 1],
        node_obs[0],
        adj[0],
        [0, 1],
    )
    assert not bool(state["mask_h1"][action1_unsafe])

def test_predictive_horizon_rejects_later_collision():
    table = torch.as_tensor(build_action_table_np(9))
    args = make_args()
    args.cbf_alpha = 1.0
    args.dtcbf_early_brake_buffer = 0.0
    node_obs, _, _ = make_pair_observations()
    node_obs[0, 1, 0, 0] = 0.5
    node_obs[0, 1, 0, 2] = -0.8
    fixed_action = action_index(table, [0.0, 0.0])
    toward = action_index(table, [-1.0, 0.0])

    args.dtcbf_horizon = 1
    one_step = DecentralizedPriorityJointDTCBFShield(
        args, table, torch.device("cpu")
    )
    one_step_compat, _ = one_step.pairwise_compatibility(
        node_obs[0, 1], 1, 0, fixed_action
    )

    args.dtcbf_horizon = 3
    predictive = DecentralizedPriorityJointDTCBFShield(
        args, table, torch.device("cpu")
    )
    predictive_compat, _ = predictive.pairwise_compatibility(
        node_obs[0, 1], 1, 0, fixed_action
    )
    assert bool(one_step_compat[toward])
    assert not bool(predictive_compat[toward])


def test_joint_repair_changes_blocking_earlier_action():
    table = torch.as_tensor(build_action_table_np(9))
    args = make_args()
    args.cbf_alpha = 1.0
    args.dtcbf_horizon = 1
    args.dtcbf_early_brake_buffer = 0.0
    shield = DecentralizedPriorityJointDTCBFShield(
        args, table, torch.device("cpu")
    )
    node_obs, adj, agent_id = make_pair_observations()
    node_obs[0, 0, 1, 0] = -0.25
    node_obs[0, 1, 0, 0] = 0.25
    adj.fill_(0.25)
    adj[:, :, 0, 0] = 0.0
    adj[:, :, 1, 1] = 0.0
    action_count = table.shape[0]
    logits = torch.full((1, 2, action_count), -20.0)
    local_mask = torch.zeros_like(logits, dtype=torch.bool)
    toward0 = action_index(table, [1.0, 0.0])
    away0 = action_index(table, [-1.0, 0.0])
    toward1 = action_index(table, [-1.0, 0.0])
    local_mask[0, 0, [toward0, away0]] = True
    local_mask[0, 1, toward1] = True
    logits[0, 0, toward0] = 10.0
    logits[0, 0, away0] = 9.0
    logits[0, 1, toward1] = 10.0

    actions, _, _, stats = shield.sample(
        logits, local_mask, node_obs, adj, agent_id, deterministic=True
    )
    assert int(actions[0, 0].item()) == away0
    assert int(actions[0, 1].item()) == toward1
    assert stats["num_joint_repair_successes"] == 1.0
    assert stats["num_recovery_used"] == 0.0


def test_action_grid_sizes_supported_by_predictive_shield():
    node_obs, adj, agent_id = make_pair_observations()
    for grid_size in [9, 20, 21]:
        table = torch.as_tensor(build_action_table_np(grid_size))
        shield = DecentralizedPriorityJointDTCBFShield(
            make_args(), table, torch.device("cpu")
        )
        logits = torch.zeros(1, 2, grid_size ** 2)
        local_mask = torch.ones_like(logits, dtype=torch.bool)
        actions, log_probs, final_masks, _ = shield.sample(
            logits, local_mask, node_obs, adj, agent_id, deterministic=True
        )
        assert actions.shape == (1, 2, 1)
        assert final_masks.shape == (1, 2, grid_size ** 2)
        assert torch.isfinite(log_probs).all()


def test_empty_set_uses_maxmin_recovery_instead_of_brake():
    table = torch.as_tensor(build_action_table_np(9))
    args = make_args()
    args.cbf_alpha = 1.0
    args.dtcbf_horizon = 1
    args.dtcbf_early_brake_buffer = 0.0
    args.use_joint_repair = False
    shield = DecentralizedPriorityJointDTCBFShield(
        args, table, torch.device("cpu")
    )
    node_obs, adj, agent_id = make_pair_observations()
    node_obs[0, 0, 1, 0] = -0.1
    node_obs[0, 1, 0, 0] = 0.1
    adj.fill_(0.1)
    adj[:, :, 0, 0] = 0.0
    adj[:, :, 1, 1] = 0.0
    action_count = table.shape[0]
    logits = torch.zeros(1, 2, action_count)
    local_mask = torch.ones_like(logits, dtype=torch.bool)
    zero = action_index(table, [0.0, 0.0])
    local_mask[0, 0] = False
    local_mask[0, 0, zero] = True
    _, recovery_margins = shield.pairwise_compatibility(
        node_obs[0, 1], 1, 0, zero
    )
    expected_recovery = int(recovery_margins.argmax().item())

    actions, _, _, stats = shield.sample(
        logits, local_mask, node_obs, adj, agent_id, deterministic=True
    )
    assert int(actions[0, 1].item()) == expected_recovery
    assert int(actions[0, 1].item()) != zero
    assert stats["num_recovery_used"] == 1.0


def make_layered_constraint_fixture(min_domain_size=1):
    table = torch.as_tensor(build_action_table_np(9))
    args = make_args()
    args.min_joint_domain_size = min_domain_size
    shield = DecentralizedPriorityJointDTCBFShield(
        args, table, torch.device("cpu")
    )
    action_count = table.shape[0]
    node_obs = torch.zeros(3, 3, 6)
    adj = torch.ones(3, 3, 3)
    for agent in range(3):
        adj[agent].fill_diagonal_(0.0)
    ego_indices = [0, 1, 2]

    def fake_margin(
        self, node_obs_i, ego_index_i, other_index, fixed_action_j, horizon
    ):
        margin = torch.ones(action_count)
        if horizon > 1 and other_index == 0:
            margin.fill_(0.1)
            margin[0] = -1.0
        elif horizon > 1 and other_index == 1:
            margin.fill_(2.0)
            margin[1] = -0.5
        return margin >= 0.0, margin

    shield.multi_step_pairwise_margin_vector = MethodType(
        fake_margin, shield
    )
    state = shield._constraint_state(
        2,
        {0: 0, 1: 0},
        torch.ones(action_count, dtype=torch.bool),
        node_obs,
        adj,
        ego_indices,
    )
    return shield, state


def test_only_top_risk_neighbor_uses_predictive_hard_mask():
    _, state = make_layered_constraint_fixture(min_domain_size=1)
    assert not bool(state["mask"][0])
    assert bool(state["mask"][1])
    assert state["penalty"][1] > 0


def test_predictive_domain_safeguard_relaxes_to_h1():
    _, state = make_layered_constraint_fixture(min_domain_size=81)
    assert state["predictive_relaxed"]
    assert torch.equal(state["mask"], state["mask_h1"])
    assert bool(state["mask"][0])
    assert state["penalty"][0] > 0
