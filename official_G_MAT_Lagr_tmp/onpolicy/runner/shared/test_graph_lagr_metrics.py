import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from onpolicy.runner.shared.graph_lagr_mpe_runner import GSMPERunner


def make_runner(tmp_path):
    runner = GSMPERunner.__new__(GSMPERunner)
    runner.metric_window = 3
    runner.recent_episode_rewards = deque(maxlen=3)
    runner.recent_episode_costs = deque(maxlen=3)
    runner.recent_episode_collisions = deque(maxlen=3)
    runner.best_model_cost_limit = 1.0
    runner.best_safe_reward = -np.inf
    runner.best_fallback_cost = np.inf
    runner.best_model_dir = str(Path(tmp_path) / "best")
    runner.saved = 0

    def save(save_dir=None):
        runner.saved += 1
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    runner.save = save
    return runner


def test_rolling_metrics_use_completed_rollout_episodes(tmp_path):
    runner = make_runner(tmp_path)
    runner.buffer = SimpleNamespace(
        rewards=np.array(
            [
                [[[1.0], [3.0]], [[5.0], [7.0]]],
                [[[1.0], [3.0]], [[5.0], [7.0]]],
            ]
        ),
        costs=np.array(
            [
                [[[0.0], [2.0]], [[1.0], [3.0]]],
                [[[0.0], [2.0]], [[1.0], [3.0]]],
            ]
        ),
    )
    infos = [
        [
            {"Num_agent_collisions": 2, "Num_obst_collisions": 1},
            {"Num_agent_collisions": 2, "Num_obst_collisions": 0},
        ],
        [
            {"Num_agent_collisions": 0, "Num_obst_collisions": 1},
            {"Num_agent_collisions": 0, "Num_obst_collisions": 2},
        ],
    ]
    reward, cost, collisions = runner._update_rolling_metrics(infos)
    assert list(runner.recent_episode_rewards) == [4.0, 12.0]
    assert list(runner.recent_episode_costs) == [2.0, 4.0]
    assert list(runner.recent_episode_collisions) == [3.0, 3.0]
    assert reward == 8.0
    assert cost == 3.0
    assert collisions == 3.0


def test_best_model_waits_for_full_window_and_respects_cost(tmp_path):
    runner = make_runner(tmp_path)
    runner.recent_episode_rewards.extend([10.0, 20.0])
    runner.recent_episode_costs.extend([0.0, 0.0])
    runner.recent_episode_collisions.extend([0.0, 0.0])
    assert not runner._save_best_model(15.0, 0.0, 0.0, 1, 100)

    runner.recent_episode_rewards.append(30.0)
    runner.recent_episode_costs.append(0.0)
    runner.recent_episode_collisions.append(0.0)
    assert runner._save_best_model(20.0, 0.5, 1.0, 2, 200)
    assert not runner._save_best_model(100.0, 2.0, 5.0, 3, 300)
    assert runner._save_best_model(25.0, 0.8, 2.0, 4, 400)
    assert runner.saved == 2

    metadata = json.loads(
        (Path(runner.best_model_dir) / "best_model.json").read_text()
    )
    assert metadata["episode"] == 4
    assert metadata["average_episode_rewards"] == 25.0
    assert metadata["average_episode_collisions"] == 2.0


def test_best_model_is_disabled_without_save_dir(tmp_path):
    runner = make_runner(tmp_path)
    runner.best_model_dir = None
    runner.recent_episode_rewards.extend([10.0, 20.0, 30.0])
    runner.recent_episode_costs.extend([0.0, 0.0, 0.0])
    assert not runner._save_best_model(30.0, 0.0, 0.0, 1, 100)
