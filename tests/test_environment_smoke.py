import numpy as np

from rl_env.path_env import RlGame


def test_structured_environment_step():
    env = RlGame(leader_count=1, follower_count=2, obstacle_num=1, render=False)
    try:
        observation = env.reset()
        assert set(observation) == {"leader", "followers"}
        assert observation["leader"].shape == (7,)
        assert len(observation["followers"]) == 2

        action = {"leader": [0.0, 0.0], "followers": [[0.0, 0.0], [0.0, 0.0]]}
        next_observation, reward, done, win, team_counter, distances = env.step(action)

        assert set(next_observation) == {"leader", "followers"}
        assert set(reward) == {"leader", "followers"}
        assert isinstance(done, bool)
        assert isinstance(win, (bool, np.bool_))
        assert isinstance(team_counter, float)
        assert isinstance(distances, dict)
    finally:
        env.close()


def test_legacy_environment_step():
    env = RlGame(n=1, m=2, obstacle_num=1, render=False)
    try:
        observation = env.reset()
        assert observation.shape == (3, 7)

        action = np.zeros((3, 2), dtype=np.float32)
        next_observation, reward, *_ = env.step(action)

        assert next_observation.shape == (3, 7)
        assert reward.shape == (3,)
    finally:
        env.close()
