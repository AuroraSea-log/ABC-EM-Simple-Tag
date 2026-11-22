from dataclasses import replace
import numpy as np
import pytest
import torch
from simple_tag.config import Config
from simple_tag.environment import make_env, TARGET, OBSERVER
from simple_tag.models import DQN
from simple_tag.rewards import Acquisition
from simple_tag.staged import StageSettings, episode_holdout, fit_clones, fit_controller, frozen_rewards
from simple_tag.utils import seed_everything


def tiny_settings():
    return StageSettings(clone_min_steps=2, clone_max_steps=12, clone_check_interval=2, clone_patience=2,
                         controller_min_steps=4, controller_max_steps=12, controller_check_interval=2,
                         controller_patience=2, probe_size=16)


def test_environment_joint_action_and_spaces():
    env = make_env(Config())
    try:
        observations, _ = env.reset(seed=11)
        assert observations[TARGET].shape == (12,)
        assert observations[OBSERVER].shape == (10,)
        observations, rewards, _, _, _ = env.step({TARGET: 1, OBSERVER: 2})
        assert set(rewards) == {TARGET, OBSERVER}
        assert np.isfinite(observations[TARGET]).all()
    finally:
        env.close()


def test_episode_holdout_is_consistent():
    envs = np.tile(np.arange(8), 100)
    episodes = np.repeat(np.arange(10), 80)
    mask = episode_holdout(envs, episodes, 0, 8)
    np.testing.assert_array_equal(mask[:123], episode_holdout(envs[:123], episodes[:123], 0, 8))
    for env, episode in set(zip(envs, episodes)):
        assert len(set(mask[(envs == env) & (episodes == episode)])) == 1
    assert mask.any() and (~mask).any()


def test_clone_warm_start_and_previous_snapshot():
    seed_everything(11)
    config = replace(Config(), hidden=8, clone_lr=.01)
    acquisition = Acquisition(config)
    states, actions = np.ones((40, 12), np.float32), np.zeros(40, np.int64)
    validation = np.arange(40) % 5 == 0
    initial = {k: v.clone() for k, v in acquisition.clone.model.state_dict().items()}
    first, _ = fit_clones(acquisition, states, actions, validation, tiny_settings(), 0, 1000)
    assert first[0]["best_validation_nll"] < first[0]["initial_validation_nll"]
    for key, value in acquisition.clone.previous.state_dict().items():
        torch.testing.assert_close(value, initial[key])
    accepted = {k: v.clone() for k, v in acquisition.clone.model.state_dict().items()}
    old_step = next(iter(acquisition.clone.optimizer.state.values()))["step"].item()
    fit_clones(acquisition, states, actions, validation, tiny_settings(), 0, 2000)
    for key, value in acquisition.clone.previous.state_dict().items():
        torch.testing.assert_close(value, accepted[key])
    assert next(iter(acquisition.clone.optimizer.state.values()))["step"].item() > old_step


def test_frozen_rewards_and_controller_continuation():
    seed_everything(12)
    config = replace(Config(), hidden=8, replay_capacity=40, dqn_batch=16, target_interval=2)
    controller, acquisition = DQN(config), Acquisition(config)
    rng = np.random.default_rng(2)
    states = rng.normal(size=(40, 10)).astype(np.float32)
    targets = rng.normal(size=(40, 12)).astype(np.float32)
    controller.add(states, rng.integers(5, size=40), np.zeros(40), states, np.ones(40), targets, np.zeros(40))
    settings = tiny_settings()
    rewards = frozen_rewards(acquisition, controller, settings)
    preserved = rewards.copy()
    acquisition.counter.observe(targets)
    acquisition.reward_sq = 999.
    acquisition.reward_samples = 100
    np.testing.assert_array_equal(rewards, preserved)
    with pytest.raises(ValueError):
        rewards[0] = 100.
    mask = np.arange(40) % 5 == 0
    first, _ = fit_controller(controller, rewards, mask, settings, rng, 1000)
    second, _ = fit_controller(controller, rewards, mask, settings, rng, 2000)
    assert controller.updates == first["steps"] + second["steps"]
    assert next(iter(controller.optimizer.state.values()))["step"].item() == controller.updates
