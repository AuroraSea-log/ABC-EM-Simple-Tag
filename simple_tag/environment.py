from __future__ import annotations

import gymnasium as gym
import numpy as np
from mpe2 import simple_tag_v3

TARGET = "adversary_0"
OBSERVER = "agent_0"
HOLD = 0


class AlternatingTagEnv:
    """AEC environment driven in strict alternation, pursuer first."""

    def __init__(self, env):
        self.env = env

    @property
    def possible_agents(self):
        return self.env.possible_agents

    @property
    def agents(self):
        return self.env.agents

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def action_space(self, agent):
        return self.env.action_space(agent)

    def _observations(self):
        return {agent: self.env.observe(agent) for agent in self.possible_agents}

    def _flags(self, name):
        return {agent: bool(getattr(self.env, name)[agent]) for agent in self.possible_agents}

    def _rewards(self):
        return {agent: float(self.env.rewards[agent]) for agent in self.possible_agents}

    def _closed(self, terminations, truncations):
        return all(terminations[a] or truncations[a] for a in self.possible_agents)

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        return self._observations(), {}

    def step_target(self, action):
        self.env.step(int(action))
        self.env.step(HOLD)
        return (self._observations(), self._rewards(),
                self._flags("terminations"), self._flags("truncations"))

    def step_observer(self, action):
        self.env.step(HOLD)
        self.env.step(int(action))
        return (self._observations(), self._rewards(),
                self._flags("terminations"), self._flags("truncations"))

    def step(self, actions):
        observations, target_rewards, terminations, truncations = self.step_target(actions[TARGET])
        if self._closed(terminations, truncations):
            rewards = {TARGET: target_rewards[TARGET], OBSERVER: target_rewards[OBSERVER]}
            return observations, rewards, terminations, truncations, {}
        observations, observer_rewards, terminations, truncations = self.step_observer(actions[OBSERVER])
        rewards = {TARGET: target_rewards[TARGET], OBSERVER: observer_rewards[OBSERVER]}
        return observations, rewards, terminations, truncations, {}

    def close(self):
        self.env.close()


def make_env(config):
    env = simple_tag_v3.env(
        num_good=1, num_adversaries=1, num_obstacles=2,
        max_cycles=2 * config.max_cycles, continuous_actions=False,
        dynamic_rescaling=False, render_mode=None,
    )
    expected = {TARGET: (12,), OBSERVER: (10,)}
    if set(env.possible_agents) != set(expected):
        env.close()
        raise RuntimeError("Simple Tag must contain exactly one pursuer and one evader")
    for name, shape in expected.items():
        if env.observation_space(name).shape != shape or env.action_space(name).n != 5:
            env.close()
            raise RuntimeError(f"Unexpected MPE2 observation/action space for {name}")
    return AlternatingTagEnv(env)


def is_done(terminations, truncations) -> bool:
    return all(terminations.get(name, False) or truncations.get(name, False) for name in (TARGET, OBSERVER))


class PursuerTrainingEnv(gym.Env):
    """PPO observes only the pursuer; the evader holds position on its turn."""

    metadata = {"render_modes": []}

    def __init__(self, config, seed: int):
        super().__init__()
        self.env = make_env(config)
        self.base_seed = seed
        self.episode = 0
        self.action_space = self.env.action_space(TARGET)
        self.observation_space = self.env.observation_space(TARGET)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        actual_seed = self.base_seed + self.episode if seed is None else int(seed)
        self.episode += 1
        self.rng = np.random.default_rng(actual_seed + 5_001)
        observations, _ = self.env.reset(seed=actual_seed)
        return observations[TARGET].astype(np.float32), {}

    def step(self, action):
        observations, rewards, terminations, truncations, _ = self.env.step({
            TARGET: int(np.asarray(action).item()),
            OBSERVER: int(self.rng.integers(5)),
        })
        return (
            observations[TARGET].astype(np.float32), float(rewards[TARGET]),
            bool(terminations[TARGET]), bool(truncations[TARGET]), {},
        )

    def close(self):
        self.env.close()
