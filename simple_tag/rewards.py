from __future__ import annotations
import numpy as np
import torch
from .models import Clone


class PositionCounter:
    def __init__(self, bins):
        self.bins = bins
        self.counts = np.zeros((bins, bins), dtype=np.int64)

    def indices(self, states):
        positions = np.clip(np.asarray(states)[:, 2:4], -1, 1)
        return np.clip(np.floor((positions + 1) * self.bins / 2).astype(np.int64), 0, self.bins - 1)

    def observe(self, states):
        indices = self.indices(states)
        np.add.at(self.counts, (indices[:, 0], indices[:, 1]), 1)

    def lookup(self, states):
        indices = self.indices(states)
        return self.counts[indices[:, 0], indices[:, 1]]


class JointCounter(PositionCounter):
    """Unbounded joint-position bins; retain the legacy position grid for reporting."""

    def __init__(self, bins, width):
        super().__init__(bins)
        self.width = width
        self.joint_counts = {}

    def keys(self, states):
        states = np.asarray(states)
        joint = np.concatenate((states[:, 2:4], states[:, 2:4] + states[:, 8:10]), axis=1)
        return [tuple(row) for row in np.floor(joint / self.width).astype(np.int64)]

    def observe(self, states):
        super().observe(states)
        for key in self.keys(states):
            self.joint_counts[key] = self.joint_counts.get(key, 0) + 1

    def lookup(self, states):
        return np.asarray([self.joint_counts.get(key, 0) for key in self.keys(states)])


def gated_reward(entropy, kl, counts, config):
    beta = -np.expm1(-config.delta * np.maximum(kl, 0))
    raw = entropy * beta + config.warmup_b / (counts + 1)
    return config.reward_scale * np.tanh(raw / config.reward_scale), raw, beta


class Acquisition:
    """ABC-EM entropy/change acquisition with a joint-position visitation map."""

    def __init__(self, config):
        self.config = config
        self.counter = JointCounter(config.position_bins, config.counter_width)
        self.clone = Clone(config)
        self.clone.generator.manual_seed(torch.initial_seed())
        self.updates = 0
        self.reward_sq = 0.0
        self.reward_samples = 0

    @property
    def reward_rms(self):
        corrected = self.reward_sq / max(1 - self.config.intrinsic_rms_decay**self.reward_samples, 1e-30)
        return max(float(np.sqrt(corrected)), self.config.intrinsic_rms_floor)

    def _score(self, states):
        entropy, change = self.clone.scores(states)
        rewards, raw, beta = gated_reward(entropy, change, self.counter.lookup(states), self.config)
        return np.asarray(rewards, dtype=np.float32), {"entropy": entropy, "raw_reward": raw, "beta": beta}

    def values(self, states):
        rewards = self._score(states)[0]
        if self.config.intrinsic_normalization == "rms":
            rewards = self.config.reward_scale * np.tanh(rewards / self.reward_rms / self.config.reward_scale)
        return rewards.astype(np.float32)
