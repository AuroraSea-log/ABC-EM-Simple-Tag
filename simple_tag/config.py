from __future__ import annotations
from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
PROTOCOL = 'simple-tag-1v1-v1'

@dataclass(frozen=True)
class Config:
    num_envs: int = 8
    max_cycles: int = 100
    total_interactions: int = 40000
    clone_capacity: int = 100000
    clone_interval: int = 1000
    eval_interval: int = 1000
    test_size: int = 4096
    test_seed: int = 20260101
    hidden: int = 128
    clone_lr: float = 0.0003
    clone_weight_decay: float = 1e-05
    clone_batch: int = 256
    grad_clip: float = 5.0
    delta: float = 30.0
    warmup_b: float = 1.5
    position_bins: int = 8
    reward_scale: float = 10.0
    dqn_lr: float = 0.0003
    gamma: float = 0.99
    replay_capacity: int = 20000
    dqn_batch: int = 512
    start_steps: int = 1000
    target_interval: int = 250
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_steps: int = 10000
    pretrain_steps: int = 500000
    pretrain_envs: int = 8
    pretrain_rollout: int = 256
    pretrain_batch: int = 512
    pretrain_epochs: int = 4
    pretrain_lr: float = 0.0003
    pretrain_gamma: float = 0.99
    pretrain_entropy: float = 0.01
    pretrain_eval_interval: int = 50000
    pretrain_eval_episodes: int = 16
    pretrain_seed: int = 7000
    pretrain_selection: str = 'sampled'
    observer_rng_offset: int = 3000000
    counter_width: float = 0.5
    observer_boundary_cost: float = 1.0
    replay_relabel: bool = True
    isolated_clone_rng: bool = True
    intrinsic_normalization: str = 'rms'
    intrinsic_rms_decay: float = 0.9998
    intrinsic_rms_floor: float = 0.05
    device: str = 'cpu'
    pretrain_device: str = 'cpu'
    opponent_device: str = 'cpu'
    threads: int = 1

    @classmethod
    def load(cls, path: str | Path) -> Config:
        values = json.loads(Path(path).read_text(encoding='utf-8'))
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f'Unknown configuration fields: {sorted(unknown)}')
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if field.type == 'int' and field.name != 'observer_rng_offset' and (type(value) is not int or value <= 0):
                raise ValueError(f'{field.name} must be a positive integer')
            if isinstance(value, float) and (not __import__('math').isfinite(value)):
                raise ValueError(f'{field.name} must be finite')
        if type(self.observer_rng_offset) is not int or self.observer_rng_offset < 0:
            raise ValueError('observer_rng_offset must be a nonnegative integer')
        if self.pretrain_selection not in ('deterministic', 'sampled'):
            raise ValueError('Invalid PPO checkpoint selection')
        if self.intrinsic_normalization not in ('none', 'rms'):
            raise ValueError('Invalid reward normalization')
        if self.clone_capacity < self.total_interactions:
            raise ValueError('Clone data capacity must retain the full interaction budget')
        if self.pretrain_batch > self.pretrain_envs * self.pretrain_rollout:
            raise ValueError('PPO batch exceeds its rollout')
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError('Invalid exploration schedule')
        if not 0 < self.gamma <= 1 or not 0 < self.pretrain_gamma <= 1:
            raise ValueError('Invalid discount')
        if not 0 <= self.intrinsic_rms_decay < 1:
            raise ValueError('Invalid reward moment decay')
        for key in ('clone_lr', 'dqn_lr', 'pretrain_lr', 'grad_clip', 'counter_width', 'reward_scale', 'intrinsic_rms_floor'):
            if getattr(self, key) <= 0:
                raise ValueError(f'{key} must be positive')
        for key in ('delta', 'warmup_b', 'observer_boundary_cost', 'clone_weight_decay'):
            if getattr(self, key) < 0:
                raise ValueError(f'{key} must be nonnegative')

    def as_dict(self) -> dict:
        return asdict(self)

    def pretrain_signature(self) -> dict:
        keys = [f.name for f in fields(self) if f.name.startswith('pretrain_')]
        keys += ['hidden', 'max_cycles', 'threads']
        return {'protocol': PROTOCOL, **{k: getattr(self, k) for k in keys}}

def fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
