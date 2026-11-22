from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
from .config import fingerprint
from .environment import TARGET, OBSERVER, PursuerTrainingEnv, is_done, make_env
from .utils import device_for, seed_everything, sha256, write_json

def validate_policy(model, config, seed: int, deterministic=True, details=False):
    returns = []
    entropies, maxima = ([], [])
    env = make_env(config)
    try:
        for episode in range(config.pretrain_eval_episodes):
            episode_seed = config.pretrain_seed + 1000000000 + seed * 10000 + episode * 997
            rng = np.random.default_rng(episode_seed + 5001)
            target_rng = np.random.default_rng(episode_seed + 6001)
            observations, _ = env.reset(seed=episode_seed)
            total = 0.0
            while env.agents:
                with torch.no_grad():
                    tensor, _ = model.policy.obs_to_tensor(observations[TARGET])
                    probs = model.policy.get_distribution(tensor).distribution.probs.cpu().numpy()[0].astype(float)
                probs /= probs.sum()
                entropies.append(float(-(probs * np.log(probs.clip(1e-30))).sum()))
                maxima.append(float(probs.max()))
                action = int(probs.argmax()) if deterministic else int(target_rng.choice(5, p=probs))
                observations, rewards, terminations, truncations, _ = env.step({TARGET: int(np.asarray(action).item()), OBSERVER: int(rng.integers(5))})
                total += float(rewards[TARGET])
                if is_done(terminations, truncations):
                    break
            returns.append(total)
    finally:
        env.close()
    result = {'mean_target_return': float(np.mean(returns)), 'entropy': float(np.mean(entropies)), 'max_probability': float(np.mean(maxima)), 'returns': returns}
    return result if details else result['mean_target_return']

def checkpoint_paths(root: Path, seed: int):
    path = root / 'opponents' / f'seed_{seed}.zip'
    return (path, path.with_suffix('.json'))

def require_opponent(config, root: Path, seed: int) -> Path:
    path, manifest_path = checkpoint_paths(root, seed)
    if not path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f'Missing frozen opponent for seed {seed}; run --phase pretrain first')
    saved = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected = fingerprint({**config.pretrain_signature(), 'seed': seed})
    if saved['signature'] != expected or saved['checkpoint_sha256'] != sha256(path):
        raise ValueError(f'Incompatible or modified opponent checkpoint: {path}; use a new output directory')
    return path

def pretrain(config, root: Path, seed: int) -> Path:
    path, manifest_path = checkpoint_paths(root, seed)
    if path.exists() and manifest_path.exists():
        return require_opponent(config, root, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    seed_everything(config.pretrain_seed + seed, config.threads)
    factories = [lambda i=i: PursuerTrainingEnv(config, config.pretrain_seed + seed * 1000003 + i * 100003) for i in range(config.pretrain_envs)]
    vec_env = DummyVecEnv(factories)

    class Selection(BaseCallback):

        def __init__(self):
            super().__init__()
            self.rows = []
            self.best = -float('inf')
            self.selected_step = None
            self.next_eval = config.pretrain_eval_interval

        def evaluate(self):
            deterministic = config.pretrain_selection == 'deterministic'
            result = validate_policy(self.model, config, seed, deterministic=deterministic, details=True)
            mean_return = result['mean_target_return']
            self.rows.append({'step': self.num_timesteps, **result})
            write_json(path.parent / f'seed_{seed}_progress.json', {'validation': self.rows})
            print(f"  PPO seed={seed} step={self.num_timesteps} validation_return={mean_return:.3f} entropy={result['entropy']:.3f}", flush=True)
            if mean_return > self.best:
                self.best = mean_return
                self.selected_step = self.num_timesteps
                temporary = path.with_name(path.stem + '.tmp.zip')
                self.model.save(temporary)
                temporary.replace(path)

        def _on_step(self):
            if self.num_timesteps >= self.next_eval:
                self.evaluate()
                while self.next_eval <= self.num_timesteps:
                    self.next_eval += config.pretrain_eval_interval
            return True
    try:
        model = PPO('MlpPolicy', vec_env, learning_rate=config.pretrain_lr, n_steps=config.pretrain_rollout, batch_size=config.pretrain_batch, n_epochs=config.pretrain_epochs, gamma=config.pretrain_gamma, ent_coef=config.pretrain_entropy, seed=config.pretrain_seed + seed, device=device_for(config.pretrain_device), verbose=0, policy_kwargs={'net_arch': {'pi': [config.hidden] * 2, 'vf': [config.hidden] * 2}})
        callback = Selection()
        model.learn(total_timesteps=config.pretrain_steps, callback=callback)
        callback.evaluate()
        write_json(manifest_path, {'signature': fingerprint({**config.pretrain_signature(), 'seed': seed}), 'training_config': config.pretrain_signature(), 'run_seed': seed, 'model_seed': config.pretrain_seed + seed, 'nominal_transitions': config.pretrain_steps, 'actual_transitions': model.num_timesteps, 'selected_step': callback.selected_step, 'selected_validation_return': callback.best, 'validation': callback.rows, 'checkpoint_sha256': sha256(path)})
    finally:
        vec_env.close()
    return path

class FrozenOpponent:

    def __init__(self, path: Path, device: str):
        self.model = PPO.load(path, device=device_for(device))
        self.model.policy.set_training_mode(False)
        for parameter in self.model.policy.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def probabilities(self, states):
        tensor, _ = self.model.policy.obs_to_tensor(np.asarray(states, dtype=np.float32))
        distribution = self.model.policy.get_distribution(tensor).distribution
        return distribution.probs.cpu().numpy().astype(np.float64)

    def sample(self, states, rng):
        probabilities = self.probabilities(states)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return np.asarray([rng.choice(5, p=row) for row in probabilities], dtype=np.int64)


def fixed_test_set(config, opponent, root, seed, checkpoint):
    """Independent evaluation rollouts; probabilities are never used for training."""
    path = root / "test_sets" / f"seed_{seed}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    test_seed = config.test_seed + seed * 100_003
    signature = fingerprint({"checkpoint": sha256(checkpoint), "test_size": config.test_size,
                             "test_seed": test_seed, "max_cycles": config.max_cycles})
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["signature"].item()) != signature:
                raise ValueError("Evaluation set configuration changed; choose a new output directory")
            return saved["states"], saved["actions"], saved["probabilities"]
    env = make_env(config)
    rng = np.random.default_rng(test_seed)
    states, actions, probabilities = [], [], []
    episode = 0
    try:
        while len(states) < config.test_size:
            observations, _ = env.reset(seed=test_seed + episode * 10_007)
            episode += 1
            while env.agents and len(states) < config.test_size:
                state = observations[TARGET].astype(np.float32)
                probs = opponent.probabilities(state[None])[0]
                probs /= probs.sum()
                action = int(rng.choice(5, p=probs))
                states.append(state.copy())
                actions.append(action)
                probabilities.append(probs)
                observations, _, terminated, truncated, _ = env.step(
                    {TARGET: action, OBSERVER: int(rng.integers(5))})
                if is_done(terminated, truncated):
                    break
    finally:
        env.close()
    result = np.stack(states), np.asarray(actions, np.int64), np.stack(probabilities)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, states=result[0], actions=result[1], probabilities=result[2], signature=signature)
    temporary.replace(path)
    return result
