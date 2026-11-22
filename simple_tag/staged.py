"""Warm-started round training with a fixed reward during each replay solve."""
from __future__ import annotations
import copy
from dataclasses import asdict, dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .models import double_dqn_targets

@dataclass(frozen=True)
class StageSettings:
    clone_min_steps: int = 200
    clone_max_steps: int = 5000
    clone_check_interval: int = 100
    clone_patience: int = 8
    clone_min_delta: float = 0.0001
    clone_relative_delta: float = 0.001
    controller_min_steps: int = 1000
    controller_max_steps: int = 10000
    controller_check_interval: int = 250
    controller_patience: int = 4
    controller_action_tolerance: float = 0.01
    controller_q_tolerance: float = 0.01
    controller_td_tolerance: float = 0.02
    probe_size: int = 2048
    reward_batch: int = 4096

    def validate(self, config):
        for key, value in asdict(self).items():
            if isinstance(value, int) and value <= 0:
                raise ValueError(f'{key} must be positive')
            if isinstance(value, float) and (not np.isfinite(value) or value < 0):
                raise ValueError(f'{key} must be finite and nonnegative')
        for prefix in ('clone', 'controller'):
            minimum = getattr(self, prefix + '_min_steps')
            maximum = getattr(self, prefix + '_max_steps')
            interval = getattr(self, prefix + '_check_interval')
            if maximum < minimum or maximum % interval:
                raise ValueError(f'Invalid {prefix} optimization limits')
        if self.controller_check_interval % config.target_interval:
            raise ValueError('Controller checks must follow complete target-network intervals')
        if config.clone_interval != 1000 or config.eval_interval != 1000:
            raise ValueError('Staged experiments require one clone/evaluation round per 1000 interactions')
        if config.total_interactions % config.clone_interval:
            raise ValueError('The acquisition budget must contain complete rounds')
        if not config.replay_relabel or not config.isolated_clone_rng:
            raise ValueError('Staged training requires relabeling and isolated clone RNGs')
        if config.num_envs < 2:
            raise ValueError('At least two environments are required for the episode holdout')
        if config.replay_capacity < 10 * config.num_envs * config.max_cycles:
            raise ValueError('Replay must retain ten episode cycles to preserve held-out groups')

def episode_holdout(env_ids, episodes, seed, num_envs):
    """Keep whole episodes out of gradient training, consistently across rounds."""
    env_ids, episodes = (np.asarray(env_ids), np.asarray(episodes))
    hashed = (env_ids * 1000003 + episodes * 10007 + seed * 97) % 10 == 0
    return np.where(episodes == 0, env_ids == seed % num_envs, hashed)

def _nll(model, states, labels, indices, batch=4096):
    total = torch.zeros((), device=states.device)
    with torch.no_grad():
        for start in range(0, len(indices), batch):
            selected = indices[start:start + batch]
            total += F.cross_entropy(model(states[selected]), labels[selected], reduction='sum')
    return float((total / len(indices)).cpu())

def fit_clones(acquisition, states, actions, validation, settings, seed, interactions):
    """Snapshot the previous round once; select weights using acquisition-only NLL."""
    train = np.flatnonzero(~validation)
    heldout = np.flatnonzero(validation)
    if not len(train) or not len(heldout):
        raise ValueError('Both training and held-out acquisition episodes are required')
    records, summaries = ([], [])
    for member, clone in enumerate([acquisition.clone]):
        clone.previous.load_state_dict(clone.model.state_dict())
        selected = train
        x = torch.as_tensor(states, dtype=torch.float32, device=clone.device)
        y = torch.as_tensor(actions, dtype=torch.long, device=clone.device)
        train_ids = torch.as_tensor(selected, dtype=torch.long, device=clone.device)
        validation_ids = torch.as_tensor(heldout, dtype=torch.long, device=clone.device)
        initial = best = significant = _nll(clone.model, x, y, validation_ids)
        best_model = copy.deepcopy(clone.model.state_dict())
        best_optimizer = copy.deepcopy(clone.optimizer.state_dict())
        stale, best_step, reason = (0, 0, 'max_steps')
        records.append({'phase': 'clone', 'interactions': interactions, 'member': member, 'steps': 0, 'validation_nll': initial})
        for step in range(1, settings.clone_max_steps + 1):
            clone.model.train()
            picks = torch.randint(len(train_ids), (min(clone.config.clone_batch, len(train_ids)),), device=clone.device, generator=clone.generator)
            indices = train_ids[picks]
            loss = F.cross_entropy(clone.model(x[indices]), y[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite clone training loss')
            clone.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(clone.model.parameters(), clone.config.grad_clip)
            clone.optimizer.step()
            if step % settings.clone_check_interval:
                continue
            clone.model.eval()
            value = _nll(clone.model, x, y, validation_ids)
            if not np.isfinite(value):
                raise FloatingPointError('Non-finite clone validation NLL')
            if value < best:
                best, best_step = (value, step)
                best_model = copy.deepcopy(clone.model.state_dict())
                best_optimizer = copy.deepcopy(clone.optimizer.state_dict())
            tolerance = settings.clone_min_delta + settings.clone_relative_delta * abs(significant)
            if significant - value > tolerance:
                significant, stale = (value, 0)
            else:
                stale += 1
            records.append({'phase': 'clone', 'interactions': interactions, 'member': member, 'steps': step, 'training_nll': float(loss.detach().cpu()), 'validation_nll': value, 'best_validation_nll': best})
            if step >= settings.clone_min_steps and stale >= settings.clone_patience:
                reason = 'validation_plateau'
                break
        clone.model.load_state_dict(best_model)
        clone.optimizer.load_state_dict(best_optimizer)
        clone.model.eval()
        summaries.append({'member': member, 'steps': step, 'selected_step': best_step, 'initial_validation_nll': initial, 'best_validation_nll': best, 'stop_reason': reason, 'training_samples': len(train), 'validation_samples': len(heldout)})
    acquisition.updates += 1
    return (summaries, records)

def frozen_rewards(acquisition, observer, settings):
    """Materialize immutable current-reward labels before any controller updates."""
    result = np.empty(observer.size, dtype=np.float32)
    for start in range(0, observer.size, settings.reward_batch):
        end = min(observer.size, start + settings.reward_batch)
        result[start:end] = acquisition.values(observer.target_next_states[start:end]) - observer.reward_costs[start:end]
    if not np.isfinite(result).all():
        raise FloatingPointError('Non-finite frozen reward')
    result.setflags(write=False)
    return result

def fit_controller(observer, rewards, validation, settings, rng, interactions):
    """Optimize fixed replay labels; stability is explicitly not a return guarantee."""
    n = observer.size
    train, heldout = (np.flatnonzero(~validation[:n]), np.flatnonzero(validation[:n]))
    if not len(train) or not len(heldout):
        raise ValueError('Controller replay requires training and held-out episodes')
    device = observer.device
    states = torch.as_tensor(observer.states[:n], device=device)
    next_states = torch.as_tensor(observer.next_states[:n], device=device)
    actions = torch.as_tensor(observer.actions[:n], device=device)
    terminals = torch.as_tensor(observer.terminals[:n], device=device)
    labels = torch.tensor(rewards, dtype=torch.float32, device=device)
    train_ids = torch.as_tensor(train, dtype=torch.long, device=device)
    probe = torch.as_tensor(heldout[np.linspace(0, len(heldout) - 1, min(len(heldout), settings.probe_size), dtype=int)], dtype=torch.long, device=device)
    generator = torch.Generator(device=device).manual_seed(int(rng.integers(2 ** 31)))

    def measure():
        with torch.no_grad():
            q = observer.online(states[probe])
            target = double_dqn_targets(labels[probe], terminals[probe], observer.online(next_states[probe]), observer.target(next_states[probe]), observer.config.gamma)
            td = F.smooth_l1_loss(q.gather(1, actions[probe, None]).squeeze(1), target)
        return (q.detach(), float(td.cpu()))
    previous_q, previous_td = measure()
    if not np.isfinite(previous_td):
        raise FloatingPointError('Non-finite controller probe loss')
    records = [{'phase': 'controller', 'interactions': interactions, 'steps': 0, 'heldout_td_loss': previous_td}]
    stable, reason = (0, 'max_steps')
    for step in range(1, settings.controller_max_steps + 1):
        ids = train_ids[torch.randint(len(train_ids), (min(observer.config.dqn_batch, len(train_ids)),), device=device, generator=generator)]
        prediction = observer.online(states[ids]).gather(1, actions[ids, None]).squeeze(1)
        with torch.no_grad():
            target = double_dqn_targets(labels[ids], terminals[ids], observer.online(next_states[ids]), observer.target(next_states[ids]), observer.config.gamma)
        loss = F.smooth_l1_loss(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite controller training loss')
        observer.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(observer.online.parameters(), observer.config.grad_clip)
        observer.optimizer.step()
        observer.updates += 1
        if observer.updates % observer.config.target_interval == 0:
            observer.target.load_state_dict(observer.online.state_dict())
        if step % settings.controller_check_interval:
            continue
        q, td = measure()
        action_change = float((q.argmax(1) != previous_q.argmax(1)).float().mean().cpu())
        q_change = float(((q - previous_q).square().mean().sqrt() / previous_q.square().mean().sqrt().clamp_min(0.001)).cpu())
        td_change = abs(td - previous_td) / max(abs(previous_td), 0.001)
        if not np.isfinite([td, q_change, td_change]).all():
            raise FloatingPointError('Non-finite controller stability statistic')
        unchanged = action_change <= settings.controller_action_tolerance and q_change <= settings.controller_q_tolerance and (td_change <= settings.controller_td_tolerance)
        stable = stable + 1 if unchanged else 0
        records.append({'phase': 'controller', 'interactions': interactions, 'steps': step, 'heldout_td_loss': td, 'action_change': action_change, 'relative_q_change': q_change, 'relative_td_change': td_change, 'stable_checks': stable})
        previous_q, previous_td = (q, td)
        if step >= settings.controller_min_steps and stable >= settings.controller_patience:
            reason = 'replay_stable'
            break
    return ({'steps': step, 'stop_reason': reason, 'heldout_td_loss': previous_td, 'training_samples': len(train), 'validation_samples': len(heldout)}, records)


def refresh_reward(acquisition, arrival_states, settings):
    """Update the visitation map and reward moments only between training rounds."""
    acquisition.counter.observe(arrival_states)
    if acquisition.config.intrinsic_normalization == "rms":
        decay = acquisition.config.intrinsic_rms_decay
        for start in range(0, len(arrival_states), settings.reward_batch):
            rewards = acquisition._score(arrival_states[start:start + settings.reward_batch])[0]
            for reward in rewards:
                acquisition.reward_sq = decay * acquisition.reward_sq + (1 - decay) * float(reward)**2
                acquisition.reward_samples += 1
