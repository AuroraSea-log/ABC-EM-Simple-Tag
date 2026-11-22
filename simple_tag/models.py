from __future__ import annotations
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .utils import device_for

def mlp(inputs: int, hidden: int, outputs: int):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, outputs))

class Clone:

    def __init__(self, config):
        self.config = config
        self.device = device_for(config.device)
        self.model = mlp(12, config.hidden, 5).to(self.device)
        self.previous = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.clone_lr, weight_decay=config.clone_weight_decay)
        self.generator = None
        if config.isolated_clone_rng:
            self.generator = torch.Generator(device=self.device).manual_seed(torch.initial_seed())

    @torch.no_grad()
    def scores(self, states):
        tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        current = torch.distributions.Categorical(logits=self.model(tensor))
        previous = torch.distributions.Categorical(logits=self.previous(tensor))
        entropy = current.entropy().cpu().numpy()
        change = torch.distributions.kl_divergence(current, previous).clamp_min(0).cpu().numpy()
        return (entropy, change)

    @torch.no_grad()
    def probabilities(self, states):
        tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        return self.model(tensor).softmax(dim=-1).cpu().numpy()

    @torch.no_grad()
    def evaluate(self, test_set):
        states, actions, target_probs = test_set
        tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        labels = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        truth = torch.as_tensor(target_probs, dtype=torch.float32, device=self.device)
        log_probs = F.log_softmax(self.model(tensor), dim=-1)
        expected_nll = -(truth * log_probs).sum(dim=-1).mean()
        target_entropy = -(truth * truth.clamp_min(1e-30).log()).sum(dim=-1).mean()
        agreement = (log_probs.argmax(-1) == truth.argmax(-1)).float().mean()
        return {'nll': float(F.nll_loss(log_probs, labels).cpu()), 'expected_nll': float(expected_nll.cpu()), 'kl': max(0.0, float((expected_nll - target_entropy).cpu())), 'target_entropy': float(target_entropy.cpu()), 'agreement': 100.0 * float(agreement.cpu()), 'sampled_action_accuracy': 100.0 * float((log_probs.argmax(-1) == labels).float().mean().cpu())}

    def state_dict(self):
        return {'model': self.model.state_dict(), 'previous': self.previous.state_dict(), 'optimizer': self.optimizer.state_dict(), 'sampling_rng': self.generator.get_state() if self.generator else None}

def double_dqn_targets(rewards, terminals, online_next, target_next, gamma):
    best_actions = online_next.argmax(dim=1, keepdim=True)
    return rewards + gamma * (1.0 - terminals) * target_next.gather(1, best_actions).squeeze(1)

class DQN:

    def __init__(self, config):
        self.config = config
        self.device = device_for(config.device)
        self.online = mlp(10, config.hidden, 5).to(self.device)
        self.target = copy.deepcopy(self.online).eval().requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.dqn_lr)
        capacity = config.replay_capacity
        self.states = np.empty((capacity, 10), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_states = np.empty_like(self.states)
        self.terminals = np.empty(capacity, dtype=np.float32)
        self.target_next_states = np.empty((capacity, 12), dtype=np.float32)
        self.reward_costs = np.empty(capacity, dtype=np.float32)
        self.size = self.ptr = self.updates = 0

    def epsilon(self, interactions):
        fraction = min(interactions / self.config.epsilon_steps, 1.0)
        return self.config.epsilon_start + fraction * (self.config.epsilon_end - self.config.epsilon_start)

    @torch.no_grad()
    def act(self, states, interactions, rng):
        actions = rng.integers(5, size=len(states), dtype=np.int64)
        if interactions < self.config.start_steps:
            return actions
        explore = rng.random(len(states)) < self.epsilon(interactions)
        tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        greedy = self.online(tensor).argmax(dim=1).cpu().numpy()
        actions[~explore] = greedy[~explore]
        return actions

    def add(self, states, actions, rewards, next_states, terminals, target_next_states=None, reward_costs=None):
        if self.config.replay_relabel and (target_next_states is None or reward_costs is None):
            raise ValueError('Reward relabeling requires target next observations and fixed transition costs')
        for index, (state, action, reward, next_state, terminal) in enumerate(zip(states, actions, rewards, next_states, terminals)):
            self.states[self.ptr] = state
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.next_states[self.ptr] = next_state
            self.terminals[self.ptr] = terminal
            if target_next_states is not None:
                self.target_next_states[self.ptr] = target_next_states[index]
                self.reward_costs[self.ptr] = reward_costs[index]
            self.ptr = (self.ptr + 1) % len(self.actions)
            self.size = min(self.size + 1, len(self.actions))

    def replay_rewards(self, indices, reward_function=None):
        if not self.config.replay_relabel:
            return self.rewards[indices]
        if reward_function is None:
            raise ValueError('Reward relabeling requires a pure reward function')
        return reward_function(self.target_next_states[indices]) - self.reward_costs[indices]

    def update(self, rng, interactions, reward_function=None):
        if interactions < self.config.start_steps or self.size < self.config.dqn_batch:
            return None
        indices = rng.integers(self.size, size=self.config.dqn_batch)
        tensor = lambda x: torch.as_tensor(x[indices], device=self.device)
        states, next_states = (tensor(self.states), tensor(self.next_states))
        actions, terminals = (tensor(self.actions), tensor(self.terminals))
        rewards = torch.as_tensor(self.replay_rewards(indices, reward_function), dtype=torch.float32, device=self.device)
        predictions = self.online(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            targets = double_dqn_targets(rewards, terminals, self.online(next_states), self.target(next_states), self.config.gamma)
        loss = F.smooth_l1_loss(predictions, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config.target_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach().cpu())

    def state_dict(self):
        return {'online': self.online.state_dict(), 'target': self.target.state_dict(), 'optimizer': self.optimizer.state_dict(), 'updates': self.updates}
