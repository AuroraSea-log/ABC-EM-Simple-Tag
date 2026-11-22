from __future__ import annotations
import copy
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from .config import fingerprint
from .environment import TARGET, OBSERVER, is_done, make_env
from .models import DQN
from .opponent import FrozenOpponent, fixed_test_set, require_opponent
from .rewards import Acquisition
from .staged import episode_holdout, fit_clones, fit_controller, frozen_rewards, refresh_reward
from .utils import OutputLock, runtime_info, save_torch, seed_everything, sha256, write_frame, write_json


def boundary_cost(states, weight):
    positions = np.abs(states[:, 2:4])
    cost = np.where(positions < .9, 0., np.where(positions < 1., (positions - .9) * 10.,
                    np.exp(np.minimum(2 * positions - 2, np.log(10)))))
    return (weight * cost.sum(axis=1)).astype(np.float32)


class Collector:
    def __init__(self, config, seed, opponent):
        self.config, self.seed, self.opponent = config, seed, opponent
        self.target_rng = np.random.default_rng(50_000 + seed)
        self.action_rng = np.random.default_rng(50_000 + seed + config.observer_rng_offset)
        self.envs, self.observations = [], []
        self.episodes = np.zeros(config.num_envs, dtype=np.int32)
        self.cycles = np.zeros(config.num_envs, dtype=np.int32)
        self.interactions = 0
        self.pending = [None] * config.num_envs
        try:
            for index in range(config.num_envs):
                env = make_env(config)
                self.envs.append(env)
                self.observations.append(env.reset(seed=seed * 100_003 + 31 + index * 1_009)[0])
        except BaseException:
            self.close()
            raise

    def close(self):
        for env in self.envs:
            env.close()

    def restart(self, index):
        self.episodes[index] += 1
        self.cycles[index] = 0
        state = self.envs[index].reset(seed=int(self.seed * 100_003 + 31 + index * 1_009
                                               + self.episodes[index] * 10_007))[0]
        self.observations[index] = state

    def collect(self, observer, count):
        # The action network is fixed for this collection round; CPU avoids tiny CUDA transfers.
        policy = copy.deepcopy(observer.online).to("cpu").eval().requires_grad_(False)
        records = []
        while len(records) < count:
            batch = min(self.config.num_envs, count - len(records))
            ids = [(self.interactions + i) % self.config.num_envs for i in range(batch)]
            target_states = np.stack([self.observations[i][TARGET] for i in ids]).astype(np.float32)
            target_actions = self.opponent.sample(target_states, self.target_rng)
            middles, target_rewards, target_done = [], [], []
            for row, index in enumerate(ids):
                middle, reward_t, term_t, trunc_t = self.envs[index].step_target(int(target_actions[row]))
                middles.append(middle)
                target_rewards.append(float(reward_t[TARGET]))
                target_done.append(is_done(term_t, trunc_t))
            # B's successor is only known after the pursuer moves again.
            for row, index in enumerate(ids):
                deferred = self.pending[index]
                if deferred is None:
                    continue
                deferred["next_observer_states"] = np.asarray(middles[row][OBSERVER], dtype=np.float32).copy()
                deferred["terminal"] = bool(target_done[row])
                records.append(deferred)
                self.pending[index] = None
            live = [row for row in range(batch) if not target_done[row]]
            states = actions = None
            if live:
                states = np.stack([middles[row][OBSERVER] for row in live]).astype(np.float32)
                actions = self.action_rng.integers(5, size=len(live), dtype=np.int64)
                if self.interactions >= self.config.start_steps:
                    explore = self.action_rng.random(len(live)) < observer.epsilon(self.interactions)
                    with torch.no_grad():
                        greedy = policy(torch.as_tensor(states)).argmax(-1).numpy()
                    actions[~explore] = greedy[~explore]
            finished = []
            for slot, row in enumerate(live):
                index = ids[row]
                nxt, rewards, terminated, truncated = self.envs[index].step_observer(int(actions[slot]))
                terminal = is_done(terminated, truncated)
                record = {"target_states": target_states[row], "target_actions": target_actions[row],
                          "observer_states": states[slot], "observer_actions": actions[slot],
                          "next_observer_states": None,
                          "next_target_states": np.asarray(nxt[TARGET], dtype=np.float32).copy(),
                          "terminal": terminal, "env_id": index, "episode": self.episodes[index],
                          "cycle": self.cycles[index], "native_target_reward": target_rewards[row],
                          "native_observer_reward": float(rewards[OBSERVER])}
                self.cycles[index] += 1
                if terminal:
                    record["next_observer_states"] = np.asarray(nxt[OBSERVER], dtype=np.float32).copy()
                    records.append(record)
                    finished.append(index)
                else:
                    self.pending[index] = record
                    self.observations[index] = nxt
            finished.extend(ids[row] for row in range(batch) if target_done[row])
            for index in finished:
                self.restart(index)
            self.interactions += batch
        return {key: np.asarray([row[key] for row in records]) for key in records[0]}


def run_experiment(config, settings, root: Path, seed: int):
    settings.validate(config)
    checkpoint = require_opponent(config, root, seed)
    run = root / "runs" / f"seed_{seed}"
    run.mkdir(parents=True, exist_ok=True)
    with OutputLock(run):
        runtime = runtime_info()
        signature = fingerprint({"config": config.as_dict(), "stages": asdict(settings), "seed": seed,
                                 "opponent": sha256(checkpoint), "source": runtime["source_sha256"]})
        completed = run / "complete.json"
        if completed.exists():
            saved = json.loads(completed.read_text(encoding="utf-8"))
            if saved["signature"] != signature:
                raise ValueError("Completed run configuration changed; choose a new output directory")
            for name, digest in saved["artifacts"].items():
                if sha256(run / name) != digest:
                    raise ValueError(f"Artifact changed: {name}")
            return
        if (run / "metrics.csv").exists():
            raise ValueError("Partial run exists; preserve it and choose a new output directory")
        opponent = FrozenOpponent(checkpoint, config.opponent_device)
        test_set = fixed_test_set(config, opponent, root, seed, checkpoint)
        seed_everything(50_000 + seed, config.threads)
        acquisition = Acquisition(config)
        devices = [acquisition.clone.device.index or 0] if acquisition.clone.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(50_000 + seed + 4_000_000)
            observer = DQN(config)
        collector = Collector(config, seed, opponent)
        validation_replay = np.zeros(config.replay_capacity, dtype=bool)
        rng = np.random.default_rng(1_050_000 + seed)
        blocks, curves, inner, rounds = [], [], [], []
        started = time.perf_counter()
        clone_updates = 0

        def evaluate(interactions):
            curves.append({"seed": seed, "interactions": interactions,
                           "wall_seconds": time.perf_counter() - started,
                           "clone_rounds": acquisition.updates, "clone_gradient_steps": clone_updates,
                           "controller_gradient_steps": observer.updates,
                           "position_coverage_percent": 100 * float(np.mean(acquisition.counter.counts > 0)),
                           "joint_position_bins": len(acquisition.counter.joint_counts),
                           **acquisition.clone.evaluate(test_set)})
            write_frame(run / "metrics.csv", pd.DataFrame(curves))

        try:
            evaluate(0)
            for start in range(0, config.total_interactions, config.clone_interval):
                end = start + config.clone_interval
                begin = time.perf_counter()
                block = collector.collect(observer, config.clone_interval)
                collection_seconds = time.perf_counter() - begin
                blocks.append(block)
                data = {key: np.concatenate([part[key] for part in blocks])
                        for key in ("target_states", "target_actions", "env_id", "episode")}
                validation = episode_holdout(data["env_id"], data["episode"], seed, config.num_envs)
                begin = time.perf_counter()
                clone_summary, checks = fit_clones(acquisition, data["target_states"], data["target_actions"],
                                                   validation, settings, seed, end)
                clone_seconds = time.perf_counter() - begin
                clone_updates += clone_summary[0]["steps"]
                inner.extend(checks)
                begin = time.perf_counter()
                refresh_reward(acquisition, block["next_target_states"], settings)
                slots = (observer.ptr + np.arange(len(block["terminal"]))) % config.replay_capacity
                validation_replay[slots] = episode_holdout(block["env_id"], block["episode"], seed, config.num_envs)
                costs = boundary_cost(block["next_observer_states"], config.observer_boundary_cost)
                observer.add(block["observer_states"], block["observer_actions"], np.zeros(len(slots), np.float32),
                             block["next_observer_states"], block["terminal"], block["next_target_states"], costs)
                rewards = frozen_rewards(acquisition, observer, settings)
                controller_summary, checks = fit_controller(observer, rewards, validation_replay, settings, rng, end)
                inner.extend(checks)
                rounds.append({"interactions": end, "clone": clone_summary[0], "controller": controller_summary,
                               "collection_seconds": collection_seconds, "clone_seconds": clone_seconds,
                               "reward_and_controller_seconds": time.perf_counter() - begin})
                evaluate(end)
                write_frame(run / "inner_curves.csv", pd.DataFrame(inner))
                write_json(run / "rounds.json", {"rounds": rounds})
                save_torch(run / "latest.pt", {"interactions": end, "clone": acquisition.clone.state_dict(),
                           "controller": observer.state_dict(), "joint_counts": acquisition.counter.joint_counts,
                           "position_counts": acquisition.counter.counts, "reward_sq": acquisition.reward_sq,
                           "reward_samples": acquisition.reward_samples,
                           "config": config.as_dict(), "stages": asdict(settings)})
                print(f"ABC-EM seed={seed} interactions={end}/{config.total_interactions} "
                      f"KL={curves[-1]['kl']:.6f} controller={controller_summary['stop_reason']}", flush=True)
            np.savez_compressed(run / "trajectories.npz",
                                **{key: np.concatenate([part[key] for part in blocks]) for key in blocks[0]})
            names = ("metrics.csv", "inner_curves.csv", "rounds.json", "latest.pt", "trajectories.npz")
            write_json(completed, {"signature": signature, "seed": seed, "algorithm": "ABC-EM",
                                  "target_interactions": config.total_interactions,
                                  "opponent_sha256": sha256(checkpoint),
                                  "artifacts": {name: sha256(run / name) for name in names}})
        finally:
            collector.close()
