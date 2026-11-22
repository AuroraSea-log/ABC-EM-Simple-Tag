"""Train the Simple Tag pursuer and run ABC-EM."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from simple_tag.config import Config
from simple_tag.experiment import run_experiment
from simple_tag.opponent import pretrain
from simple_tag.staged import StageSettings
from simple_tag.utils import OutputLock, runtime_info, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/default.json")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--phase", choices=("all", "pretrain", "train"), default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--smoke", action="store_true", help="Short functional check with an untrained-size pursuer")
    args = parser.parse_args()
    if any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct nonnegative integers")
    values = json.loads(args.config.read_text(encoding="utf-8"))
    if set(values) - {"config", "stages"}:
        parser.error("Configuration must contain only config and stages")
    config = Config(**values.get("config", {}))
    settings = StageSettings(**values.get("stages", {}))
    if args.device:
        config = replace(config, device=args.device)
    if args.smoke:
        config = replace(config, num_envs=2, max_cycles=10, total_interactions=2000, clone_capacity=2000,
                         hidden=16, replay_capacity=512, dqn_batch=32, target_interval=4, test_size=32,
                         pretrain_steps=64, pretrain_envs=2, pretrain_rollout=16, pretrain_batch=16,
                         pretrain_epochs=1, pretrain_eval_interval=32, pretrain_eval_episodes=2)
        settings = replace(settings, clone_min_steps=4, clone_max_steps=12, clone_check_interval=4,
                           clone_patience=2, controller_min_steps=4, controller_max_steps=12,
                           controller_check_interval=4, controller_patience=2, probe_size=32)
    config.validate()
    settings.validate(config)
    args.output.mkdir(parents=True, exist_ok=True)
    with OutputLock(args.output):
        declaration = {"algorithm": "ABC-EM", "config": config.as_dict(), "stages": asdict(settings),
                       "smoke": args.smoke}
        path = args.output / "config.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != declaration:
            parser.error("Output directory contains a different configuration")
        write_json(path, declaration)
        write_json(args.output / "runtime.json", runtime_info())
        for seed in args.seeds:
            if args.phase in ("all", "pretrain"):
                pretrain(config, args.output, seed)
            if args.phase in ("all", "train"):
                run_experiment(config, settings, args.output, seed)


if __name__ == "__main__":
    main()
