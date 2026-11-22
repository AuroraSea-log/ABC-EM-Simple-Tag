from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys

import numpy as np
import torch


def seed_everything(seed: int, threads: int = 1) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def device_for(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    result = torch.device(requested)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this PyTorch installation has no available GPU")
    return result


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_frame(path, frame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def save_torch(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def runtime_info() -> dict:
    versions = {}
    for package in ("torch", "numpy", "gymnasium", "mpe2", "stable-baselines3", "pandas", "scipy", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    root = Path(__file__).resolve().parents[1]
    source_files = [*root.glob("*.py"), *root.glob("simple_tag/*.py")]
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "command": sys.argv,
        "source_sha256": {p.relative_to(root).as_posix(): sha256(p) for p in sorted(source_files)},
    }


class OutputLock:
    """Protect a shared result directory; release the OS lock even after a crash."""

    def __init__(self, root: Path):
        self.path = root / ".run.lock"

    def __enter__(self):
        self.stream = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                self.stream.seek(0)
                if not self.stream.read(1):
                    self.stream.write(b"0")
                    self.stream.flush()
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            raise RuntimeError(f"Another process is using {self.path.parent}") from exc
        return self

    def __exit__(self, *args):
        self.stream.close()
