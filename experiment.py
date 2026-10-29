"""Shared configuration and reproducibility helpers for the learning scripts.

Loading YAML does not import PyTorch, so the survival demo stays independent
of the image/text training dependencies.
"""

import os
from pathlib import Path
import random

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(path=None):
    """Read config.yaml from the project root, or an explicitly supplied file."""
    config_path = PROJECT_ROOT / "config.yaml" if path is None else Path(path)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    seed = config.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1.")
    runtime = config.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("runtime must be a YAML mapping.")
    if type(runtime.get("deterministic")) is not bool:
        raise ValueError("runtime.deterministic must be true or false.")
    workers = runtime.get("num_workers")
    if type(workers) is not int or workers < 0:
        raise ValueError("runtime.num_workers must be a nonnegative integer.")
    if not isinstance(runtime.get("device"), str):
        raise ValueError("runtime.device must be auto, cpu, or a device such as cuda:0.")
    if not isinstance(config.get("paths"), dict):
        raise ValueError("paths must be a YAML mapping.")
    return config


def project_path(config, key):
    """Resolve a configured path relative to the project, rather than cwd."""
    path = Path(config["paths"][key]).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def seed_everything(seed, deterministic=True):
    """Seed Python, NumPy and PyTorch before creating models or random data.

    NumPy default_rng instances still need an explicit seed. Python's hash
    randomization is selected at interpreter startup, so it is not changed here.
    """
    # Set this before creating a CUDA context or executing CUDA operations.
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def get_device(config):
    """Select CUDA when available in auto mode; honor an explicit device."""
    import torch

    name = config["runtime"]["device"]
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use auto or cpu.")
    return device


def seed_worker(worker_id):
    """Initialize Python and NumPy RNGs from the seed assigned by DataLoader."""
    import numpy as np
    import torch

    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def dataloader_options(config):
    """Return independent, reproducible RNG settings for one DataLoader."""
    import torch

    workers = config["runtime"]["num_workers"]
    if os.name == "nt" and workers != 0:
        raise ValueError(
            "Keep runtime.num_workers: 0 for the Windows learning scripts. "
            "Multiple workers require an if __name__ == '__main__' guard first."
        )
    generator = torch.Generator()
    generator.manual_seed(config["seed"])
    return {
        "num_workers": workers,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }


def save_config(config, path):
    """Save the effective configuration beside an experiment's outputs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
