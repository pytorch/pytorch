"""Helpers for routing FlyDSL JIT artifacts through TorchInductor's cache."""

from __future__ import annotations

import os
from pathlib import Path


def _cache_dir() -> Path:
    cache_root = os.environ.get(
        "TORCHINDUCTOR_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "torch_inductor"),
    )
    return Path(cache_root) / "flydsl_compile_cache"


def ensure_flydsl_cache_dir() -> str:
    """Ensure FlyDSL uses TorchInductor's cache root by default.

    FlyDSL has its own disk cache controlled by ``FLYDSL_RUNTIME_CACHE_DIR``.
    Inductor-generated kernels should participate in Inductor cache cleanup and
    subprocess warming, while still honoring a user-provided FlyDSL cache dir.
    """
    cache_dir = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if not cache_dir:
        cache_dir = str(_cache_dir())
        os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = cache_dir
    return cache_dir
