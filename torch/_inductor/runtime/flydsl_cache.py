"""Helpers for routing FlyDSL JIT artifacts through TorchInductor's cache."""

from __future__ import annotations

import os
from pathlib import Path

from torch._inductor.runtime.cache_dir_utils import cache_dir


def _cache_dir() -> Path:
    return Path(cache_dir()) / "flydsl_compile_cache"


def ensure_flydsl_cache_dir() -> str:
    """Route FlyDSL's disk cache through TorchInductor's cache root by default.

    FlyDSL has its own disk cache controlled by ``FLYDSL_RUNTIME_CACHE_DIR``.
    Inductor-generated kernels should participate in Inductor cache cleanup and
    subprocess warming, so default FlyDSL to an Inductor-owned subdirectory --
    but respect an explicit ``FLYDSL_RUNTIME_CACHE_DIR`` the user already set.
    """
    existing = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if existing:
        return existing
    cache_dir = str(_cache_dir())
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = cache_dir
    return cache_dir
