"""Helpers for routing FlyDSL JIT artifacts through TorchInductor's cache."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from torch._inductor.runtime.cache_dir_utils import cache_dir


_compiled_cache_lock = threading.RLock()


def run_cached_flydsl(
    jit_func: Any,
    *compile_args: Any,
    constexpr_param: Any,
    compiler: Callable[..., Any],
    dispatch_args: tuple[Any, ...],
) -> Any:
    """Cache a layout-dynamic FlyDSL dispatcher by constexpr param."""
    with _compiled_cache_lock:
        cache_key = constexpr_param.__cache_signature__()
        compiled_cache = getattr(jit_func, "_compiled_cache", None)
        if compiled_cache is None:
            compiled_cache = {}
            jit_func._compiled_cache = compiled_cache

        compiled = compiled_cache.get(cache_key)
        if compiled is None:
            compiled = compiler(jit_func, *compile_args)
            compiled_cache[cache_key] = compiled
        else:
            compiled(*dispatch_args)
        return compiled


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
