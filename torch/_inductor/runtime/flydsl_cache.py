"""Helpers for routing FlyDSL JIT artifacts through TorchInductor's cache."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, TYPE_CHECKING

from torch._inductor.runtime.cache_dir_utils import cache_dir


if TYPE_CHECKING:
    from collections.abc import Callable


_compiled_cache_lock = threading.Lock()


def run_cached_flydsl(
    jit_func: Any,
    *compile_args: Any,
    constexpr_param: Any,
    compiler: Callable[..., Any],
    dispatch_args: tuple[Any, ...],
) -> Any:
    """Cache a layout-dynamic FlyDSL dispatcher by constexpr param."""
    cache_key = constexpr_param.__cache_signature__()
    with _compiled_cache_lock:
        compiled_cache = getattr(jit_func, "_compiled_cache", None)
        if compiled_cache is None:
            compiled_cache = {}
            jit_func._compiled_cache = compiled_cache
        compile_locks = getattr(jit_func, "_compiled_cache_locks", None)
        if compile_locks is None:
            compile_locks = {}
            jit_func._compiled_cache_locks = compile_locks
        compile_lock = compile_locks.setdefault(cache_key, threading.Lock())

    with compile_lock:
        compiled = compiled_cache.get(cache_key)
        if compiled is None:
            # FlyDSL compilation executes the first invocation using compile_args.
            compiled = compiler(jit_func, *compile_args)
            compiled_cache[cache_key] = compiled
            return compiled

    # Keep steady-state kernel launches outside all compilation locks.
    compiled(*dispatch_args)
    return compiled


def _cache_dir() -> Path:
    return Path(cache_dir()) / "flydsl_compile_cache"


def ensure_flydsl_cache_dir() -> str:
    """Put FlyDSL artifacts under the Inductor cache unless explicitly set."""
    existing = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if existing:
        return existing
    resolved = str(_cache_dir())
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = resolved
    return resolved
