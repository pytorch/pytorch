"""Helpers for routing FlyDSL JIT artifacts through TorchInductor's cache."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, TYPE_CHECKING

from torch._inductor.runtime.cache_dir_utils import cache_dir


if TYPE_CHECKING:
    from collections.abc import Callable


# Serialize cold compiles process-wide; warm cache hits bypass this lock.
_compiled_cache_lock = threading.Lock()


def run_cached_flydsl(
    jit_func: Any,
    *compile_args: Any,
    constexpr_param: Any,
    compiler: Callable[..., Any],
    dispatch_args: tuple[Any, ...],
    compile_args_factory: Callable[[], tuple[Any, ...]] | None = None,
) -> Any:
    """Cache a layout-dynamic FlyDSL dispatcher by constexpr param.

    `compile_args` are only read on the compile path -- a cache hit dispatches
    straight through `dispatch_args`. Building them eagerly therefore wastes
    work on every call after the first for a given param: the FlyDSL argument
    descriptors (`from_torch_tensor(...).mark_layout_dynamic()`) are
    constructed and immediately discarded. Callers that can defer the work
    should pass `compile_args_factory` instead, a zero-argument callable
    invoked only when a compile is actually needed.
    """
    if compile_args_factory is not None and compile_args:
        raise TypeError("pass compile_args or compile_args_factory, not both")
    device = getattr(dispatch_args[0], "device", None)
    cache_key = (
        os.getpid(),
        getattr(device, "index", None),
        constexpr_param.__cache_signature__(),
    )
    compiled_cache = getattr(jit_func, "_compiled_cache", None)
    if compiled_cache is not None:
        compiled = compiled_cache.get(cache_key)
        if compiled is not None:
            compiled(*dispatch_args)
            return compiled

    dispatch_after_wait = False
    with _compiled_cache_lock:
        compiled_cache = getattr(jit_func, "_compiled_cache", None)
        if compiled_cache is None:
            compiled_cache = {}
            jit_func._compiled_cache = compiled_cache

        compiled = compiled_cache.get(cache_key)
        if compiled is None:
            # compile() executes this invocation; dispatching again would double-run.
            args = (
                compile_args_factory()
                if compile_args_factory is not None
                else compile_args
            )
            compiled = compiler(jit_func, *args)
            compiled_cache[cache_key] = compiled
        else:
            dispatch_after_wait = True

    if dispatch_after_wait:
        compiled(*dispatch_args)
    return compiled


def _cache_dir() -> Path:
    return Path(cache_dir()) / "flydsl_compile_cache"


def configure_flydsl_cache_dir() -> str:
    """Route FlyDSL's disk cache through TorchInductor's cache root by default.

    FlyDSL has its own disk cache controlled by ``FLYDSL_RUNTIME_CACHE_DIR``.
    Inductor-generated kernels should participate in Inductor cache cleanup and
    subprocess warming, so default FlyDSL to an Inductor-owned subdirectory --
    but respect an explicit ``FLYDSL_RUNTIME_CACHE_DIR`` the user already set.
    """
    existing = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if existing is not None:
        return existing
    resolved = str(_cache_dir())
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = resolved
    return resolved
