"""Helpers for routing FlyDSL JIT artifacts through TorchInductor's cache."""

from __future__ import annotations

import ctypes
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
    """Cache a FlyDSL dispatcher on its JIT function by constexpr param."""
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


def make_flydsl_inductor_launcher(
    executor: Any,
    output: Any,
    mat1: Any,
    mat2: Any,
    *,
    m: int,
    n: int,
    k: int,
    param: Any,
) -> Callable[[Any, Any, Any, int], Any]:
    """Specialize FlyDSL's packed C ABI for a static Inductor GEMM.

    FlyDSL's generic ``CompiledFunction`` dispatch updates every runtime slot
    through Python fill callbacks. Inductor has a fixed ABI here, so initialize
    M/N/K/param once and leave only three tensor pointers plus the stream on the
    hot path. The private FlyDSL state is validated completely; incompatible
    versions use the ordinary executor through a four-argument adapter.
    """

    def fallback(out: Any, lhs: Any, rhs: Any, stream: int) -> Any:
        return executor(out, lhs, rhs, m, n, k, param, stream)

    if os.environ.get("TORCHINDUCTOR_USE_FAST_FLYDSL_LAUNCHER", "1") == "0":
        return fallback

    debug_launcher = (
        os.environ.get("TORCHINDUCTOR_FLYDSL_LAUNCHER_DEBUG", "0") == "1"
    )
    try:
        state = executor._call_state
        slot_specs = tuple(state._spec)
        func_exe = state._func_exe

        storages = []
        packed = (ctypes.c_void_p * len(slot_specs))()
        dynamic_storages: dict[int, list[Any]] = {0: [], 1: [], 2: [], 7: []}
        static_values = {3: m, 4: n, 5: k, 6: param}

        for slot_index, (arg_index, ctype, fill) in enumerate(slot_specs):
            try:
                storage = ctype(0)
            except TypeError:
                storage = ctype()
            storages.append(storage)
            packed[slot_index] = ctypes.addressof(storage)

            if arg_index in dynamic_storages:
                if ctype is not ctypes.c_void_p or fill is None:
                    return fallback
                dynamic_storages[arg_index].append(storage)
            elif arg_index in static_values:
                if fill is None:
                    return fallback
                fill(static_values[arg_index], storage)
            else:
                return fallback

        if any(len(dynamic_storages[index]) != 1 for index in (0, 1, 2, 7)):
            return fallback

        func_ptr = ctypes.cast(func_exe, ctypes.c_void_p).value
        if not func_ptr:
            return fallback

        # The generated host stub does not call Python. Keeping the GIL avoids
        # CFUNCTYPE's release/reacquire cost and protects the reusable slots.
        invoke = ctypes.PYFUNCTYPE(None, ctypes.c_void_p)(func_ptr)
        packed_ptr = ctypes.cast(packed, ctypes.c_void_p)
        out_storage = dynamic_storages[0][0]
        lhs_storage = dynamic_storages[1][0]
        rhs_storage = dynamic_storages[2][0]
        stream_storage = dynamic_storages[7][0]
        out_data_ptr = type(output).data_ptr
        lhs_data_ptr = type(mat1).data_ptr
        rhs_data_ptr = type(mat2).data_ptr

        def launch(out: Any, lhs: Any, rhs: Any, stream: int) -> None:
            out_storage.value = out_data_ptr(out)
            lhs_storage.value = lhs_data_ptr(lhs)
            rhs_storage.value = rhs_data_ptr(rhs)
            stream_storage.value = stream
            invoke(packed_ptr)

        # Keep the ExecutionEngine and every address referenced by ``packed``
        # alive without adding work to each launch.
        launch._flydsl_keepalive = (  # type: ignore[attr-defined]
            executor,
            storages,
            packed,
            invoke,
        )
        if debug_launcher:
            print("[flydsl] launcher=python-packed", flush=True)
        return launch
    except Exception:
        return fallback
