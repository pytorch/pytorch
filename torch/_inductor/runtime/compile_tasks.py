from __future__ import annotations

import functools
import linecache
import os
import sys
import time
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, TYPE_CHECKING


if TYPE_CHECKING:
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner


def _reload_python_module(
    key: str, path: str, set_sys_modules: bool = True
) -> ModuleType:
    with open(path) as f:
        try:
            code = compile(f.read(), path, "exec", dont_inherit=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import {path}\n{type(e).__name__}: {e}"
            ) from None
        mod = ModuleType(f"{__name__}.{key}")
        mod.__file__ = path
        mod.key = key  # type: ignore[attr-defined]
        exec(code, mod.__dict__, mod.__dict__)
        if set_sys_modules:
            sys.modules[mod.__name__] = mod
        return mod


@functools.cache
def _set_triton_ptxas_path() -> None:
    if os.environ.get("TRITON_PTXAS_PATH") is not None:
        return
    ptxas = Path(__file__).absolute().parents[1] / "bin" / "ptxas"
    if not ptxas.exists():
        return
    if ptxas.is_file() and os.access(ptxas, os.X_OK):
        os.environ["TRITON_PTXAS_PATH"] = str(ptxas)
    else:
        warnings.warn(f"{ptxas} exists but is not an executable")


def _worker_compile_triton(
    load_kernel: Callable[[], CachingAutotuner],
    extra_env: dict[str, str],
    extra_config: dict[str, Any],
) -> tuple[CachingAutotuner, int]:
    _set_triton_ptxas_path()
    os.environ.update(extra_env)
    from torch._inductor import config

    with config.patch(extra_config):
        start_ns = time.time_ns()
        kernel = load_kernel()
        kernel.precompile(warm_cache_only=True)
        elapsed_ns = time.time_ns() - start_ns
        kernel.prepare_for_pickle()
        # We can release this memory in the compile subprocesses:
        linecache.clearcache()
        return kernel, elapsed_ns // 1000
