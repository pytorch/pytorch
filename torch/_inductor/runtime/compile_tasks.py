from __future__ import annotations

import functools
import os
import sys
import warnings
from types import ModuleType
from typing import Any, Callable


def _reload_triton_kernel_in_subproc(reload_module, kernel_name):
    return _module_to_triton_kernel(reload_module(), kernel_name)


def _module_to_triton_kernel(mod, kernel_name):
    kernel = getattr(mod, kernel_name)
    kernel._reload_in_subproc = functools.partial(
        _reload_triton_kernel_in_subproc,
        mod._reload_in_subproc,
        kernel_name,
    )
    return kernel


def _reload_python_module_in_subproc(key, path):
    codecache = sys.modules.get("torch._inductor.codecache")
    if codecache:
        return codecache.PyCodeCache.load_by_key_path(key, path)
    else:
        return _reload_python_module(key, path)


def _reload_python_module(key, path):
    with open(path) as f:
        try:
            code = compile(f.read(), path, "exec")
        except Exception as e:
            raise RuntimeError(
                f"Failed to import {path}\n{type(e).__name__}: {e}"
            ) from None
        mod = ModuleType(f"{__name__}.{key}")
        mod.__file__ = path
        mod.key = key  # type: ignore[attr-defined]
        exec(code, mod.__dict__, mod.__dict__)
        sys.modules[mod.__name__] = mod
        return mod


@functools.lru_cache(None)
def _set_triton_ptxas_path() -> None:
    if os.environ.get("TRITON_PTXAS_PATH") is not None:
        return
    ptxas_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "bin", "ptxas")
    )
    if not os.path.exists(ptxas_path):
        return
    if os.path.isfile(ptxas_path) and os.access(ptxas_path, os.X_OK):
        os.environ["TRITON_PTXAS_PATH"] = ptxas_path
    else:
        warnings.warn(f"{ptxas_path} exists but is not an executable")


def _worker_compile_triton(
    load_kernel: Callable[[], Any],
):
    _set_triton_ptxas_path()
    load_kernel().precompile(warm_cache_only=True)
