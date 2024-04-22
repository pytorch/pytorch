from __future__ import annotations

import functools
import logging
import os
import subprocess
import sys
import warnings
from types import ModuleType
from typing import Any, Callable

from . import standalone_compile_main
from .runtime_utils import cache_dir


standalone_compile_main_path = os.path.abspath(standalone_compile_main.__file__)
log = logging.getLogger(__name__)


def _reload_triton_kernel_in_subproc(key, path, kernel_name):
    return _module_to_triton_kernel(
        _reload_python_module_in_subproc(key, path), kernel_name
    )


def _module_to_triton_kernel(mod, kernel_name):
    kernel = getattr(mod, kernel_name)

    reload_mod = getattr(mod, "_reload_in_subproc", None)
    if reload_mod:
        assert (
            isinstance(reload_mod, functools.partial)
            and reload_mod.func is _reload_python_module_in_subproc
            and not reload_mod.keywords
        )
        key, path = reload_mod.args

        kernel._reload_in_subproc = functools.partial(
            _reload_triton_kernel_in_subproc,
            key,
            path,
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


def call_standalone_triton_compile(kernel):
    load_kernel = kernel._reload_in_subproc
    assert (
        isinstance(load_kernel, functools.partial)
        and load_kernel.func is _reload_triton_kernel_in_subproc
        and not load_kernel.keywords
    )
    key, path, kernel_name = load_kernel.args
    cmd = [
        sys.executable,
        standalone_compile_main_path,
        kernel.device_props.type,
        key,
        path,
        kernel_name,
    ]
    process = subprocess.Popen(
        cmd,
        env={
            **os.environ,
            "TORCHINDUCTOR_CACHE_DIR": cache_dir(),
        },
    )
    try:
        process.communicate(timeout=120)
        if process.returncode != 0:
            log.warning("Triton compile failed in subprocess: %s", " ".join(cmd))
    except subprocess.TimeoutExpired:
        log.warning("Triton compile timeout in subprocess: %s", " ".join(cmd))
        process.terminate()
