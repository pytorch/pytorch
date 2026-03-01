from __future__ import annotations

import functools
import linecache
import os
import sys
import time
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any, TYPE_CHECKING

from torch._utils_internal import log_triton_builds


if TYPE_CHECKING:
    from collections.abc import Callable

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
    ptxas = Path(__file__).absolute().parents[2] / "bin" / "ptxas"
    if not ptxas.exists():
        return
    if ptxas.is_file() and os.access(ptxas, os.X_OK):
        os.environ["TRITON_PTXAS_PATH"] = str(ptxas)
    else:
        warnings.warn(f"{ptxas} exists but is not an executable")


def _set_triton_libdevice_path() -> None:
    """
    Use the CUDA toolkit's libdevice instead of Triton's bundled version.
    This ensures Triton's pow matches CUDA's powf for bitwise precision.
    Gated by config.eager_numerics.use_pytorch_libdevice.
    """
    from torch._inductor import config

    if not config.eager_numerics.use_pytorch_libdevice:
        return

    _set_triton_libdevice_path_impl()


def _set_triton_libdevice_path_impl() -> None:
    try:
        from triton import knobs
    except ImportError:
        return

    env_path = os.environ.get("TRITON_LIBDEVICE_PATH")
    if env_path is not None:
        knobs.nvidia.libdevice_path = env_path
        return

    if knobs.nvidia.libdevice_path is not None:
        return

    try:
        from torch.utils.cpp_extension import CUDA_HOME

        if CUDA_HOME is None:
            warnings.warn(
                "CUDA_HOME not set; using Triton's bundled libdevice which may "
                "cause minor precision differences in pow operations. "
                "To fix: set TRITON_LIBDEVICE_PATH to your CUDA toolkit's libdevice, "
                "e.g., export TRITON_LIBDEVICE_PATH=/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
                stacklevel=3,
            )
            return
        libdevice = Path(CUDA_HOME) / "nvvm" / "libdevice" / "libdevice.10.bc"
        if libdevice.is_file():
            knobs.nvidia.libdevice_path = str(libdevice)
            # Also set env var so subprocess compile workers inherit it
            os.environ["TRITON_LIBDEVICE_PATH"] = str(libdevice)
        else:
            warnings.warn(
                f"CUDA libdevice not found at {libdevice}; using Triton's bundled "
                "libdevice which may cause minor precision differences in pow operations. "
                "To fix: set TRITON_LIBDEVICE_PATH to your CUDA toolkit's libdevice, "
                "e.g., export TRITON_LIBDEVICE_PATH=/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
                stacklevel=3,
            )
    except ImportError:
        warnings.warn(
            "torch.utils.cpp_extension not available; using Triton's bundled "
            "libdevice which may cause minor precision differences in pow operations. "
            "To fix: set TRITON_LIBDEVICE_PATH to your CUDA toolkit's libdevice, "
            "e.g., export TRITON_LIBDEVICE_PATH=/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
            stacklevel=3,
        )


def _worker_compile_triton(
    load_kernel: Callable[[], CachingAutotuner],
    extra_env: dict[str, str],
    extra_config: dict[str, Any],
) -> tuple[CachingAutotuner, int]:
    _set_triton_ptxas_path()
    os.environ.update(extra_env)
    # Set libdevice path if passed via env from main process
    libdevice_path = extra_env.get("TRITON_LIBDEVICE_PATH")
    if libdevice_path:
        try:
            from triton import knobs

            knobs.nvidia.libdevice_path = libdevice_path
        except ImportError:
            pass
    from torch._inductor import config

    with config.patch(extra_config):
        fail = None
        try:
            start_ns = time.time_ns()
            kernel = load_kernel()
            kernel.precompile(warm_cache_only=True)
            elapsed_ns = time.time_ns() - start_ns
            kernel.prepare_for_pickle()
            # We can release this memory in the compile subprocesses:
            linecache.clearcache()
            return kernel, elapsed_ns // 1000
        except Exception as e:
            fail = str(e)
            raise
        finally:
            log_triton_builds(fail=fail)
