from __future__ import annotations

from typing import Any, Union

import torch


try:
    import triton
except ImportError:
    triton = None


if triton is not None:
    import triton.language as tl
    from triton import Config
    from triton.compiler import CompiledKernel
    from triton.runtime.autotuner import OutOfResources
    from triton.runtime.jit import KernelInterface

    try:
        from triton.runtime.autotuner import PTXASError
    except ImportError:

        class PTXASError(Exception):  # type: ignore[no-redef]
            pass

    try:
        from triton.compiler.compiler import ASTSource
    except ImportError:
        ASTSource = None

    try:
        from triton.backends.compiler import GPUTarget
    except ImportError:

        def GPUTarget(
            backend: str,
            arch: Union[int, str],
            warp_size: int,
        ) -> Any:
            if torch.version.hip:
                return [backend, arch, warp_size]
            return (backend, arch)

    # In the latest triton, math functions were shuffled around into different modules:
    # https://github.com/openai/triton/pull/3172
    try:
        from triton.language.extra import libdevice

        libdevice = tl.extra.libdevice  # noqa: F811
        math = tl.math
    except ImportError:
        if hasattr(tl.extra, "cuda") and hasattr(tl.extra.cuda, "libdevice"):
            libdevice = tl.extra.cuda.libdevice
            math = tl.math
        elif hasattr(tl.extra, "intel") and hasattr(tl.extra.intel, "libdevice"):
            libdevice = tl.extra.intel.libdevice
            math = tl.math
        else:
            libdevice = tl.math
            math = tl

    try:
        from triton.language.standard import _log2
    except ImportError:

        def _log2(x: Any) -> Any:
            raise NotImplementedError

else:

    def _raise_error(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("triton package is not installed")

    class OutOfResources(Exception):  # type: ignore[no-redef]
        pass

    class PTXASError(Exception):  # type: ignore[no-redef]
        pass

    Config = object
    CompiledKernel = object
    KernelInterface = object
    ASTSource = None
    GPUTarget = None
    _log2 = _raise_error
    libdevice = None
    math = None

    class triton:  # type: ignore[no-redef]
        @staticmethod
        def jit(*args: Any, **kwargs: Any) -> Any:
            return _raise_error

    class tl:  # type: ignore[no-redef]
        @staticmethod
        def constexpr(val: Any) -> Any:
            return val

        tensor = Any
        dtype = Any


def cc_warp_size(cc: Union[str, int]) -> int:
    if torch.version.hip:
        cc_str = str(cc)
        if "gfx10" in cc_str or "gfx11" in cc_str:
            return 32
        else:
            return 64
    else:
        return 32


try:
    autograd_profiler = torch.autograd.profiler
except AttributeError:  # Compile workers only have a mock version of torch

    class autograd_profiler:  # type: ignore[no-redef]
        _is_profiler_enabled = False


__all__ = [
    "Config",
    "CompiledKernel",
    "OutOfResources",
    "KernelInterface",
    "PTXASError",
    "ASTSource",
    "GPUTarget",
    "tl",
    "_log2",
    "libdevice",
    "math",
    "triton",
    "cc_warp_size",
]
