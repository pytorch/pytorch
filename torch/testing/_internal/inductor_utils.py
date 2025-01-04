# mypy: ignore-errors

import logging
import torch
import re
import unittest
import functools
import os
from subprocess import CalledProcessError
import sys
import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
from typing import Callable
from torch._inductor.codecache import CppCodeCache
from torch._inductor.utils import get_gpu_shared_memory, is_big_gpu
from torch._inductor.utils import GPU_TYPES, get_gpu_type
from torch._inductor.codegen.common import (
    init_backend_registration,
    get_scheduling_for_device,
)
from torch.testing._internal.common_utils import (
    LazyVal,
    IS_FBCODE,
)
from torch.testing._internal.common_utils import (
    TestCase,
    IS_CI,
    IS_WINDOWS,
)
from torch._dynamo.device_interface import get_interface_for_device

log: logging.Logger = logging.getLogger(__name__)

def test_cpu():
    try:
        CppCodeCache.load("")
        return not IS_FBCODE
    except (
        CalledProcessError,
        OSError,
        torch._inductor.exc.InvalidCxxCompiler,
        torch._inductor.exc.CppCompileError,
    ):
        return False

HAS_CPU = LazyVal(test_cpu)


def has_inductor_available(device_type: str) -> bool:
    # Ensure the scheduling backends are registered first. Note that this is
    # decorated with `lru_cache`, so safe to call multiple times.
    init_backend_registration()

    try:
        scheduling_factory = get_scheduling_for_device(device_type)
        if scheduling_factory is None:
            return False
        scheduling_factory(None).raise_if_unavailable(device_type)
        return True
    except RuntimeError:
        return False


def has_triton_backend_available(device_type: str) -> bool:
    try:
        di = get_interface_for_device(device_type)
        di.raise_if_triton_unavailable(None)
        return True
    except RuntimeError:
        return False


def _has_triton() -> bool:
    try:
        import triton.runtime

        return triton.runtime.driver.active.is_active()
    except (RuntimeError, ImportError):
        return False


HAS_TRITON = _has_triton()

# Triton for CPU is available.
HAS_CPU_TRITON = LazyVal(lambda: HAS_TRITON and has_triton_backend_available("cpu"))

# We have a CUDA device and a compatible Inductor backend.
HAS_CUDA = LazyVal(lambda: torch.cuda.is_available() and has_inductor_available("cuda"))
# We have a CUDA device and the CUDA Triton backend for Inductor is available.
HAS_CUDA_TRITON = LazyVal(lambda: HAS_CUDA and has_triton_backend_available("cuda"))

# We have an XPU device and a compatible Inductor backend.
HAS_XPU = LazyVal(lambda: torch.xpu.is_available() and has_inductor_available("xpu"))
# We have an XPU device and the XPU Triton backend for Inductor is available.
HAS_XPU_TRITON = LazyVal(lambda: HAS_XPU and has_triton_backend_available("xpu"))

HAS_GPU = LazyVal(lambda: HAS_CUDA or HAS_XPU)
HAS_GPU_TRITON = LazyVal(lambda: HAS_CUDA_TRITON or HAS_XPU_TRITON)

GPU_TYPE = get_gpu_type()

HAS_MULTIGPU = any(
    getattr(torch, gpu).is_available() and getattr(torch, gpu).device_count() >= 2
    for gpu in GPU_TYPES
)

def _check_has_dynamic_shape(
    self: TestCase,
    code,
):
    for_loop_found = False
    has_dynamic = False
    lines = code.split("\n")
    for line in lines:
        if "for(" in line:
            for_loop_found = True
            if re.search(r";.*ks.*;", line) is not None:
                has_dynamic = True
                break
    self.assertTrue(
        has_dynamic, msg=f"Failed to find dynamic for loop variable\n{code}"
    )
    self.assertTrue(for_loop_found, f"Failed to find for loop\n{code}")


def skipDeviceIf(cond, msg, *, device):
    if cond:
        def decorate_fn(fn):
            @functools.wraps(fn)
            def inner(self, *args, **kwargs):
                if not hasattr(self, "device"):
                    warn_msg = (
                        "Expect the test class to have attribute device but not found. "
                    )
                    if hasattr(self, "device_type"):
                        warn_msg += "Consider using the skip device decorators in common_device_type.py"
                    log.warning(warn_msg)
                if self.device == device:
                    raise unittest.SkipTest(msg)
                return fn(self, *args, **kwargs)

            return inner
    else:
        def decorate_fn(fn):
            return fn

    return decorate_fn

def skip_windows_ci(name: str, file: str) -> None:
    if IS_WINDOWS and IS_CI:
        module = os.path.basename(file).strip(".py")
        sys.stderr.write(
            f"Windows CI does not have necessary dependencies for {module} tests yet\n"
        )
        if name == "__main__":
            sys.exit(0)
        raise unittest.SkipTest("requires sympy/functorch/filelock")

def _skip_lazily_if_decorator(cb: Callable[[], bool], msg: str):
    def decorator(fn_or_cls):
        if isinstance(fn_or_cls, type):
            if cb():
                fn_or_cls.__unittest_skip__ = True
                fn_or_cls.__unittest_skip_why__ = msg

            return fn_or_cls

        @functools.wraps(fn_or_cls)
        def wrapper(*args, **kwargs):
            if cb():
                raise unittest.SkipTest(msg)

            return fn_or_cls(*args, **kwargs)

        return wrapper

    def decorator_wrapper(fn_or_cls=None):
        if callable(fn_or_cls):
            return decorator(fn_or_cls)

        return decorator

    return decorator_wrapper


requires_cuda = _skip_lazily_if_decorator(lambda: not HAS_CUDA, "requires CUDA device with Inductor support")
requires_gpu = _skip_lazily_if_decorator(lambda: not HAS_GPU, "requires GPU with Inductor support")
requires_triton = _skip_lazily_if_decorator(lambda: not HAS_TRITON, "requires Triton")
requires_gpu_triton = _skip_lazily_if_decorator(lambda: not HAS_GPU_TRITON, "requires GPU and Triton")

skipCUDAIf = functools.partial(skipDeviceIf, device="cuda")
skipXPUIf = functools.partial(skipDeviceIf, device="xpu")
skipCPUIf = functools.partial(skipDeviceIf, device="cpu")

IS_A100 = LazyVal(lambda: HAS_CUDA and get_gpu_shared_memory() == 166912)

IS_H100 = LazyVal(lambda: HAS_CUDA and get_gpu_shared_memory() == 232448)

IS_BIG_GPU = LazyVal(lambda: HAS_CUDA and is_big_gpu())
