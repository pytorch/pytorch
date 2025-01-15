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
from torch._inductor.codecache import CppCodeCache
from torch._inductor.utils import get_gpu_shared_memory, is_big_gpu
from torch._inductor.utils import GPU_TYPES, get_gpu_type
from torch.utils._triton import has_triton
from torch.testing._internal.common_utils import (
    LazyVal,
    IS_FBCODE,
)
from torch.testing._internal.common_utils import (
    TestCase,
    IS_CI,
    IS_WINDOWS,
)

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

HAS_TRITON = has_triton()

if HAS_TRITON:
    import triton
    TRITON_HAS_CPU = "cpu" in triton.backends.backends
else:
    TRITON_HAS_CPU = False


HAS_CUDA = torch.cuda.is_available() and HAS_TRITON

HAS_XPU = torch.xpu.is_available() and HAS_TRITON

HAS_GPU = HAS_CUDA or HAS_XPU

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
                    warn_msg = "Expect the test class to have attribute device but not found. "
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

requires_gpu = functools.partial(unittest.skipIf, not HAS_GPU, "requires gpu")
requires_triton = functools.partial(unittest.skipIf, not HAS_TRITON, "requires triton")

skipCUDAIf = functools.partial(skipDeviceIf, device="cuda")
skipXPUIf = functools.partial(skipDeviceIf, device="xpu")
skipCPUIf = functools.partial(skipDeviceIf, device="cpu")

IS_A100 = LazyVal(
    lambda: HAS_CUDA
    and get_gpu_shared_memory() == 166912
)

IS_H100 = LazyVal(
    lambda: HAS_CUDA
    and get_gpu_shared_memory() == 232448
)

IS_BIG_GPU = LazyVal(lambda: HAS_CUDA and is_big_gpu())
