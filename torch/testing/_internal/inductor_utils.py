# mypy: ignore-errors

import torch
import re
import unittest
import functools
import os
from subprocess import CalledProcessError
import sys
import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
from torch._inductor.codecache import CppCodeCache
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

HAS_CUDA = torch.cuda.is_available() and has_triton()

HAS_XPU = torch.xpu.is_available() and has_triton()

HAS_GPU = HAS_CUDA or HAS_XPU

GPUS = ["cuda", "xpu"]

HAS_MULTIGPU = any(
    getattr(torch, gpu).is_available() and getattr(torch, gpu).device_count() >= 2
    for gpu in GPUS
)

tmp_gpus = [x for x in GPUS if getattr(torch, x).is_available()]
assert len(tmp_gpus) <= 1
GPU_TYPE = "cuda" if len(tmp_gpus) == 0 else tmp_gpus.pop()
del tmp_gpus

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
            def inner(self, *args, **kwargs):
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

skipCUDAIf = functools.partial(skipDeviceIf, device="cuda")
skipXPUIf = functools.partial(skipDeviceIf, device="xpu")
skipCPUIf = functools.partial(skipDeviceIf, device="cpu")
