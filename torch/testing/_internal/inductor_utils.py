# mypy: ignore-errors

import torch
import re
import unittest
import functools
from subprocess import CalledProcessError

from torch._inductor.codecache import CppCodeCache
from torch.utils._triton import has_triton
from torch.testing._internal.common_utils import (
    LazyVal,
    IS_FBCODE,
)
from torch._dynamo.backends.registry import register_backend
from torch._inductor.compile_fx import compile_fx, count_bytes_inner
from torch.testing._internal.common_utils import TestCase

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

HAS_GPU = HAS_CUDA

GPUS = ["cuda"]

HAS_MULTIGPU = any(
    getattr(torch, gpu).is_available() and getattr(torch, gpu).device_count() >= 2
    for gpu in GPUS
)

tmp_gpus = [x for x in GPUS if getattr(torch, x).is_available()]
assert len(tmp_gpus) <= 1
GPU_TYPE = "cuda" if len(tmp_gpus) == 0 else tmp_gpus.pop()
del tmp_gpus

@register_backend
def count_bytes_inductor(gm, example_inputs):
    return compile_fx(gm, example_inputs, inner_compile=count_bytes_inner)

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

skipCUDAIf = functools.partial(skipDeviceIf, device="cuda")
skipCPUIf = functools.partial(skipDeviceIf, device="cpu")
