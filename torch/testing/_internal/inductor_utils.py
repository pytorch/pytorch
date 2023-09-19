from subprocess import CalledProcessError

from torch._inductor.codecache import CppCodeCache
from torch._inductor.utils import has_triton
from torch.testing._internal.common_utils import (
    LazyVal,
    IS_FBCODE,
)
from torch._dynamo.backends.registry import register_backend
from torch._inductor.compile_fx import compile_fx, count_bytes_inner
from torch.testing._internal.common_utils import TestCase
import functools
import re
import torch
import unittest

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

HAS_CUDA = has_triton()

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

HAS_MULTIGPU = HAS_CUDA and torch.cuda.device_count() >= 2

requires_multigpu = functools.partial(
    unittest.skipIf, not HAS_MULTIGPU, "requires multiple cuda devices"
)
