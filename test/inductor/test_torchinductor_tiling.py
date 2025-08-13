# Owner(s): ["module: inductor"]
import os
import sys
import unittest

import sympy
import itertools

import torch
from torch._C import memory_format
# from torch._inductor.codegen.cpp import cexpr
# from torch._inductor.codegen.triton import texpr
# from torch._inductor.codegen.wrapper import pexpr
# from torch._inductor.runtime.benchmarking import benchmarker
# from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_triton_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_MACOS,
    IS_WINDOWS,
    parametrize,
)
from torch.testing import FileCheck
from torch._inductor.utils import (
    run_and_get_code
)
from torch.testing._internal.inductor_utils import requires_gpu, GPU_TYPE
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU
from torch.utils._sympy.functions import (
    FloorDiv,
    Mod,
    ModularIndexing,
    PythonMod,
    RoundDecimal,
    RoundToInt,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"

layouts = ("cont",  "NHWC", "T")

@requires_gpu()
@instantiate_parametrized_tests
class TestTiling(InductorTestCase):
    
    @parametrize("a", layouts)
    @parametrize("b", layouts)
    def test_pointwise(self, a, b):

        SIZE_A = 128
        SIZE_B = 256
        SIZE_C = 512

        def T(layout: str):
            if layout == "cont":
                return torch.rand(SIZE_A, SIZE_B, SIZE_C, device=GPU_TYPE).unsqueeze(0)
            elif layout == "T":
                return torch.rand(SIZE_A, SIZE_B, SIZE_C, device=GPU_TYPE).transpose(1, 2).contiguous().transpose(1, 2).unsqueeze(0)
            else:
                return torch.rand([1, SIZE_A, SIZE_B, SIZE_C], device=GPU_TYPE).to(memory_format=torch.channels_last)
    
        def foo(x, y):
            return x + y

        x, y = T(a), T(b)
        res, code = run_and_get_code(torch.compile(foo), x, y)    

        if a != b:
            FileCheck().check("ynumel").run(code[0])
        else:
            FileCheck().check_not("ynumel").run(code[0])

        self.assertEqual(res, foo(x, y))

    def test_reduce(self):

        @torch.compile()
        def foo(x):
            return x.sum(dim=1)

        x = torch.rand([512, 256], device=GPU_TYPE).T

        foo(x)

    def test_contg_sum(self):

        @torch.compile()
        def foo(x):
            return x.sum()

        x = torch.rand([512], device=GPU_TYPE)

        foo(x)

    def test_non_contg_sum(self):

        @torch.compile()
        def foo(x):
            return x.sum(dim=0)

        x = torch.rand([512, 512], device=GPU_TYPE)

        foo(x)

    def test_mm(self):

        def mm(A, B):
            out = (A.unsqueeze(-1) * B.unsqueeze(0)).sum(dim=1)
            return out

        compiled = torch.compile(mm)

        N = 1024
        A = torch.randn((N,N), device="cuda")
        B = torch.randn((N,N), device="cuda")

        out = compiled(A,B)


        


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests()
