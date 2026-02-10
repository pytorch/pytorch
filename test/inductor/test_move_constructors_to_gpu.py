# Owner(s): ["module: inductor"]

import functools
import unittest

import torch
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU_AND_TRITON,
    HAS_MULTIGPU,
)


requires_multigpu = functools.partial(
    unittest.skipIf, not HAS_MULTIGPU, f"requires multiple {GPU_TYPE} devices"
)


aten = torch.ops.aten


class TestMoveConstructorsToGpu(TestCase):
    def _check_fn(self, func, expect_cpu, *args):
        out_eager = func(*args)

        out_compiled, code = run_and_get_code(torch.compile(func), *args)
        self.assertEqual(out_eager, out_compiled)

        if len(code) != 1:
            raise AssertionError
        if expect_cpu:
            FileCheck().check("cpp_fused").run(code[0])
        else:
            FileCheck().check_not("cpp_fused").run(code[0])

    def test_simple(self):
        def foo(x):
            return x[torch.arange(x.shape[0])]

        inp = torch.rand(32, 77, 512, device=GPU_TYPE)

        self._check_fn(foo, False, inp)

    def test_output_failure(self):
        def foo(x):
            tmp1 = torch.arange(x.shape[0])
            return tmp1, x[tmp1]

        inp = torch.rand(32, 77, 512, device=GPU_TYPE)

        self._check_fn(foo, True, inp)

    def test_non_convertable_op_failure(self):
        def foo(x):
            y = torch.arange(x.shape[0])
            return x + y, torch.ones([4], device=GPU_TYPE)

        inp = torch.rand([100])

        self._check_fn(foo, True, inp)

    def test_multiple_constructors(self):
        def foo(x):
            tmp1 = torch.arange(x.shape[0])
            o1 = x[tmp1]
            tmp2 = torch.arange(x.shape[1]).view([1, x.shape[1]])
            o2 = x[tmp2]
            return o1, o2, o1 + o2

        inp = torch.rand([200, 200])
        self._check_fn(foo, True, inp)

    def test_sets_equiv(self):
        @torch.compile()
        def foo(x):
            c1 = torch.ones([4], dtype=torch.long)
            c2 = torch.arange(-1, 3)
            return x[c1 + c2], c2 - 4 * 2

        inp = torch.rand([4]).to(GPU_TYPE)
        _, code = run_and_get_code(foo, inp)
        FileCheck().check_not("triton.jit").run(code[0])

        @torch.compile()
        def foo(x):
            c2 = torch.arange(-1, 3)
            c1 = torch.ones([4], dtype=torch.long)
            return x[c1 + c2], c2 - 4 * 2

        _, code = run_and_get_code(foo, inp)
        FileCheck().check_not("triton.jit").run(code[0])

    @requires_multigpu()
    @unittest.skip("https://github.com/pytorch/pytorch/issues/139520")
    def test_multi_gpu(self):
        def foo(x):
            return (
                x[torch.arange(x.shape[0])],
                torch.ones([4], device=f"{GPU_TYPE}:0"),
                torch.ones([4], device=f"{GPU_TYPE}:1"),
            )

        # nyi, multi-gpu
        inp = torch.rand([100], device=GPU_TYPE)
        self._check_fn(foo, True, inp)

    def test_no_gpu(self):
        def foo(x):
            return x[torch.arange(x.shape[0])]

        inp = torch.rand([100])
        self._check_fn(foo, True, inp)


if __name__ == "__main__":
    if IS_LINUX and HAS_GPU_AND_TRITON:
        run_tests()
