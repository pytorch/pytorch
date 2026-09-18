# Owner(s): ["module: inductor"]

import unittest

import torch
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import HardwareClassification, IS_LINUX
from torch.utils._triton import has_triton


def _has_multigpu():
    if torch.accelerator.is_available():
        return torch.accelerator.device_count() > 1
    return False


requires_multigpu = unittest.skipIf(
    not _has_multigpu(), "requires multiple accelerator devices"
)


def _check_fn(test_case, func, expect_cpu, *args):
    out_eager = func(*args)

    out_compiled, code = run_and_get_code(torch.compile(func), *args)
    test_case.assertEqual(out_eager, out_compiled)

    if len(code) != 1:
        raise AssertionError
    if expect_cpu:
        FileCheck().check("cpp_fused").run(code[0])
    else:
        FileCheck().check_not("cpp_fused").run(code[0])


class TestMoveConstructorsToGpu(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_simple(self, device):
        def foo(x):
            return x[torch.arange(x.shape[0])]

        inp = torch.rand(32, 77, 512, device=device)

        _check_fn(self, foo, False, inp)

    def test_output_failure(self, device):
        def foo(x):
            tmp1 = torch.arange(x.shape[0])
            return tmp1, x[tmp1]

        inp = torch.rand(32, 77, 512, device=device)

        _check_fn(self, foo, True, inp)

    def test_non_convertable_op_failure(self, device):
        def foo(x):
            y = torch.arange(x.shape[0])
            return x + y, torch.ones([4], device=device)

        inp = torch.rand([100])

        _check_fn(self, foo, True, inp)

    def test_sets_equiv(self, device):
        @torch.compile()
        def foo(x):
            c1 = torch.ones([4], dtype=torch.long)
            c2 = torch.arange(-1, 3)
            return x[c1 + c2], c2 - 4 * 2

        inp = torch.rand([4]).to(device)
        _, code = run_and_get_code(foo, inp)
        FileCheck().check_not("triton.jit").run(code[0])

        @torch.compile()
        def foo(x):
            c2 = torch.arange(-1, 3)
            c1 = torch.ones([4], dtype=torch.long)
            return x[c1 + c2], c2 - 4 * 2

        _, code = run_and_get_code(foo, inp)
        FileCheck().check_not("triton.jit").run(code[0])

    @requires_multigpu
    @unittest.skip("https://github.com/pytorch/pytorch/issues/139520")
    def test_multi_gpu(self, device):
        device_type = torch.device(device).type

        def foo(x):
            return (
                x[torch.arange(x.shape[0])],
                torch.ones([4], device=f"{device_type}:0"),
                torch.ones([4], device=f"{device_type}:1"),
            )

        # nyi, multi-gpu
        inp = torch.rand([100], device=device)
        _check_fn(self, foo, True, inp)

    def test_random_constructor_not_moved(self, device):
        from torch._inductor import config

        for random_fn in [torch.randn, torch.rand]:
            torch._dynamo.reset()

            def foo(x, fn=random_fn):
                values = fn(2, x.size(1)).to(x.device)
                indices = torch.tensor([0, 2], dtype=torch.long).to(x.device)
                return torch.index_add(x, 0, indices, values)

            inp = torch.randn(4, 8, device=device)

            with config.patch(fallback_random=True):
                torch.manual_seed(0)
                out_eager = foo(inp)

                torch.manual_seed(0)
                out_compiled = torch.compile(foo)(inp)

            self.assertEqual(out_eager, out_compiled)


class TestMoveConstructorsGeneric(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_multiple_constructors(self):
        def foo(x):
            tmp1 = torch.arange(x.shape[0])
            o1 = x[tmp1]
            tmp2 = torch.arange(x.shape[1]).view([1, x.shape[1]])
            o2 = x[tmp2]
            return o1, o2, o1 + o2

        inp = torch.rand([200, 200])
        _check_fn(self, foo, True, inp)

    def test_no_gpu(self):
        def foo(x):
            return x[torch.arange(x.shape[0])]

        inp = torch.rand([100])
        _check_fn(self, foo, True, inp)


instantiate_device_type_tests(
    TestMoveConstructorsToGpu, globals(), except_for="cpu", allow_xpu=True
)


if __name__ == "__main__":
    if IS_LINUX and has_triton():
        run_tests()
