# Owner(s): ["module: inductor"]
import contextlib
import functools
import importlib
import os
import sys
import unittest
import weakref

import torch

import torch._dynamo
from torch._inductor import config
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

from torch.testing._internal.common_utils import (
    IS_CI,
    IS_WINDOWS,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
    TestCase as TorchTestCase,
)

if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

from inductor.test_torchinductor import check_model, check_model_cuda, copy_tests

importlib.import_module("functorch")
importlib.import_module("filelock")

from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA

HAS_MULTIGPU = HAS_CUDA and torch.cuda.device_count() >= 2
aten = torch.ops.aten
requires_cuda = functools.partial(unittest.skipIf, not HAS_CUDA, "requires cuda")


class TestCase(TorchTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(
            config.patch(
                {
                    "debug": True,
                    "cpp.min_chunk_size": 1,
                    "triton.autotune_pointwise": False,  # too slow
                    "implicit_fallbacks": False,
                    "freezing": True,
                    "freezing_discard_parameters": True,
                }
            )
        )

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()
        super().tearDownClass()

    def setUp(self):
        torch._dynamo.reset()
        super().setUp()

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()


class ConvBN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001, dtype=torch.float)

    def forward(self, x):
        return self.bn(self.conv(x))


class OptimizeForInferenceTemplate(TestCase):
    def test_mutation(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mutated_param = torch.nn.Parameter(torch.zeros([10, 10]))

            def forward(self):
                self.mutated_param.add_(10)
                return self.mutated_param

        with torch.no_grad():
            mod = Mod().to(self.device)
            out_eager = mod()
            out_eager2 = mod()

            mod = Mod().to(self.device)

            @torch.compile
            def foo(mod):
                return mod()

            out_comp = foo(mod)
            out_comp2 = foo(mod)

            self.assertEqual(out_eager, out_comp)
            self.assertEqual(out_eager2, out_comp2)

    def test_aliased_param_return(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.aliased_param = torch.nn.Parameter(torch.zeros([10, 10]))

            def forward(self):
                return self.aliased_param[1:], self.aliased_param

        mod = Mod().to(self.device).eval()

        @torch.compile()
        def foo(mod):
            return mod()

        with torch.no_grad():
            mod_eager = mod()
            self.assertEqual(foo(mod), mod_eager)

    def test_autocast(self):
        if self.device == "cpu":
            raise unittest.SkipTest("MLKDNN Bug")

        mod = torch.nn.Linear(10, 10).to(self.device).eval()
        inp = torch.rand([10, 10]).to(self.device).to(torch.half)

        @torch.compile()
        def foo(mod, inp):
            return mod(inp)

        with torch.no_grad():
            with self.autocast():
                out_eager = mod(inp)
                out_compiled, code = run_and_get_code(foo, mod, inp)

                FileCheck().check_not("@triton.jit").run(code[0])
                self.assertEqual(out_eager, out_compiled)

    def test_mm_concat(self):
        class MM(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.t1 = torch.nn.Parameter(torch.rand(10, 10))
                self.t2 = torch.nn.Parameter(torch.rand(10, 10))
                self.t3 = torch.nn.Parameter(torch.rand(10, 10))

            def forward(self, x):
                return x @ self.t1, x @ self.t2, x @ self.t3

        class AddMM(MM):
            def __init__(self):
                super().__init__()

                self.b1 = torch.nn.Parameter(torch.rand([10]))
                self.b2 = torch.nn.Parameter(torch.rand([10]))
                self.b3 = torch.nn.Parameter(torch.rand([10]))

            def forward(self, x):
                return [
                    aten.addmm(b, x, p)
                    for b, p in [
                        (self.b1, self.t1),
                        (self.b2, self.t2),
                        (self.b3, self.t3),
                    ]
                ]

        for mod in [MM().to(self.device), AddMM().to(self.device)][1:]:
            inp = torch.rand([10, 10]).to(self.device)

            @torch.compile()
            def foo(mod, inp):
                return mod(inp)

            kernel_invoke = "kernel_cpp_0" if self.device == "cpu" else "triton.jit"

            with torch.no_grad():
                out_eager = mod(inp)
                out, code = run_and_get_code(foo, mod, inp)
                FileCheck().check_not(kernel_invoke).check_count(
                    "mm(", count=1, exactly=True
                ).run(code[0])
                self.assertEqual(out_eager, out)

    def test_error_on_eager(self):
        mod = ConvBN(3, 32, kernel_size=3, stride=2).eval().to(self.device)

        x = torch.rand(3, 3, 32, 32).to(self.device)

        @torch.compile()
        def foo(mod, x):
            return mod(x)

        with torch.no_grad():
            foo(mod, x)

        with self.assertRaisesRegex(
            RuntimeError, "Trying to Run Pytorch Eager Module After Dynamo Freezing"
        ):
            mod(x)

    def test_rng_op(self):
        @torch.compile()
        def foo():
            return torch.rand([4, 4], device=self.device) + 1

        with torch.no_grad():
            o1 = foo()
            o2 = foo()
            self.assertNotEqual(o1, o2)

    def test_symint_not_folded(self):
        def fn(a):
            return a.cos(), torch.zeros(a.shape[0], a.shape[1])

        fn_opt = torch._dynamo.optimize("inductor", dynamic=True)(fn)
        inp = torch.randn(2, 4, 6).to(self.device)
        torch._dynamo.mark_dynamic(inp, 0)
        torch._dynamo.mark_dynamic(inp, 1)

        with torch.no_grad():
            self.assertEqual(fn(inp), fn_opt(inp))
            inp2 = torch.randn(3, 5, 6).to(self.device)
            torch._dynamo.mark_dynamic(inp2, 0)
            torch._dynamo.mark_dynamic(inp2, 1)
            self.assertEqual(fn(inp2), fn_opt(inp2))

    def test_unfolded_bn(self):
        x = torch.rand([3, 32, 15, 15]).to(self.device)

        mod = torch.nn.BatchNorm2d(32, eps=0.001).eval().to(self.device)

        @torch.compile()
        def foo(mod, x):
            return mod(x) + 10

        out_compiled_no_inference = foo(mod, x)

        # would error if not decomposed
        with torch.no_grad():
            out_compiled = foo(mod, x)

            self.assertEqual(out_compiled_no_inference, out_compiled)

    def test_folded_conv_bn(self):
        mod = ConvBN(3, 32, kernel_size=3, stride=2).eval().to(self.device)
        x = torch.rand(3, 3, 32, 32).to(self.device)

        @torch.compile()
        def foo(mod, x):
            return mod(x)

        # TODO - bias is separate kernel right now, we should only unfuse it
        # from conv if it can be fused

        with torch.no_grad():
            out_eager = mod(x)
            out_optimized_for_infernece, code = run_and_get_code(foo, mod, x)

        FileCheck().check_not("native_batch_norm_legit_no_training").run(code[0])

        self.assertEqual(out_optimized_for_infernece, out_eager)

    def test_param_deallocated(self):
        # TODO: cpu path keeps an extra copy of graph around somewhere,
        # memory not as important for cpu
        if self.device == "cpu":
            raise unittest.SkipTest("NYI CPU")

        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.zeros([10, 10]))

            def forward(self, x):
                return (self.param + 10) + x

        mod = Mod().eval().to(self.device)
        inp = torch.rand([10], device=self.device)

        with torch.no_grad():
            eager = mod(inp)

        weight_ref = weakref.ref(mod.param)

        @torch.compile()
        def foo(mod, inp):
            return mod(inp)

        with torch.no_grad():
            compiled = foo(mod, inp)

        self.assertEqual(eager, compiled)
        self.assertTrue(weight_ref() is None)


if HAS_CPU and not torch.backends.mps.is_available():

    class FreezingCpuTests(TestCase):
        common = check_model
        device = "cpu"
        autocast = torch.cpu.amp.autocast

    copy_tests(OptimizeForInferenceTemplate, FreezingCpuTests, "cpu")

if HAS_CUDA and not TEST_WITH_ASAN:

    class FreezingCudaTests(TestCase):
        common = check_model_cuda
        device = "cuda"
        autocast = torch.cuda.amp.autocast

    copy_tests(OptimizeForInferenceTemplate, FreezingCudaTests, "cuda")


del OptimizeForInferenceTemplate

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if (HAS_CPU or HAS_CUDA) and not TEST_WITH_ROCM:
        run_tests(needs="filelock")
