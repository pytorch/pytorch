# Owner(s): ["module: inductor"]
import contextlib
import copy
import functools
import importlib
import itertools
import os
import sys
import unittest
import weakref

import torch

import torch._dynamo as torchdynamo
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch import nn
from torch._inductor import config
from torch._inductor.compile_fx import compile_fx
from torch._inductor.utils import override_lowering, run_and_get_code
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer import X86InductorQuantizer
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import SM80OrLater
from torch.testing._internal.common_quantization import (
    skipIfNoDynamoSupport,
    skipIfNoONEDNN,
)

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

from torch.testing._internal.common_utils import (
    IS_CI,
    IS_WINDOWS,
    skipIfRocm,
    TEST_WITH_ASAN,
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
prims = torch.ops.prims
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
    def __init__(self, in_channels, out_channels, bias=False, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=bias, **kwargs)
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
        # CPU path will replace mm with mkl._linear,
        # skip this case for now.
        if self.device == "cpu":
            raise unittest.SkipTest("NYI CPU")

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
        for use_bias, dtype in itertools.product(
            [True, False], [torch.float16, torch.bfloat16, torch.float32]
        ):
            if self.device == "cpu" and dtype == torch.float16:
                continue

            if self.device == "cuda" and dtype == torch.bfloat16 and not SM80OrLater:
                continue

            mod = (
                ConvBN(3, 32, bias=use_bias, kernel_size=3, stride=2)
                .eval()
                .to(self.device)
                .to(dtype)
            )

            x = torch.rand(3, 3, 32, 32).to(self.device).to(dtype)

            @torch.compile()
            def foo(mod, x):
                return mod(x)

            # TODO - bias is separate kernel right now, we should only unfuse it
            # from conv if it can be fused

            with torch.no_grad():
                out_eager = mod(x)
                out_optimized_for_infernece, code = run_and_get_code(foo, mod, x)

            self.assertNotIn(
                "aten._native_batch_norm_legit_no_training(",
                code[0],
            )

            # we unfuse the conv bias, but it should only have one constant in the kernel
            if self.device == "cuda":
                FileCheck().check_not(".run(").check("conv").check(".run(").check_same(
                    "constant"
                ).check_not("constant").check_next("return").run(code[0])

            self.assertEqual(
                out_optimized_for_infernece, out_eager, atol=1e-2, rtol=1e-2
            )

    def test_dont_change_dtype_folding(self):
        dtype = torch.float16 if self.device == "cuda" else torch.bfloat16

        mod = (
            torch.nn.Conv2d(3, 32, bias=None, kernel_size=3, stride=2)
            .eval()
            .to(self.device)
            .to(dtype)
        )
        x = torch.rand(3, 3, 32, 32).to(self.device).to(dtype)

        def foo(mod, x):
            return mod(x) * torch.full([1], 2.0, device=self.device)

        foo_c = torch.compile(foo)

        with torch.no_grad():
            out_eager = foo(mod, x)
            out_compiled = foo_c(mod, x)
            self.assertEqual(out_eager, out_compiled)

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

    def test_conv_with_as_strided(self):
        class Model(nn.Module):
            def __init__(self, groups):
                super().__init__()
                self.kv = torch.nn.Conv2d(
                    256,
                    384,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    bias=False,
                    groups=groups,
                )

            def forward(self, x):
                convolution = self.kv(x)
                constant_pad_nd = torch.ops.aten.constant_pad_nd.default(
                    convolution, [2, 2, 2, 2], 0.0
                )
                # as_strided inputs are depend on input's size and stide.
                as_strided = torch.ops.aten.as_strided.default(
                    constant_pad_nd, [8, 384, 2, 20, 12], [153600, 400, 160, 1, 20]
                )
                as_strided_1 = torch.ops.aten.as_strided.default(
                    as_strided, [8, 384, 2, 2, 12, 12], [153600, 400, 160, 8, 20, 1]
                )
                clone = torch.ops.aten.clone.default(
                    as_strided_1, memory_format=torch.contiguous_format
                )
                return clone

        @torch.compile()
        def foo(mod, inp):
            return mod(inp)

        with torch.no_grad():
            x = torch.randn(8, 256, 16, 16).to(self.device)
            for groups in [1, 2]:
                mod = Model(groups).to(self.device).eval()
                mod_eager = mod(x)
                self.assertEqual(foo(mod, x), mod_eager)

    @skipIfRocm
    def test_cpp_wrapper(self):
        mod = ConvBN(3, 32, kernel_size=3, stride=2).eval().to(self.device)

        x = torch.rand(3, 3, 32, 32).to(self.device)

        @torch.compile(options={"cpp_wrapper": True})
        def foo(mod, x):
            return mod(x)

        out_eager = mod(x)

        with torch.no_grad():
            self.assertEqual(foo(mod, x), out_eager)
            self.assertEqual(foo(mod, x), out_eager)

    def test_conv_layout_convert_with_view(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(
                    3, 128, kernel_size=3, padding=1, stride=1, bias=False
                )
                self.bn = nn.BatchNorm2d(3)

            def forward(self, x):
                x = self.bn(x)
                x = self.conv(x)
                return torch.flatten(x, 1)

        mod = Model().to(self.device).eval()

        @torch.compile()
        def foo(mod, inp):
            return mod(inp)

        with torch.no_grad():
            x = torch.rand(2, 3, 5, 5).to(self.device)
            mod_eager = mod(x)
            self.assertEqual(foo(mod, x), mod_eager)

    def test_conv_weight_layout_convert(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(
                    3, 128, kernel_size=3, padding=1, stride=1, bias=False
                )

            def forward(self, x):
                return self.conv(x)

            @staticmethod
            def get_example_inputs():
                return (torch.rand(2, 3, 5, 5).to(self.device),)

        from torch._inductor.compile_fx import compile_fx, compile_fx_inner

        nconv = 0

        def my_inner_compile(gm, example_inputs, *args, **kwargs):
            out = compile_fx_inner(gm, example_inputs, *args, **kwargs)

            nonlocal nconv
            convs = [n for n in gm.graph.nodes if n.target == aten.convolution.default]
            nconv += len(convs)
            for conv in convs:
                weight_node = conv.args[1]
                weight_const_tensor = getattr(gm, weight_node.target)
                self.assertTrue(
                    weight_const_tensor.is_contiguous(memory_format=torch.channels_last)
                )
                self.assertTrue(
                    weight_node.meta["val"].is_contiguous(
                        memory_format=torch.channels_last
                    )
                )

            return out

        mod = torch.compile(
            Model().eval().to(self.device),
            backend=functools.partial(compile_fx, inner_compile=my_inner_compile),
        )
        inp = mod.get_example_inputs()
        with torch.no_grad():
            mod(*inp)

        # Only check the assertion for CUDA.
        # For CPU, we may get torch.ops.mkldnn._convolution_pointwise.default
        # in the joint graph rather than torch.ops.aten.convolution.default.
        # Currently we only handle aten.convolution.default in layout
        # optimization. That's why the count may be 0 here for CPU.
        if self.device == "cuda":
            self.assertTrue(nconv == 1)

    def test_redundant_clone_for_layout_convert(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(
                    3, 128, kernel_size=3, padding=1, stride=1, bias=False
                )

            def forward(self, x):
                y = x + 1
                return self.conv(x), y

            @staticmethod
            def get_example_inputs():
                return (torch.rand(2, 3, 5, 5).to(self.device),)

        mod = Model().eval().to(self.device)
        inp = mod.get_example_inputs()
        with torch.no_grad():
            expected_outputs = mod(*inp)

        num_same_stride = 0
        num_diff_stride = 0

        def debug_inductor_force_stride_order(orig_fn, input_tensor, stride):
            nonlocal num_same_stride, num_diff_stride
            input_tensor.realize()
            if tuple(input_tensor.get_stride()) == tuple(stride):
                num_same_stride += 1
            else:
                num_diff_stride += 1
            return orig_fn(input_tensor, stride)

        with override_lowering(
            prims.inductor_force_stride_order.default, debug_inductor_force_stride_order
        ):
            opt_mod = torch.compile(mod)
            with torch.no_grad():
                actual_outputs = opt_mod(*inp)

        self.assertEqual(len(actual_outputs), len(expected_outputs))
        self.assertEqual(2, len(actual_outputs))
        for i, actual, expected in zip(
            itertools.count(), actual_outputs, expected_outputs
        ):
            self.assertTrue(
                torch.allclose(expected, actual, atol=1e-4, rtol=1e-4),
                f"{i}th output: expected {expected}, actual {actual}",
            )

        if self.device == "cpu":
            # CPU use different convolution implementation, skip the checks below
            return

        self.assertTrue(
            actual_outputs[0].is_contiguous(memory_format=torch.contiguous_format)
        )
        self.assertTrue(
            actual_outputs[1].is_contiguous(memory_format=torch.contiguous_format)
        )

        # we don't change the stride of y returned by forward. So there will
        # be no extra copy
        self.assertTrue(num_same_stride == 1, f"num_same_stride is {num_same_stride}")
        # we changed the stride of self.conv(x) returned by forward. So there
        # may be an extra copy
        self.assertTrue(num_diff_stride == 1, f"num_diff_stride is {num_diff_stride}")


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


@skipIfNoDynamoSupport
class OptimizeForInferenceQuantizationPT2E(TestCase):
    @skipIfNoONEDNN
    def test_functional_constant_folding_after_dynamo_export(self):
        m = ConvBN(3, 3, kernel_size=3, stride=2).eval().to("cpu")
        example_inputs = (torch.randn(1, 3, 9, 9).to("cpu"),)
        export_model, guards = torchdynamo.export(
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )

        quantizer = X86InductorQuantizer()
        operator_config = xiq.get_default_x86_inductor_quantization_config()
        quantizer.set_global(operator_config)
        with torch.no_grad(), config.patch({"implicit_fallbacks": True}):
            # TODO(leslie) Remove implicit_fallbacks=True after we enable the int8 fusion of
            # int8_weight -> dequant_per_channel -> convolution
            self.assertTrue(torch._inductor.config.freezing)

            prepare_model = prepare_pt2e(export_model, quantizer)
            prepare_model(*example_inputs)

            convert_model = convert_pt2e(prepare_model)
            convert_model.eval()
            compiler_model = compile_fx(convert_model, example_inputs)

            # First Run
            _ = compiler_model(*example_inputs)
            # Second Run
            _ = compiler_model(*example_inputs)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if HAS_CPU or HAS_CUDA:
        run_tests(needs="filelock")
