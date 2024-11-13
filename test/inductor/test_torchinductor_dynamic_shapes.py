# Owner(s): ["module: inductor"]
import contextlib
import importlib
import math
import operator
import os
import sys
import unittest
from functools import partial
from typing import List, Tuple

import torch
import torch.library
from torch._dynamo.testing import CompileCounterWithBackend, make_test_cls_with_patches
from torch._inductor import metrics
from torch._inductor.codegen.common import device_codegens, register_backend_for_device
from torch._inductor.codegen.cpp import CppScheduling
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import V
from torch.testing import FileCheck
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCPU,
    onlyOn,
)
from torch.testing._internal.common_utils import (
    IS_ARM64,
    IS_FBCODE,
    parametrize,
    TEST_CUDA_MEM_LEAK_CHECK,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from inductor.test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
    check_model,
    check_model_gpu,
    CommonTemplate,
    copy_tests,
    TestFailure,
)


importlib.import_module("filelock")

# xfail by default, set is_skip=True to skip
test_failures = {
    "test_kwargs_dynamic_shapes": TestFailure(("cpu",)),
    # calling div on only symint args
    "test_AllenaiLongformerBase_repro_dynamic_shapes": TestFailure(
        ("cpu", "cuda", "xpu")
    ),
    "test_conv_inference_heuristics_dynamic_shapes": TestFailure(("cuda", "xpu")),
}

if TEST_WITH_ROCM:
    # Tensor-likes are not close
    test_failures["test_dynamic_stride_nobreak"] = TestFailure(
        ("cpu", "cuda"), is_skip=True
    )
    test_failures["test_item_to_inputs_kernel_nobreak"] = TestFailure(
        ("cpu", "cuda"), is_skip=True
    )
    test_failures["test_unbacked_reduction"] = TestFailure(("cpu"), is_skip=True)


if os.getenv("BUILD_ENVIRONMENT", "").endswith("-debug"):
    # Fails with TORCH_INTERNAL_ASSERT(!is_heap_allocated()), see https://github.com/pytorch/pytorch/issues/130073
    test_failures["test_resize_as_dynamic_shapes"] = TestFailure(("cpu", "cuda"))
    test_failures["test_resize_dynamic_shapes"] = TestFailure(("cpu", "cuda"))


def make_dynamic_cls(cls, xfail_prop="_expected_failure_dynamic"):
    return make_test_cls_with_patches(
        cls,
        "DynamicShapes",
        "_dynamic_shapes",
        (torch._dynamo.config, "assume_static_by_default", False),
        xfail_prop=xfail_prop,
    )


DynamicShapesCommonTemplate = make_dynamic_cls(CommonTemplate)


if HAS_CPU:

    class DynamicShapesCpuTests(TestCase):
        common = check_model
        device = "cpu"

    copy_tests(DynamicShapesCommonTemplate, DynamicShapesCpuTests, "cpu", test_failures)


if HAS_GPU and not TEST_WITH_ASAN:

    class DynamicShapesGPUTests(TestCase):
        common = check_model_gpu
        device = GPU_TYPE

    copy_tests(
        DynamicShapesCommonTemplate, DynamicShapesGPUTests, GPU_TYPE, test_failures
    )


class TestInductorDynamic(TestCase):
    compile_fn = partial(torch.compile, dynamic=True)

    def setUp(self):
        # HAS_CUDA also checks compute capability to skip tests
        # on older devices
        if not HAS_GPU:
            self.skipTest("Triton not available")
        torch._dynamo.reset()
        TestCase.setUp(self)
        # this should be in setUpClass, but device-generic tests
        # don't work with setUpClass well (non-deterministically the wrong setUpClass is resolved),
        # so put it in test setUp, it's cheap
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(
            torch._inductor.config.patch(
                {
                    "debug": False,
                    "cpp.min_chunk_size": 1,
                    "triton.autotune_pointwise": False,  # too slow
                    "implicit_fallbacks": False,
                }
            )
        )

    def tearDown(self):
        self._stack.close()
        TestCase.tearDown(self)
        torch._dynamo.reset()

    def test_constant_fold_uniform_value_dynamic(self, device):
        def full_add_zero(x):
            a = torch.full(x.shape, 1, dtype=x.dtype, device=x.device)
            b = a - 1
            return x + b

        def full_mul_one(x):
            a = torch.full(x.shape, -1, dtype=x.dtype, device=x.device)
            b = 2 + a
            return x * b

        def full_view_op(x):
            a = torch.ones([1], dtype=x.dtype, device=x.device)
            a = a[:, None]
            return x * a

        def full_mul_symint(x):
            a = torch.full(x.shape, -1, dtype=x.dtype, device=x.device)
            b = 2 + a
            return b * x.shape[0]

        fns = (full_add_zero, full_mul_one, full_view_op)

        x = torch.randn((2, 4), device=device)
        y = torch.randn((3, 4), device=device)

        for dynamic in [False, True]:
            torch._dynamo.reset()
            for fn in fns:
                ref = fn(x)
                fn_c = torch.compile(fn, dynamic=dynamic)

                actual, source_codes = run_and_get_code(fn_c, x)

                if fn is not full_mul_symint:
                    # due to constant folding, fn returns x directly.
                    if device == "cpu":
                        FileCheck().check_not("cpp_fused").run(source_codes[0])
                    else:
                        FileCheck().check_not("triton.jit").run(source_codes[0])

                self.assertEqual(ref, actual)
                self.assertEqual(fn(x), fn_c(x))
                self.assertEqual(fn(y), fn_c(y))

    def test_arange_dynamic(self, device):
        def fn(a):
            batch_size = a.numel()
            max_len = a.max()
            return ~(
                torch.arange(0, max_len, device=a.device)
                .type_as(a)
                .repeat(batch_size, 1)
                .lt(a.unsqueeze(1))
            )

        a = torch.randint(10, 30, (10,), device=device)
        a[0] = 29  # fix max_len
        opt = self.compile_fn(fn)
        res = opt(a)
        ref = fn(a)
        self.assertEqual(res, ref)

    def test_shape_as_constant_reciprocal_float_exp(self, device):
        def fn(x, a):
            return x, -1 / a**1.0

        x = torch.rand(10, 20, device=device)
        opt = self.compile_fn(fn)
        res = opt(x, x.size(0))
        ref = fn(x, x.size(0))
        self.assertEqual(res, ref)

    # not supported yet on cpu, https://github.com/pytorch/pytorch/issues/109897
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_bool_mask_nobreak(self, device):
        def f(x, b):
            return (x[b] * 2).sum()

        opt_f = torch.compile(f, fullgraph=True)
        x = torch.randn(5, device=device)
        b = torch.tensor([True, True, False, False, True], device=device)
        r = f(x, b)
        opt_r = opt_f(x, b)
        self.assertEqual(r, opt_r)

    def test_adaptive_max_pool3d_with_indices(self, device):
        x = 5
        y = torch.rand([9, 10, 9, 8, 6], dtype=torch.float32, device=device)

        def fn(x, y):
            return torch.nn.functional.adaptive_max_pool3d_with_indices(
                output_size=x, input=y, return_indices=True
            )

        opt_f = self.compile_fn(fn)
        r = fn(x, y)
        opt_r = opt_f(x, y)
        self.assertEqual(r, opt_r)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unwrap_storage_didnt_work_repro(self, device):
        def f():
            full = torch.full((), 11)
            i0 = full.item()
            torch._check_is_size(i0)
            return torch.full((i0,), 0)

        opt_f = torch.compile(f, fullgraph=True)
        r = f()
        opt_r = opt_f()
        self.assertEqual(r, opt_r)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_sym_sum_unbacked(self, device):
        def f(a):
            xs = a.tolist()
            y = sum(xs)
            return torch.tensor(y)

        splits = torch.randint(10, (100,), device=device)

        opt_f = torch.compile(f, fullgraph=True)
        r = f(splits)
        opt_r = opt_f(splits)
        self.assertEqual(r, opt_r)

    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_nonzero_size_factory_nobreak(self, device):
        def f(x, b):
            y = torch.nonzero(b)
            return x.new_zeros(y.size(0))

        opt_f = torch.compile(f, fullgraph=True)
        x = torch.randn(5, device=device)
        b = torch.tensor([True, True, False, False, True], device=device)
        r = f(x, b)
        opt_r = opt_f(x, b)
        self.assertEqual(r, opt_r)

    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_nonzero_no_realloc(self, device):
        @torch.compile(fullgraph=True, dynamic=True)
        def f(x, y):
            z = x.nonzero()
            return torch.split(z, [y.size(0)])

        f(torch.tensor([1, 0, 1, 1, 0, 1, 0]), torch.randn(4))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_item_nobreak(self, device):
        @torch.compile(fullgraph=True)
        def f(x):
            y = x.item()
            return torch.empty(y)

        f(torch.tensor([3], device=device))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_item_bool_nobreak(self, device):
        @torch.compile(fullgraph=True)
        def f(x):
            return x.item()

        f(torch.tensor([True], device=device))

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_noops_tensor_repropagate(self, device):
        @torch.compile(fullgraph=True)
        def f(x):
            b = torch.ops.prims.convert_element_type.default(x, torch.int64)
            r = b.nonzero()
            return r * 2

        f(torch.tensor([0, 4, 2, 0, 1], dtype=torch.int64, device=device))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_item_zeros_nobreak(self, device):
        @torch.compile(fullgraph=True)
        def f(x):
            y = x.item()
            torch.empty(y)
            # This will avoid a NopSchedulerNode
            return x.new_zeros(y)

        f(torch.tensor([3], device=device))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_item_return(self, device):
        @torch.compile(fullgraph=True)
        def f(x):
            y = x.item()
            z = x.item()
            return y + z

        f(torch.tensor([3], device=device))

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_float_item_inf(self, device):
        @torch.compile(fullgraph=True)
        def f(x):
            return x.item() == math.inf

        f(torch.tensor([3.0], device=device))

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_float_item_neginf(self, device):
        @torch.compile(fullgraph=True)
        def f(x):
            return x.item() == -math.inf

        f(torch.tensor([3.0], device=device))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    @torch._inductor.config.patch(implicit_fallbacks=True)
    def test_item_to_inputs_kernel_nobreak(self, device):
        @torch.library.custom_op("test::foo", mutates_args=())
        def foo(x: torch.Tensor, y: int) -> torch.Tensor:
            return x.clone()

        @foo.register_fake
        def _(x: torch.Tensor, y: int) -> torch.Tensor:
            return x.clone()

        @torch.compile(fullgraph=True)
        def f(x, r):
            y = x.item()
            return torch.ops.test.foo(r, y)

        f(torch.tensor([3], device=device), torch.randn(10, device=device))

    @unittest.skipUnless(IS_FBCODE, "")
    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_float_item_return(self, device):
        @torch.compile(fullgraph=True)
        def f(x):
            return x.item()

        f(torch.tensor([3.0], device=device))

    @unittest.skipIf(TEST_CUDA_MEM_LEAK_CHECK, "failing memory leak check")
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_index_select(self, device):
        # Tests if unbacked symbols captured by inner_fn are properly tracked
        def f(x):
            y = x.item()
            return torch.index_select(
                torch.ones(y, device=device), 0, torch.tensor([0, 2, 1], device=device)
            )

        cf = torch.compile(fullgraph=True)(f)
        arg = torch.tensor(5, device=device)
        self.assertEqual(f(arg), cf(arg))

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_return_unbacked_view_split(self, device):
        def f(values, length_per_key):
            u0, u1 = length_per_key.tolist()
            torch._check_is_size(u0)
            torch._check_is_size(u1)
            v1, v2 = torch.functional.split(values, [u0, u1])
            return v1, v2

        cf = torch.compile(fullgraph=True)(f)
        args = (
            torch.randn(8, requires_grad=True, device=device),
            torch.tensor([3, 5], device=device),
        )
        self.assertEqual(f(*args), cf(*args))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_matmul(self, device):
        def f(x):
            y = x.item()
            return torch.ones(1, y, device=device) @ torch.ones(y, 1, device=device)

        cf = torch.compile(fullgraph=True)(f)
        arg = torch.tensor(5, device=device)
        self.assertEqual(f(arg), cf(arg))

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    @torch._inductor.config.patch(implicit_fallbacks=True)
    def test_unbacked_save_for_backwards(self, device) -> None:
        @torch.library.custom_op("_test::_cat", mutates_args=())
        def _cat(t: torch.Tensor, ds: List[int]) -> torch.Tensor:
            return t * t.new_ones([sum(ds)])

        @torch.library.register_fake("_test::_cat")
        def _cat_fake(t: torch.Tensor, ds: List[int]) -> torch.Tensor:
            [torch._check_is_size(d) for d in ds]
            return t.new_empty([sum(ds)])

        def _cat_setup_context(ctx, inputs, output):
            pass

        def _cat_backward(ctx, grad):
            return grad.sum(), None

        torch.library.register_autograd(
            "_test::_cat",
            _cat_backward,
            setup_context=_cat_setup_context,
        )

        def fn(t, sizes):
            r = torch.ops._test._cat(t, sizes.tolist())
            return r * t

        t = torch.randn((), requires_grad=True, device=device)
        sizes = torch.tensor([4, 8], dtype=torch.int64, device="cpu")
        out = fn(t, sizes)
        out.sum().backward()
        expect = t.grad
        t.grad = None
        torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)(
            t, sizes
        ).sum().backward()
        self.assertEqual(t.grad, expect)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_reduction(self, device):
        expect_fail = device == "cpu" and not IS_ARM64
        try:

            def f(x):
                y = x.item()
                return torch.ones(y, device=device).sum()

            cf = torch.compile(fullgraph=True)(f)
            arg = torch.tensor(5, device=device)
            self.assertEqual(f(arg), cf(arg))
        except Exception:
            if not expect_fail:
                raise
        else:
            if expect_fail:
                self.fail("expected to fail, but actually passed")

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_cat_unbacked_duplicate_size(self, device):
        def f(x):
            device = x.device
            s, s2 = x.tolist()
            g = torch.zeros(s, device=device)
            g2 = torch.ones(s2, device=device)
            return torch.ops.aten.cat.default([g, g, g2])

        cf = torch.compile(fullgraph=True)(f)
        arg = torch.tensor([4, 6], device=GPU_TYPE)
        self.assertEqual(f(arg), cf(arg))

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_unbacked_cat_backwards(self, device):
        def f(x, w):
            device = w.device
            a, b = x.tolist()
            ta = torch.ones(a, device=device)
            tb = torch.ones(b, device=device)
            pa = ta * w  # make it require gradients
            pb = tb * w
            r = torch.cat([pa, pb])
            return r.sum()

        x = torch.tensor([4, 9])
        w = torch.randn(1, requires_grad=True)
        f(x, w).backward()
        orig_w = w.grad
        w.grad = None

        torch.compile(fullgraph=True)(f)(x, w).backward()
        self.assertEqual(orig_w, w.grad)

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_unbacked_cat_backwards_save_data_dependent(self, device):
        def f(x, w):
            device = w.device
            a, b = x.tolist()
            ta = torch.ones(a, device=device)
            tb = torch.ones(b, device=device)
            pa = ta * w  # make it require gradients
            pb = tb * w
            r = torch.cat([pa, pb])
            return r

        x = torch.tensor([4, 9])
        w = torch.randn(1, requires_grad=True)
        f(x, w).sum().backward()
        orig_w = w.grad
        w.grad = None

        torch.compile(fullgraph=True)(f)(x, w).sum().backward()
        self.assertEqual(orig_w, w.grad)

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    @torch._inductor.config.patch(implicit_fallbacks=True)
    def test_dynamic_stride_nobreak(self, device):
        @torch.library.custom_op("test::foo", mutates_args=())
        def foo(x: torch.Tensor) -> torch.Tensor:
            stride = x.item()
            return torch.empty_strided((1,), (stride,), device=x.device)

        @foo.register_fake
        def _(x: torch.Tensor) -> torch.Tensor:
            ctx = torch.library.get_ctx()
            stride = ctx.new_dynamic_size()
            return torch.empty_strided((1,), (stride,), device=x.device)

        @torch.compile(fullgraph=True)
        def f(x):
            r = torch.ops.test.foo(x)
            y = r.stride(0)
            return torch.empty(y, device=x.device)

        f(torch.tensor([3], device=device))

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    @torch._inductor.config.patch(implicit_fallbacks=True)
    def test_multi_output_unbacked_custom_op(self, device):
        @torch.library.custom_op("test::foo", mutates_args=())
        def foo(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            return torch.empty(2, device=x.device), torch.empty(3, device=x.device)

        @foo.register_fake
        def _(x: torch.Tensor) -> torch.Tensor:
            ctx = torch.library.get_ctx()
            u0 = ctx.new_dynamic_size()
            return torch.empty(u0, device=x.device), torch.empty(3, device=x.device)

        @torch.compile(fullgraph=True)
        def f(x):
            a, b = torch.ops.test.foo(x)
            return a.sum() + b.sum()

        f(torch.tensor([3], device=device))

    @torch._inductor.config.patch(disable_cpp_codegen=True)
    def test_floor(self):
        # `int(n * 0.2)` will be generated as `floor(0.2*s0)` of torch.SymInt type.
        # If cpp codegen is disabled, we should generate `math.floor` using PythonPrinter.
        def fn(x):
            n = x.size(-1)
            y = x + int(n * 0.2) + 1
            return y

        opt = self.compile_fn(fn)
        # The first run doesn't trigger dynamic shapes.
        x0 = torch.rand(5)
        ref0 = fn(x0)
        res0 = opt(x0)
        self.assertEqual(ref0, res0)
        # The second run triggers dynamic shapes.
        x1 = torch.rand(8)
        ref1 = fn(x1)
        res1 = opt(x1)
        self.assertEqual(ref1, res1)

    @onlyOn(GPU_TYPE)
    def test_pad_dynamic(self, device):
        def get_same_padding(x: int, k: int, s: int, d: int):
            return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)

        def pad_same(x, k, s, d=(1, 1), value=0):
            ih, iw = x.size()[-2:]
            pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(
                iw, k[1], s[1], d[1]
            )
            if pad_h > 0 or pad_w > 0:
                x = torch.nn.functional.pad(
                    x,
                    [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
                    value=value,
                )
            return x

        x = torch.randn(2, 24, 110, 110, device=device)
        opt = self.compile_fn(pad_same)
        res = opt(x, (5, 5), (2, 2))
        ref = pad_same(x, (5, 5), (2, 2))
        self.assertEqual(res, ref, atol=0, rtol=0)

    def test_slice_scatter(self, device):
        def fn(i):
            s3 = i.size(0)
            x = torch.ones(64, s3, device=device)
            y = torch.ones(64, s3 // 2, device=device)
            return torch.slice_scatter(x, y, 1, s3 // 2, 2 * (s3 // 2))

        a = torch.randn(16, device=device)
        cfn = self.compile_fn(fn)
        expect = fn(a)
        actual = cfn(a)
        self.assertEqual(expect, actual)

    def test_slice_index_changing_sign(self, device):
        def fn(x, y):
            y0, y1 = y.shape
            return x[: (y0 - y1)].clone()

        a = torch.randn(32, 32, device=device)
        cfn = self.compile_fn(fn)

        # y0 > y1 -> y0 - y1 is positive
        b = torch.randn(16, 2, device=device)
        expect = fn(a, b)
        actual = cfn(a, b)
        self.assertEqual(expect, actual)

        # y0 < y1 -> y0 - y1 is negative
        b = torch.randn(2, 16, device=device)
        expect = fn(a, b)
        actual = cfn(a, b)
        self.assertEqual(expect, actual)

    def test_sym_stride_lowering(self, device):
        def fn(x):
            s0 = (x + 1).stride(0)
            return x * s0

        a = torch.randn(32, 32, device=device)
        cfn = self.compile_fn(fn)
        self.assertEqual(fn(a), cfn(a))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_item_materialize(self, device):
        def fn(x):
            return x.sum(dim=0).view(4).tolist()

        cfn = torch.compile(fullgraph=True)(fn)

        a = torch.ones(3, 4, dtype=torch.int64, device=device)
        self.assertEqual(cfn(a), fn(a))

    def test_abs(self, device):
        def fn(x, y):
            y0, y1 = y.shape
            # Slicing checks abs in wrapper code,
            # multiplication tests abs in kernel code
            return x[: abs(y0 - y1)] * abs(y0 - y1)

        a = torch.randn(32, 32, device=device)
        cfn = self.compile_fn(fn)

        # y0 > y1 -> y0 - y1 is positive
        b = torch.randn(16, 2, device=device)
        expect = fn(a, b)
        actual = cfn(a, b)
        self.assertEqual(expect, actual)

        # y0 < y1 -> y0 - y1 is negative
        b = torch.randn(2, 16, device=device)
        expect = fn(a, b)
        actual = cfn(a, b)
        self.assertEqual(expect, actual)

    def test_float_is_integer(self, device):
        def fn(x, mul, dim=-1):
            size = x.size(dim)
            m = size / mul
            if m.is_integer():
                return m
            return size

        a = torch.randn((3, 6, 4, 2), device=device)
        cfn = self.compile_fn(fn)

        expect = fn(a, 2)
        actual = cfn(a, 2)
        self.assertEqual(expect, actual)

    @onlyCPU
    def test_arithmetic_constant_folding(self, device):
        def test(fn):
            cfn = self.compile_fn(fn)
            expect = fn(3)
            actual = cfn(3)
            self.assertEqual(expect, actual)

        def add(x):
            return x + torch.zeros(3)

        test(add)

        def mul(x):
            return x * torch.ones(3)

        test(mul)

        def div(x):
            return x / torch.ones(3)

        test(div)

    @onlyCPU
    def test_sub_constant_folding(self, device):
        def sub(x):
            return x - torch.zeros(3)

        cfn = self.compile_fn(sub)
        expect = sub(3)
        actual = cfn(3)
        self.assertEqual(expect, actual)

    def test_full_symbolic_value(self, device):
        def fn(a):
            return torch.full((3,), a), torch.full((3,), torch.sym_float(a))

        cfn = self.compile_fn(fn)
        expect = fn(5)
        actual = cfn(5)
        self.assertEqual(expect, actual)

    def test_interpolate_ceil_eq(self, device):
        ceiling = math.ceil
        IntTrueDiv = operator.truediv

        def fn(t):
            s0, s2, s3 = t.size()
            x = torch.zeros(
                (
                    s0,
                    2048,
                    ceiling(IntTrueDiv(2 * ((s2 - 1) // 8) + 2, 1)),
                    ceiling(IntTrueDiv(2 * ((s3 - 1) // 8) + 2, 1)),
                ),
                dtype=torch.bfloat16,
            )
            return torch.nn.functional.interpolate(
                x,
                scale_factor=2,
                mode="nearest",
            )

        cfn = self.compile_fn(fn)
        arg = torch.randn(4, 16, 18)
        expect = fn(arg)
        actual = cfn(arg)
        self.assertEqual(expect, actual)

    def test_full_recompiles(self, device):
        def fn(x):
            _, L = x.shape
            return torch.full((L, L), torch.finfo(torch.float16).min, device=device)

        cfn = self.compile_fn(fn)

        import functools

        input_fn = functools.partial(torch.randint, 10, 1000, device=device)

        cfn(input_fn((2, 3)))
        cfn(input_fn((2, 4)))  # expect don't recompile here

        # check compiled times of frame 0
        from torch._dynamo.convert_frame import FRAME_COMPILE_COUNTER

        self.assertEqual(FRAME_COMPILE_COUNTER[0], 1)

    @parametrize(
        "op",
        [
            math.sqrt,
            math.sin,
            math.cos,
            math.cosh,
            math.sin,
            math.sinh,
            math.tan,
            math.tanh,
            math.asin,
            math.acos,
            math.atan,
        ],
    )
    def test_math_ops(self, device, op):
        def func(x, fn, a):
            return x + fn(a)

        cfunc = self.compile_fn(func, fullgraph=True)
        x = torch.rand(10, device=device)
        a = -1 if op in (math.asin, math.acos) else 12
        expected = func(x, op, a)
        output = cfunc(x, op, a)
        self.assertEqual(output, expected)

    def test_wrapper_codegen_statically_known_int_or_none(self):
        torch._dynamo.reset()

        _x = torch.randn([5, 3, 3])
        torch._dynamo.maybe_mark_dynamic(_x, 0)

        # Simple functions introducing constraints on x.shape[0]
        def fn_1(x):
            # no constraint
            return x.sin()

        def fn_2(x):
            # constrain in two directions
            if x.shape[0] > 5:
                return x.cos()
            if x.shape[0] < 5:
                return x * 2
            # x.shape[0] == 5 at this point
            return x.sin()

        def fn_3(x):
            # equality constraint, which matches example shape
            if x.size(0) == 5:
                return x.sin()
            else:
                return x.cos()

        call_count = 0

        def _test_wrapper_codegen_statically_known_int_or_none_in_context():
            nonlocal call_count
            call_count += 1
            graph = V.graph
            input_layouts = [
                inp.layout
                for inp in graph.graph_inputs.values()
                if hasattr(inp, "layout")
            ]
            batch_dim = input_layouts[0].size[0]
            if call_count == 1:
                # testing fn_1
                assert (
                    PythonWrapperCodegen.statically_known_int_or_none(batch_dim) is None
                ), "Should not be statically known on first call"
            elif call_count == 2:
                # testing fn_2
                assert (
                    PythonWrapperCodegen.statically_known_int_or_none(batch_dim) == 5
                ), "Should be limited to exactly 5 on second call due to multiple constraints"
            elif call_count == 2:
                # testing fn_3
                assert (
                    PythonWrapperCodegen.statically_known_int_or_none(batch_dim) == 5
                ), "Should be exactly 5 on third call"

        class TestWrapperCodegen(PythonWrapperCodegen):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def generate(self, is_inference, *args, **kwargs):
                _test_wrapper_codegen_statically_known_int_or_none_in_context()
                return super().generate(is_inference, *args, **kwargs)

        if "cpu" not in device_codegens:
            register_backend_for_device("cpu", CppScheduling, PythonWrapperCodegen)
        orig_cpu_codegens = device_codegens["cpu"]
        try:
            register_backend_for_device(
                "cpu", orig_cpu_codegens.scheduling, TestWrapperCodegen
            )
            # Compile each of the functions above, with an example input
            # that has 5 in the first dimension, but is marked as dynamic

            torch.compile(backend="inductor", dynamic=None)(fn_1)(_x)
            torch.compile(backend="inductor", dynamic=None)(fn_2)(_x)
            torch.compile(backend="inductor", dynamic=None)(fn_3)(_x)
        finally:
            register_backend_for_device(
                "cpu", orig_cpu_codegens.scheduling, orig_cpu_codegens.wrapper_codegen
            )

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_item_unbacked_stride_nobreak(self, device):
        @torch.compile(fullgraph=True, dynamic=True)
        def f(x):
            a = x.item()
            torch._check_is_size(a)
            torch._check(a >= 1)
            torch._check(a <= 10)
            return torch.ones(a, a)

        f(torch.tensor([5], device=device))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_symint_sum_list(self, device):
        @torch.compile()
        def f(xt):
            xs = xt.tolist()
            for x in xs:
                torch._check_is_size(x)
            y = sum(xs)
            return torch.zeros(y, device=device)

        f(torch.tensor([5] * 320))

    def test_mark_unbacked_slice(self):
        @torch.compile(backend="inductor", mode="reduce-overhead", fullgraph=True)
        def f(x):
            return x.sum()

        x = torch.empty_strided((1, 4), (5, 1), device=GPU_TYPE)
        torch._dynamo.decorators.mark_unbacked(x, 0)
        f(x)

    @torch._dynamo.config.patch(specialize_float=False, capture_scalar_outputs=True)
    def test_unspecialized_float_operations(self):
        operations = {
            "multiply": operator.mul,
            "add": operator.add,
            "subtract": operator.sub,
            "divide": operator.truediv,
        }

        for name, op in operations.items():
            with self.subTest(operation=name):

                def fn(x, y):
                    return op(x, y)

                cnt = CompileCounterWithBackend("inductor")
                fn_opt = torch._dynamo.optimize(cnt)(fn)

                x = torch.arange(3)
                self.assertEqual(fn(x, 2.0), fn_opt(x, 2.0))
                self.assertEqual(fn(x, 3.0), fn_opt(x, 3.0))
                self.assertEqual(cnt.frame_count, 1)

    @torch._dynamo.config.patch(specialize_float=False)
    def test_unspecialized_float_fallback_specialization(self):
        def fn(x, y, z):
            return (
                torch.tensor(z),
                torch.exp(torch.tensor(z)) * (x * y),
                x.size(0),
                math.sqrt(x.size(0)),
                math.floor(math.sqrt(x.size(0))),
                math.floor(math.sqrt(x.numel())),
                math.floor(math.sqrt(x.dim())),
                math.floor(math.sqrt(z)),
            )

        cnt = CompileCounterWithBackend("inductor")
        fn_opt = torch._dynamo.optimize(cnt)(fn)
        x = torch.arange(3)
        z = 1.3

        self.assertEqual(fn(x, 2.0, z), fn_opt(x, 2.0, z))
        self.assertEqual(fn(x, 3.0, z), fn_opt(x, 3.0, z))
        self.assertEqual(cnt.frame_count, 1)

    @torch._dynamo.config.patch(specialize_float=False)
    def test_unspecialized_float_fallback_symint_specialization(self):
        def fn(x, y):
            return math.floor(x**2) * y

        cnt = CompileCounterWithBackend("inductor")
        fn_opt = torch._dynamo.optimize(cnt)(fn)
        y = torch.arange(3)

        self.assertEqual(fn(2.0, y), fn_opt(2.0, y))
        self.assertEqual(fn(3.0, y), fn_opt(3.0, y))
        self.assertEqual(cnt.frame_count, 2)

    def test_sort_dynamic_shape_with_check(self, device):
        if TEST_WITH_ROCM or torch.device(device).type != GPU_TYPE:

            def check_count(n):
                self.assertEqual(metrics.generated_kernel_count, 0)

        else:

            def check_count(n):
                self.assertEqual(metrics.generated_kernel_count, n)

        # Test dynamic shapes with statically known small enough to generate
        # persistent sort kernel
        def fn(a, descending):
            torch._check(a.shape[-1] <= 256)
            return a.sort(dim=-1, stable=True, descending=descending)

        inp = torch.rand(10, 128, dtype=torch.float32, device=device)
        inp[:, 10:20] = 1.0
        inp[:, 30:40] = 1.0
        metrics.reset()

        opt_fn = torch.compile(fn, dynamic=True)
        expect = fn(inp, False)
        actual = opt_fn(inp, False)
        self.assertEqual(actual, expect)
        check_count(1)

        expect = fn(inp, True)
        actual = opt_fn(inp, True)
        self.assertEqual(actual, expect)
        check_count(2)

        # Non-power of two
        inp[:, :120]

        expect = fn(inp, False)
        actual = opt_fn(inp, False)
        self.assertEqual(actual, expect)
        check_count(2)  # Reused existing kernel

        expect = fn(inp, True)
        actual = opt_fn(inp, True)
        self.assertEqual(actual, expect)
        check_count(2)  # Reused existing kernel


instantiate_device_type_tests(TestInductorDynamic, globals(), allow_xpu=True)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    # Slow on ASAN after https://github.com/pytorch/pytorch/pull/94068
    if (HAS_CPU or HAS_GPU) and not TEST_WITH_ASAN:
        run_tests(needs="filelock")
