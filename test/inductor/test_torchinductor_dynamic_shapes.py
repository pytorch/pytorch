# Owner(s): ["module: inductor"]
import contextlib
import importlib
import math
import os
import sys
import unittest
from functools import partial

import torch
from torch._dynamo.testing import make_test_cls_with_patches
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import (
    IS_CI,
    IS_WINDOWS,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
    TestCase,
)
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA

if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor_dynamic_shapes yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from inductor.test_torchinductor import (
    check_model,
    check_model_cuda,
    CommonTemplate,
    copy_tests,
    TestFailure,
)

importlib.import_module("filelock")

# xfail by default, set is_skip=True to skip
test_failures = {
    "test_kwargs_dynamic_shapes": TestFailure(("cpu",)),
    "test_conv2d_unary_dynamic_shapes": TestFailure(("cpu",), is_skip=True),
}

if TEST_WITH_ROCM:
    # Tensor-likes are not close
    test_failures["test_convolution1_dynamic_shapes"] = TestFailure(
        ("cpu", "cuda"), is_skip=True
    )
    test_failures["test_convolution3_dynamic_shapes"] = TestFailure(
        ("cuda"), is_skip=True
    )
    test_failures["test_expanded_reduction_dynamic_shapes"] = TestFailure(
        ("cuda"), is_skip=True
    )
    test_failures["test_batch_norm_2d_dynamic_shapes"] = TestFailure(
        ("cuda"), is_skip=True
    )


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


if HAS_CUDA and not TEST_WITH_ASAN:

    class DynamicShapesCudaTests(TestCase):
        common = check_model_cuda
        device = "cuda"

    copy_tests(
        DynamicShapesCommonTemplate, DynamicShapesCudaTests, "cuda", test_failures
    )


class TestInductorDynamic(TestCase):
    compile_fn = partial(torch.compile, dynamic=True)

    def setUp(self):
        # HAS_CUDA also checks compute capability to skip tests
        # on older devices
        if self.device_type == "cuda" and not HAS_CUDA:
            self.skipTest("Triton not available")
        torch._dynamo.reset()
        super(TestCase, self).setUp()
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
        super(TestCase, self).tearDown()
        torch._dynamo.reset()

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

    @onlyCUDA
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


instantiate_device_type_tests(TestInductorDynamic, globals())

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    # Slow on ASAN after https://github.com/pytorch/pytorch/pull/94068
    if (HAS_CPU or HAS_CUDA) and not TEST_WITH_ASAN:
        run_tests(needs="filelock")
