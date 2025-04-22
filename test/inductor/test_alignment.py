# Owner(s): ["module: inductor"]
import contextlib
import sys
import unittest

import torch
from torch._inductor import config
from torch.testing._internal.common_utils import MACOS_VERSION
from torch.testing._internal.inductor_utils import GPU_TYPE, RUN_CPU, RUN_GPU


try:
    try:
        from . import test_torchinductor
    except ImportError:
        import test_torchinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise

TestCase = test_torchinductor.TestCase
check_model = test_torchinductor.check_model
check_model_gpu = test_torchinductor.check_model_gpu
skip_if_cpp_wrapper = test_torchinductor.skip_if_cpp_wrapper
copy_tests = test_torchinductor.copy_tests
define_custom_op_for_test = test_torchinductor.define_custom_op_for_test


class CommonTemplate:
    def test_unaligned_input(self):
        def fn(x):
            return torch.nn.functional.relu(x)

        x = torch.randn(1024 + 16, device=self.device)[1:-15]
        # TODO (malfet): Investigate failures on MacOS-14
        with (
            contextlib.nullcontext()
            if self.device != "mps" or MACOS_VERSION >= 15.0
            else self.assertRaises(AssertionError)
        ):
            self.common(fn, (x,), check_lowp=False)

    def test_unaligned_input_2d(self):
        def fn(x):
            return torch.nn.functional.relu(x)

        x = torch.randn(1024, 1024 + 16, device=self.device)[:, 1:-15]
        self.common(fn, (x,), check_lowp=False)

    def test_alignment_without_custom_op(self):
        def fn(x):
            a = torch.nn.functional.relu(x)
            b = (3 * a)[1:-15]
            c = torch.cos(b)
            return c

        x = torch.randn(1024 + 16, device=self.device)
        self.common(fn, (x,), check_lowp=False)

    @config.patch(implicit_fallbacks=True)
    def test_no_align_for_custom_op(self):
        def slice1d(x):
            return (3 * x)[1:-15]

        def slice1d_meta(x):
            return torch.empty_like(x)[1:-15]

        define_custom_op_for_test("slice1d", slice1d, slice1d_meta)

        def fn(x):
            a = torch.nn.functional.relu(x)
            b = torch.ops.test.slice1d(a)
            c = torch.cos(b)
            return c

        x = torch.randn(1024 + 16, device=self.device)
        self.common(fn, (x,), check_lowp=False)

    @config.patch(implicit_fallbacks=True)
    def test_no_align_for_custom_op_2d(self):
        def slice2d(x):
            return (3 * x)[..., 1:-15]

        def slice2d_meta(x):
            return torch.empty_like(x)[..., 1:-15]

        define_custom_op_for_test("slice2d", slice2d, slice2d_meta)

        def fn(x):
            a = torch.nn.functional.relu(x)
            b = torch.ops.test.slice2d(a)
            c = torch.cos(b)
            return c

        x = torch.randn(1024, 1024 + 16, device=self.device)
        self.common(fn, (x,), check_lowp=False)

    @config.patch(implicit_fallbacks=True, alignment_asserts=True)
    @skip_if_cpp_wrapper(
        "Inductor does not generate alignment assertion for cpp_wrapper right now"
    )
    def test_incorrect_meta_for_custom_op_2d(self):
        def slice2d(x):
            return (3 * x)[..., 1:-15]

        def slice2d_meta(x):
            return torch.empty_like(x)[..., 0:-16]

        define_custom_op_for_test("slice2d_incorrect_meta", slice2d, slice2d_meta)

        def fn(x):
            a = torch.nn.functional.relu(x)
            b = torch.ops.test.slice2d_incorrect_meta(a)
            c = torch.cos(b)
            return c

        x = torch.randn(1024, 1024 + 16, device=self.device)

        expected_error = "Expect the tensor to be 16 bytes aligned. Fail due to storage_offset=1 itemsize=4"
        with self.assertRaisesRegex(AssertionError, expected_error):
            self.common(fn, (x,), check_lowp=False)


if RUN_CPU:

    class CpuTests(TestCase):
        common = check_model
        device = "cpu"

    copy_tests(CommonTemplate, CpuTests, "cpu")

if RUN_GPU:

    class GPUTests(TestCase):
        common = check_model_gpu
        device = GPU_TYPE

    copy_tests(CommonTemplate, GPUTests, GPU_TYPE)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if RUN_CPU or RUN_GPU:
        run_tests()
