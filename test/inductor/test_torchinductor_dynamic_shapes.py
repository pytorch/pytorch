# Owner(s): ["module: inductor"]
import contextlib
import importlib
import os
import sys
import unittest
from functools import partial
from unittest.mock import patch

import torch
from torch._dynamo.testing import make_test_cls_with_patches
from torch.testing._internal.common_device_type import instantiate_device_type_tests
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
)

importlib.import_module("filelock")

test_skips = {
    "test_baddbmm_dynamic_shapes": ("cpu", "cuda"),
    "test_cpp_wrapper_dynamic_shapes": ("cpu",),
    "test_cudnn_rnn_dynamic_shapes": ("cuda",),
    "test_gather3_dynamic_shapes": ("cpu", "cuda"),
    "test_kwargs_dynamic_shapes": ("cpu",),
    "test_randn_like_empty_dynamic_shapes": ("cpu", "cuda"),
    # test_roi_align uses torchvision, which doesn't work with dynamic shapes
    "test_roi_align_dynamic_shapes": ("cpu", "cuda"),
    "test_unroll_small_reduction_dynamic_shapes": ("cpu", "cuda"),
    "test_upsample_nearest2d_backward_dynamic_shapes": ("cpu", "cuda"),
}


def make_dynamic_cls(cls):
    return make_test_cls_with_patches(
        cls,
        "DynamicShapes",
        "_dynamic_shapes",
        (torch._dynamo.config, "dynamic_shapes", True),
    )


DynamicShapesCommonTemplate = make_dynamic_cls(CommonTemplate)


if HAS_CPU:

    class DynamicShapesCpuTests(TestCase):
        common = check_model
        device = "cpu"

    copy_tests(DynamicShapesCommonTemplate, DynamicShapesCpuTests, "cpu", test_skips)


if HAS_CUDA and not TEST_WITH_ASAN:

    class DynamicShapesCudaTests(TestCase):
        common = check_model_cuda
        device = "cuda"

    copy_tests(DynamicShapesCommonTemplate, DynamicShapesCudaTests, "cuda", test_skips)


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

    @patch.object(torch._dynamo.config, "specialize_int", False)
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


instantiate_device_type_tests(TestInductorDynamic, globals())

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if (HAS_CPU or HAS_CUDA) and not TEST_WITH_ROCM:
        run_tests(needs="filelock")
