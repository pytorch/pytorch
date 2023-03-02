# Owner(s): ["module: inductor"]
import importlib
import os
import sys
import unittest

import torch
from torch._dynamo.testing import make_test_cls_with_patches
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
    "test_alexnet_prefix_dynamic_shapes": ("cuda",),
    "test_baddbmm_dynamic_shapes": ("cpu", "cuda"),
    "test_cpp_wrapper_dynamic_shapes": ("cpu",),
    "test_cudnn_rnn_dynamic_shapes": ("cuda",),
    "test_kwargs_dynamic_shapes": ("cpu",),
    "test_lowmem_dropout2_dynamic_shapes": ("cpu", "cuda"),
    "test_rand_like_deterministic_dynamic_shapes": ("cpu", "cuda"),
    "test_randn_like_empty_dynamic_shapes": ("cpu", "cuda"),
    # test_roi_align uses torchvision, which doesn't work with dynamic shapes
    "test_roi_align_dynamic_shapes": ("cpu", "cuda"),
    "test_sizehint_issue1_dynamic_shapes": ("cpu", "cuda"),
    "test_unroll_small_reduction_dynamic_shapes": ("cpu", "cuda"),
    "test_upsample_bilinear2d_a_dynamic_shapes": ("cpu"),
    "test_upsample_bilinear2d_b_dynamic_shapes": ("cpu"),
    "test_upsample_nearest1d_dynamic_shapes": ("cpu"),
    "test_upsample_nearest2d_backward_dynamic_shapes": ("cpu", "cuda"),
    "test_upsample_nearest2d_dynamic_shapes": ("cpu"),
    "test_upsample_nearest3d_dynamic_shapes": ("cpu"),
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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if (HAS_CPU or HAS_CUDA) and not TEST_WITH_ROCM:
        run_tests(needs="filelock")
