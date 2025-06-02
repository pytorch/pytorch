# Owner(s): ["module: vulkan"]
import importlib
import os
import sys

import numpy as np

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import get_all_dtypes
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    MACOS_VERSION,
    parametrize,
)

VULKAN_UNSUPPORTED_TYPES = [torch.double, torch.cdouble, torch.half, torch.bfloat16, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.complex64]
VULKAN_DTYPES = [t for t in get_all_dtypes() if t not in VULKAN_UNSUPPORTED_TYPES]

importlib.import_module("filelock")

pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

from inductor.test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
    check_model_gpu,
    CommonTemplate,
    TestCase,
)

torch._register_device_module("privateuseone", object())
torch.set_printoptions(threshold=99999)

@instantiate_parametrized_tests
class VulkanBasicTests(TestCase):
    is_dtype_supported = CommonTemplate.is_dtype_supported
    common = check_model_gpu
    device = "vulkan"

    @parametrize("dtype", VULKAN_DTYPES)
    def test_add(self, dtype):
        self.common(
            lambda a, b: a + b,
            (
                make_tensor(1024, dtype=dtype, device="cpu").to(self.device),
                make_tensor(1024, dtype=dtype, device="cpu").to(self.device),
            ),
            check_lowp=False,
            reference_on_cpu=True,
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if torch.is_vulkan_available():
        run_tests(needs="filelock")
