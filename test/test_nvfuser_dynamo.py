# Owner(s): ["module: nvfuser"]

import unittest
import warnings

import torch
import torch._dynamo as torchdynamo
from torch.testing import make_tensor
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfTorchDynamo,
    TEST_WITH_ROCM,
    TestCase,
    IS_WINDOWS,
)
from torch.testing._internal.jit_utils import RUN_CUDA

RUN_NVFUSER = RUN_CUDA and not TEST_WITH_ROCM


def is_pre_volta():
    if not RUN_NVFUSER:
        return False
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.major < 7


@skipIfTorchDynamo("Not a suitable test for TorchDynamo")
@unittest.skipIf(IS_WINDOWS, "TorchDynamo is not supported on Windows")
@unittest.skipIf(not RUN_NVFUSER, "requires CUDA")
@unittest.skipIf(is_pre_volta(), "Only supported on Volta and newer devices.")
class TestNvFuserDynamo(TestCase):
    def test_basic(self):
        input1 = make_tensor((2, 4, 8), device="cuda", dtype=torch.float32)
        input2 = make_tensor((2, 4, 8), device="cuda", dtype=torch.float32)

        @torchdynamo.optimize("nvprims_nvfuser")
        def func(a, b):
            return a.sin() + b.cos()

        # No warnings and no errors
        with warnings.catch_warnings(record=True) as w:
            nvfuser_result = func(input1, input2)
            self.assertEqual(len(w), 0)
        eager_result = func.__wrapped__(input1, input2)
        self.assertEqual(eager_result, nvfuser_result)

    def test_dtype_correctness(self):
        input1 = make_tensor((2, 4, 8), device="cuda", dtype=torch.float32)

        @torchdynamo.optimize("nvprims_nvfuser")
        def func(a):
            tmp = a + 1.0
            return torch.where(tmp > 0, tmp, 0.0)

        # No warnings and no errors
        with warnings.catch_warnings(record=True) as w:
            nvfuser_result = func(input1)
            self.assertEqual(len(w), 0)
        eager_result = func.__wrapped__(input1)
        self.assertEqual(eager_result, nvfuser_result)


if __name__ == "__main__":
    run_tests()
