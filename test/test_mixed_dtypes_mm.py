# FIXME: find proper module name to put here!
# Owner(s): ["module: ???"]

import itertools
import random
import unittest

import torch
from torch import nn

from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    run_tests,
    TestCase,
    TEST_WITH_ROCM,
)

class TestMixedDtypesMM(TestCase):
    @unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support CUTLASS")
    def test_fp16_int8_mm(self, device):
        def run_test(m, n, k, device):
            fp16_low, fp16_high = 0.5, 1.5
            ui8_low, ui8_high = 129, 130 #FIXME: make this range broader
            a = make_tensor(m, k, low=fp16_low, high=fp16_high, dtype=torch.float16, device=device)
            b = make_tensor(k, n, low=ui8_low, high=ui8_high, dtype=torch.uint8, device=device)
            scale = make_tensor((1, n), low=fp16_low, high=fp16_high, dtype=a.dtype, device=device)
            bias = make_tensor((1, n), low=fp16_low, high=fp16_high, dtype=a.dtype, device=device)

            weights = (b.to(a.dtype) - 128) * scale.expand((k, n))
            
            c_ref = torch.mm(a, weights) + bias
            
            c = torch.ops.aten._fp16_uint8_mm(a, b, scale, bias)

            torch.testing.assert_close(c, c_ref, rtol=1e-3, atol=0)

        shapes = [[32, 32, 128], [32, 64, 128], [64, 32, 128], [64, 64, 128]]
        for m, n, k in shapes:
            run_test(m, n, k, device)

instantiate_device_type_tests(TestMixedDtypesMM, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
