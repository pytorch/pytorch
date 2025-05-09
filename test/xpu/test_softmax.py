# Owner(s): ["module: intel"]

import itertools
import math
import random
from functools import partial
from itertools import product

import numpy as np

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    precisionOverride,
)
from torch.testing._internal.common_utils import (
    iter_indices,
    parametrize,
    run_tests,
    TestCase,
)
import torch.nn.functional as F

class TestBasicSoftmax(TestCase):
    def test_softmax_half_to_float(self, device):
        shape = [
            [8],
            [7, 8],
            [8192, 64],
            [8192, 8192],
            [7, 8, 512],
            [7, 8, 11],
            [16, 7, 8, 512],
            [16, 7, 8, 512, 35],
            [117, 7, 9, 513, 35],
        ]
        input_type = torch.float16
        output_type = torch.float
        for i in range(len(shape)):
            for j in range(len(shape[i])):
                dim = j - 1
                x = torch.randn(shape[i]).to(input_type)
                grad = torch.randn(shape[i]).to(output_type)
                x_cpu = x.clone().requires_grad_()
                y_cpu = F.softmax(x_cpu, dim, dtype=output_type)
                y_cpu.backward(grad.clone())

                x_xpu = x.clone().to(device).requires_grad_()
                y_xpu = F.softmax(x_xpu, dim, dtype=output_type)
                self.assertEqual(y_xpu.dtype, torch.float32)
                y_xpu.backward(grad.clone().to(device))
                self.assertEqual(y_cpu, y_xpu.cpu())
                self.assertEqual(x_cpu.grad, x_xpu.grad.cpu())

instantiate_device_type_tests(TestBasicSoftmax, globals(), only_for="xpu", allow_xpu=True)

if __name__ == "__main__":
    run_tests()
