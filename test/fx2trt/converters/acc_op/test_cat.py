# Owner(s): ["oncall: fx"]

import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
import torch.nn as nn
from torch.testing._internal.common_fx2trt import AccTestCase, InputTensorSpec
from torch.testing._internal.common_utils import run_tests


class TestCatConverter(AccTestCase):
    def test_cat(self):
        class Cat(nn.Module):
            def forward(self, x, y, z):
                return torch.cat((x, y, z), 1)

        inputs = [torch.randn(1, 2, 3), torch.randn(1, 1, 3), torch.randn(1, 3, 3)]
        self.run_test(Cat(), inputs, expected_ops={acc_ops.cat})

    def test_cat_with_dynamic_shape(self):
        class Cat(nn.Module):
            def forward(self, x, y):
                x = x + y
                return torch.cat((x, y), 0)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1), (2, 3, 4), (2, 3, 10))],
            ),
            InputTensorSpec(
                shape=(-1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1), (2, 3, 4), (2, 3, 10))],
            ),
        ]
        self.run_test_with_dynamic_shape(Cat(), input_specs, expected_ops={acc_ops.cat})

if __name__ == '__main__':
    run_tests()
