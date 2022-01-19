# Owner(s): ["oncall: fx"]

import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
import torch.nn as nn
from torch.testing._internal.common_fx2trt import AccTestCase, InputTensorSpec
from torch.testing._internal.common_utils import run_tests


class TestSizeConverter(AccTestCase):
    def test_size(self):
        class Size(nn.Module):
            def forward(self, x):
                bs = x.size(0)
                return x.view(bs, -1)

        inputs = [torch.randn(1, 2, 3, 4)]
        self.run_test(Size(), inputs, expected_ops={acc_ops.size})

    def test_size_dynamic_shape(self):
        class Size(nn.Module):
            def forward(self, x):
                bs = x.size(0)
                return x.view(bs, -1)

        input_specs = [
            InputTensorSpec(
                shape=(-1, 12, 32),
                dtype=torch.float32,
                shape_ranges=[((1, 12, 32), (3, 12, 32), (100, 12, 32))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            Size(), input_specs, expected_ops={acc_ops.size}
        )

if __name__ == '__main__':
    run_tests()
