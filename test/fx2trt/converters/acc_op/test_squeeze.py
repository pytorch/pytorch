# Owner(s): ["oncall: fx"]

import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
import torch.nn as nn
from torch.testing._internal.common_fx2trt import AccTestCase, InputTensorSpec
from torch.testing._internal.common_utils import run_tests


class TestSqueeze(AccTestCase):
    def test_squeeze(self):
        class Squeeze(nn.Module):
            def forward(self, x):
                return x.squeeze(2)

        inputs = [torch.randn(1, 2, 1)]
        self.run_test(Squeeze(), inputs, expected_ops={acc_ops.squeeze})

    def test_squeeze_with_dynamic_shape(self):
        class Squeeze(nn.Module):
            def forward(self, x):
                return x.squeeze(0)

        input_specs = [
            InputTensorSpec(
                shape=(1, -1, 2),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 2), (1, 2, 2), (1, 3, 2))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            Squeeze(), input_specs, expected_ops={acc_ops.squeeze}
        )

if __name__ == '__main__':
    run_tests()
