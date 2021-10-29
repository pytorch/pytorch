# Owner(s): ["oncall: fx"]

import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
import torch.nn as nn
from caffe2.torch.fb.fx2trt.tests.test_utils import AccTestCase, InputTensorSpec


class TestTanh(AccTestCase):
    def test_tanh(self):
        class Tanh(nn.Module):
            def forward(self, x):
                return torch.tanh(x)

        inputs = [torch.randn(1, 2, 3)]
        self.run_test(Tanh(), inputs, expected_ops={acc_ops.tanh})

    def test_tanh_with_dynamic_shape(self):
        class Tanh(nn.Module):
            def forward(self, x):
                return torch.tanh(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1), (1, 2, 3), (3, 3, 3))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            Tanh(), input_specs, expected_ops={acc_ops.tanh}
        )
