# Owner(s): ["oncall: fx"]

import torch
import torch.fx
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
import torch.nn as nn
from caffe2.torch.fb.fx2trt.tests.test_utils import AccTestCase, InputTensorSpec


class TestUnsqueeze(AccTestCase):
    def test_unsqueeze(self):
        class Unsqueeze(nn.Module):
            def forward(self, x):
                return torch.unsqueeze(x, 1)

        inputs = [torch.randn(1, 2, 3)]
        self.run_test(Unsqueeze(), inputs, expected_ops={acc_ops.unsqueeze})

    def test_unsqueeze_with_dynamic_shape(self):
        class Unsqueeze(nn.Module):
            def forward(self, x):
                return torch.unsqueeze(x, 1)

        input_specs = [
            InputTensorSpec(
                shape=(-1, 2, 3),
                dtype=torch.float32,
                shape_ranges=[((1, 2, 3), (2, 2, 3), (3, 2, 3))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            Unsqueeze(), input_specs, expected_ops={acc_ops.unsqueeze}
        )
