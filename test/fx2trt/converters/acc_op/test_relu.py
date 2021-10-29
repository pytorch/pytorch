# Owner(s): ["oncall: fx"]

import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
import torch.nn as nn
from caffe2.torch.fb.fx2trt.tests.test_utils import AccTestCase, InputTensorSpec


class TestReLUConverter(AccTestCase):
    def test_relu(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return nn.functional.relu(x)

        inputs = [torch.randn(1, 10)]
        self.run_test(TestModule(), inputs, expected_ops={acc_ops.relu})

    def test_relu_with_dynamic_shape(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return nn.functional.relu(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1), (1, 2, 3), (3, 3, 3))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, expected_ops={acc_ops.relu}
        )
