import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
import torch.nn as nn
from torch.testing._internal.common_fx2trt import AccTestCase, InputTensorSpec
from torch.testing._internal.common_utils import run_tests


class TestGELU(AccTestCase):
    def test_gelu(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return nn.functional.gelu(x)

        inputs = [torch.randn(3, 10, 20)]
        self.run_test(
            TestModule(),
            inputs,
            expected_ops={acc_ops.gelu},
            test_implicit_batch_dim=False,
        )

    def test_gelu_with_dynamic_shape(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return nn.functional.gelu(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1), (1, 2, 3), (3, 3, 3))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, expected_ops={acc_ops.gelu}
        )

if __name__ == '__main__':
    run_tests()
