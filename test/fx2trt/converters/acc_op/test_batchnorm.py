# Owner(s): ["oncall: fx"]

import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
from torch.testing._internal.common_fx2trt import AccTestCase, InputTensorSpec
from torch.testing._internal.common_utils import run_tests


class TestBatchNormConverter(AccTestCase):
    def test_batchnorm(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                return self.bn(x)

        inputs = [torch.randn(1, 3, 224, 224)]
        self.run_test(TestModule(), inputs, expected_ops={acc_ops.batch_norm})

    def test_batchnorm_with_dynamic_shape(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                return self.bn(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, 3, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 3, 1, 1), (1, 3, 5, 5), (2, 3, 10, 10))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, expected_ops={acc_ops.batch_norm}
        )

if __name__ == '__main__':
    run_tests()
