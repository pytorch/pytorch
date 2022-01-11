# Owner(s): ["oncall: fx"]

import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
from torch.testing._internal.common_fx2trt import AccTestCase
from torch.testing._internal.common_utils import run_tests


class TestMinimumConverter(AccTestCase):
    def test_minimum(self):
        class Minimum(torch.nn.Module):
            def forward(self, x, y):
                return torch.minimum(x, y)

        inputs = [
            torch.randn(3, 4),
            torch.randn(3, 4),
        ]
        self.run_test(Minimum(), inputs, expected_ops={acc_ops.minimum})


class TestMinimumMethodConverter(AccTestCase):
    def test_minimum(self):
        class Minimum(torch.nn.Module):
            def forward(self, x, y):
                return x.minimum(y)

        inputs = [
            torch.randn(3, 4),
            torch.randn(3, 4),
        ]
        self.run_test(Minimum(), inputs, expected_ops={acc_ops.minimum})

if __name__ == '__main__':
    run_tests()
