# Owner(s): ["oncall: fx"]

import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
from torch.testing._internal.common_fx2trt import AccTestCase
from torch.testing._internal.common_utils import run_tests


class TestMaximumConverter(AccTestCase):
    def test_maximum(self):
        class Maximum(torch.nn.Module):
            def forward(self, x, y):
                return torch.maximum(x, y)

        inputs = [
            torch.randn(3, 4),
            torch.randn(3, 4),
        ]
        self.run_test(Maximum(), inputs, expected_ops={acc_ops.maximum})


class TestMaximumMethodConverter(AccTestCase):
    def test_maximum(self):
        class Maximum(torch.nn.Module):
            def forward(self, x, y):
                return x.maximum(y)

        inputs = [
            torch.randn(3, 4),
            torch.randn(3, 4),
        ]
        self.run_test(Maximum(), inputs, expected_ops={acc_ops.maximum})

if __name__ == '__main__':
    run_tests()
