# Owner(s): ["oncall: aiacc"]

import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
import torch.nn as nn
from torch.testing._internal.common_fx2trt import AccTestCase
from torch.testing._internal.common_utils import run_tests
from parameterized import parameterized


class TestHardtanhConverter(AccTestCase):
    @parameterized.expand([
        (-2.0, 6),
        (0, 1),
        (0.5, 7),
    ])
    def test_hardtanh(self, test_min_value, test_max_value):
        class Hardtanh(nn.Module):
            def forward(self, x):
                return nn.functional.hardtanh(x, min_val=test_min_value, max_val=test_max_value)

        inputs = [torch.randn(2, 10, 10, 10)]
        self.run_test(Hardtanh(), inputs, expected_ops={acc_ops.hardtanh})

if __name__ == '__main__':
    run_tests()
