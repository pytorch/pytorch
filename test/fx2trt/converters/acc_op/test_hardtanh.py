# Owner(s): ["oncall: fx"]

import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
import torch.nn as nn
from torch.testing._internal.common_fx2trt import AccTestCase
from torch.testing._internal.common_utils import run_tests


class TestHardtanhConverter(AccTestCase):
    def test_hardtanh(self):
        class Hardtanh(nn.Module):
            def forward(self, x):
                return nn.functional.hardtanh(x, min_val=-0.5)

        inputs = [torch.randn(2, 10, 10, 10)]
        self.run_test(Hardtanh(), inputs, expected_ops={acc_ops.hardtanh})

if __name__ == '__main__':
    run_tests()
