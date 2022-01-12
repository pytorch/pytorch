# Owner(s): ["oncall: fx"]

import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
import torch.nn as nn
from torch.testing._internal.common_fx2trt import AccTestCase
from torch.testing._internal.common_utils import run_tests


class TestSigmoid(AccTestCase):
    def test_sigmoid(self):
        class Sigmoid(nn.Module):
            def forward(self, x):
                return torch.sigmoid(x)

        inputs = [torch.randn(1, 2, 3)]
        self.run_test(Sigmoid(), inputs, expected_ops={acc_ops.sigmoid})

if __name__ == '__main__':
    run_tests()
