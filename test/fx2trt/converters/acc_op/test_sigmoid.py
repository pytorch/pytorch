# Owner(s): ["oncall: fx"]

import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
import torch.nn as nn
from caffe2.torch.fb.fx2trt.tests.test_utils import AccTestCase


class TestSigmoid(AccTestCase):
    def test_sigmoid(self):
        class Sigmoid(nn.Module):
            def forward(self, x):
                return torch.sigmoid(x)

        inputs = [torch.randn(1, 2, 3)]
        self.run_test(Sigmoid(), inputs, expected_ops={acc_ops.sigmoid})
