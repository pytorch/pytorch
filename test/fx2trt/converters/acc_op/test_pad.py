# Owner(s): ["oncall: fx"]

import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
import torch.nn as nn
from caffe2.torch.fb.fx2trt.tests.test_utils import AccTestCase
from parameterized import parameterized


class TestPadConverter(AccTestCase):
    @parameterized.expand(
        [
            ("1d", (1, 2)),
            ("2d", (2, 0, 0, 1)),
        ]
    )
    def test_pad(self, _, pad):
        class Pad(nn.Module):
            def forward(self, x):
                return torch.nn.functional.pad(x, pad)

        inputs = [torch.randn(1, 2, 3, 4)]
        self.run_test(
            Pad(),
            inputs,
            expected_ops={acc_ops.pad},
        )
