# Owner(s): ["oncall: fx"]

import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
import torch.nn as nn
from caffe2.torch.fb.fx2trt.tests.test_utils import AccTestCase
from parameterized import parameterized


class TestNarrowConverter(AccTestCase):
    @parameterized.expand(
        [
            ("positive_dim", 1, 0, 1),
            ("negative_dim", -1, 1, 2),
        ]
    )
    def test_narrow(self, _, dim, start, length):
        class Narrow(nn.Module):
            def forward(self, x):
                return x.narrow(dim, start, length)

        inputs = [torch.randn(1, 2, 3, 4)]
        self.run_test(
            Narrow(),
            inputs,
            expected_ops={acc_ops.slice_tensor},
            test_explicit_batch_dim=False,
        )
