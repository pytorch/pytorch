# Owner(s): ["oncall: fx"]

import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
from caffe2.torch.fb.fx2trt.tests.test_utils import AccTestCase
from parameterized import parameterized


class TestSumConverter(AccTestCase):
    @parameterized.expand(
        [
            ("single_dim_no_keepdim", 1, False),
            ("single_dim_keepdim", 1, True),
            ("two_dim_no_keepdim", (1, 2), False),
            ("two_dim_keepdim", (1, 2), True),
            ("three_dim_no_keepdim", (1, 2, 3), False),
            ("three_dim_keepdim", (1, 2, 3), True),
            ("dim0_keepdim", 0, True),
            ("dim0_no_keepdim", 0, False),
        ]
    )
    def test_sum(self, test_name, dim, keepdim):
        class Sum(torch.nn.Module):
            def __init__(self, dim, keepdim):
                super().__init__()
                self.dim = dim
                self.keepdim = keepdim

            def forward(self, x):
                return x.sum(dim=self.dim, keepdim=self.keepdim)

        inputs = [torch.randn(1, 2, 3, 4)]
        self.run_test(
            Sum(dim, keepdim),
            inputs,
            expected_ops={acc_ops.sum},
            test_implicit_batch_dim=(dim != 0),
        )

    @parameterized.expand(
        [
            ("no_dim_no_keepdim"),
        ]
    )
    def test_sum_explicit_only(
        self,
        test_name,
    ):
        class Sum(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.sum(x)

        inputs = [torch.randn(1, 2, 3, 4)]
        self.run_test(
            Sum(),
            inputs,
            expected_ops={acc_ops.sum},
            test_implicit_batch_dim=False,
        )
