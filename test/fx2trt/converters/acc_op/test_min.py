# Owner(s): ["oncall: fx"]

import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
from torch.testing._internal.common_fx2trt import AccTestCase
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests


class TestMinConverter(AccTestCase):
    @parameterized.expand(
        [
            ("dim0_keepdim", 0, True, torch.randn(2, 2, 3)),
            ("dim1_keepdim", 1, True, torch.randn(2, 2, 3)),
            ("dim2_keepdim", 2, True, torch.randn(2, 2, 3)),
            ("dim3_keepdim", 3, True, torch.randn(2, 2, 3, 3)),
            ("dim2_no_keepdim", 2, False, torch.randn(2, 2, 3)),
            ("dim1_no_keepdim", 1, False, torch.randn(2, 2, 3)),
            ("dim0_no_keepdim", 0, False, torch.randn(2, 2, 3)),
        ]
    )
    def test_min_dim_reduce(self, test_name, dim, keepdim, input):
        class MinDimReduce(torch.nn.Module):
            def __init__(self, dim, keepdim):
                super().__init__()
                self.dim = dim
                self.keepdim = keepdim

            def forward(self, x):
                return torch.min(x, self.dim, self.keepdim)

        inputs = [input]
        self.run_test(
            MinDimReduce(dim, keepdim),
            inputs,
            expected_ops={acc_ops.min_dim_reduce},
            test_implicit_batch_dim=(dim != 0),
        )

    @parameterized.expand(
        [
            ("no_dim_no_keepdim"),
        ]
    )
    def test_min_full_reduce(
        self,
        test_name,
    ):
        class MinFullReduce(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.min(x)

        inputs = [torch.randn(3, 2, 3, 3)]
        self.run_test(
            MinFullReduce(),
            inputs,
            expected_ops={acc_ops.min_full_reduce},
            # We can't do a full reduce over the batch dimension
            test_implicit_batch_dim=False,
        )

    @parameterized.expand(
        [
            ("min_method_no_dim_no_keepdim"),
            ("min_method_no_dim_no_keepdim"),
        ]
    )
    def test_min_method(self, test_name):
        class MinMethod(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input, other):
                return input.min(other)

        inputs = [torch.randn(3, 4), torch.randn(3, 4)]
        self.run_test(MinMethod(), inputs, expected_ops={acc_ops.minimum})

if __name__ == '__main__':
    run_tests()
