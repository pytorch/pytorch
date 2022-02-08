# Owner(s): ["oncall: aiacc"]

import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
from torch.testing._internal.common_fx2trt import AccTestCase
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

# NOTE torch.prod will only accept one dim unlike other reduce ops which accept tuples

class TestProdConverter(AccTestCase):
    @parameterized.expand(
        [
            (f"{acc_ops.prod.__name__}_dim0_keepdim", 0, True, torch.prod, acc_ops.prod),
            (f"{acc_ops.prod.__name__}_dim0_no_keepdim", 0, False, torch.prod, acc_ops.prod),
            (f"{acc_ops.prod.__name__}_dim1_keepdim", 1, True, torch.prod, acc_ops.prod),
            (f"{acc_ops.prod.__name__}_dim1_no_keepdim", 1, False, torch.prod, acc_ops.prod),
            (f"{acc_ops.prod.__name__}_dim1_keepdim", 2, True, torch.prod, acc_ops.prod),
            (f"{acc_ops.prod.__name__}_dim1_no_keepdim", 2, False, torch.prod, acc_ops.prod),
        ]
    )
    def test_prod(self, test_name, dim, keepdim, op, expected_acc_op):
        class Prod(torch.nn.Module):
            def __init__(self, dim, keepdim):
                super().__init__()
                self.dim = dim
                self.keepdim = keepdim

            def forward(self, x):
                return op(x, dim=self.dim, keepdim=self.keepdim)

        inputs = [torch.randn(1, 2, 3, 4)]
        self.run_test(
            Prod(dim, keepdim),
            inputs,
            expected_ops={expected_acc_op},
            test_implicit_batch_dim=(dim != 0),
        )

    @parameterized.expand(
        [
            (f"{acc_ops.prod.__name__}_no_dim_no_keepdim", torch.prod, acc_ops.prod)
        ]
    )
    def test_prod_all_dims(
        self,
        test_name,
        op,
        expected_acc_op,
    ):
        class Prod(torch.nn.Module):
            def forward(self, x):
                return op(x)

        inputs = [torch.randn(1, 2, 3, 4)]
        self.run_test(
            Prod(),
            inputs,
            expected_ops={expected_acc_op},
            test_implicit_batch_dim=False,
        )

if __name__ == '__main__':
    run_tests()
