# Owner(s): ["oncall: fx"]

import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
from torch.testing._internal.common_fx2trt import AccTestCase
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

reduce_ops = [(torch.sum, acc_ops.sum), (torch.mean, acc_ops.mean)]

class TestReduceConverter(AccTestCase):
    @parameterized.expand(
        case
        for op, acc_op in reduce_ops
        for case in
        [
            (f"{acc_op.__name__}_single_dim_no_keepdim", 1, False, op, acc_op),
            (f"{acc_op.__name__}_single_dim_keepdim", 1, True, op, acc_op),
            (f"{acc_op.__name__}_two_dim_no_keepdim", (1, 2), False, op, acc_op),
            (f"{acc_op.__name__}_two_dim_keepdim", (1, 2), True, op, acc_op),
            (f"{acc_op.__name__}_three_dim_no_keepdim", (1, 2, 3), False, op, acc_op),
            (f"{acc_op.__name__}_three_dim_keepdim", (1, 2, 3), True, op, acc_op),
            (f"{acc_op.__name__}_dim0_keepdim", 0, True, op, acc_op),
            (f"{acc_op.__name__}_dim0_no_keepdim", 0, False, op, acc_op),
        ]
    )
    def test_reduce(self, test_name, dim, keepdim, op, expected_acc_op):
        class Reduce(torch.nn.Module):
            def __init__(self, dim, keepdim):
                super().__init__()
                self.dim = dim
                self.keepdim = keepdim

            def forward(self, x):
                return op(x, dim=self.dim, keepdim=self.keepdim)

        inputs = [torch.randn(1, 2, 3, 4)]
        self.run_test(
            Reduce(dim, keepdim),
            inputs,
            expected_ops={expected_acc_op},
            test_implicit_batch_dim=(dim != 0),
        )

    @parameterized.expand(
        [
            (f"{acc_op.__name__}_no_dim_no_keepdim", op, acc_op) for op, acc_op in reduce_ops
        ]
    )
    def test_reduce_all_dims(
        self,
        test_name,
        op,
        expected_acc_op,
    ):
        class Reduce(torch.nn.Module):
            def forward(self, x):
                return op(x)

        inputs = [torch.randn(1, 2, 3, 4)]
        self.run_test(
            Reduce(),
            inputs,
            expected_ops={expected_acc_op},
            test_implicit_batch_dim=False,
        )

if __name__ == '__main__':
    run_tests()
