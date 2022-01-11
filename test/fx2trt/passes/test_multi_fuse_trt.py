import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
from torch.testing._internal.common_fx2trt import AccTestCase
from parameterized import parameterized
from torch.fx.experimental.fx2trt.passes.fuse_pass import (
    fuse_permute_linear,
    trt_transposed_linear,
    fuse_permute_matmul,
    trt_transposed_matmul,
)
from torch.testing._internal.common_utils import run_tests


def permute021(x):
    return x.permute(0, 2, 1)


class TestMultiFuse(AccTestCase):
    @parameterized.expand(
        [
            ("permute_both_bmm", (3, 3, 2), (3, 4, 3), permute021, permute021),
        ]
    )
    def test_fuse_permute_matmul(
        self,
        _,
        lhs_shape,
        rhs_shape,
        lhs_op=lambda x: x,
        rhs_op=lambda x: x,
        op=torch.bmm,
    ):
        """
        Module: permute1 with linear and matmul, permute2 with matmul.
                Permute1  permute2
                |      | |
            linear    matmul
        Fusion should crete pass fuse_permute_matmul and fuse_permute_linear, and eliminate both
        permute node.
        """

        class TestModule(torch.nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features)

            def forward(self, x, y):
                z = lhs_op(x)
                bmm = op(z, rhs_op(y))
                linear = self.linear(z)
                return (bmm, linear)

        inputs = [torch.randn(*lhs_shape), torch.randn(*rhs_shape)]
        self.run_test(
            TestModule(3, 6),
            inputs,
            {trt_transposed_matmul, trt_transposed_linear},
            {acc_ops.permute},
            apply_passes=[fuse_permute_matmul, fuse_permute_linear],
        )

if __name__ == '__main__':
    run_tests()
