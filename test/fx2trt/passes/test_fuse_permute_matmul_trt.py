import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
from torch.testing._internal.common_fx2trt import AccTestCase
from parameterized import parameterized, param
from torch.fx.experimental.fx2trt.passes.fuse_pass import (
    fuse_permute_matmul,
    trt_transposed_matmul,
)
from torch.testing._internal.common_utils import run_tests


def tranpose_last_two_dims(x):
    return x.transpose(-1, -2)


def permute021(x):
    return x.permute(0, 2, 1)


class TestFusePermuteMatmul(AccTestCase):
    @parameterized.expand(
        [
            ("transpose_lhs_bmm", (3, 3, 2), (3, 3, 4), tranpose_last_two_dims),
            param(
                "transpose_rhs_bmm", (3, 2, 3), (3, 4, 3), rhs_op=tranpose_last_two_dims
            ),
            ("permute_lhs_bmm", (3, 3, 2), (3, 3, 4), permute021),
            param("permute_rhs_bmm", (3, 2, 3), (3, 4, 3), rhs_op=permute021),
            ("permute_both_bmm", (3, 3, 2), (3, 4, 3), permute021, permute021),
            (
                "permute_both_matmul",
                (3, 2, 3, 2),
                (3, 2, 4, 3),
                lambda x: x.permute(0, 1, 3, 2),
                lambda x: x.permute(0, 1, 3, 2),
                torch.matmul,
            ),
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
        class TestModule(torch.nn.Module):
            def forward(self, x, y):
                return op(lhs_op(x), rhs_op(y))

        inputs = [torch.randn(*lhs_shape), torch.randn(*rhs_shape)]
        self.run_test(
            TestModule(),
            inputs,
            {trt_transposed_matmul},
            apply_passes=[fuse_permute_matmul],
        )

    @parameterized.expand(
        [
            ("permute_both_bmm", (3, 3, 2), (3, 4, 3), permute021, permute021),
        ]
    )
    def test_fuse_permute_matmul_keep_permute(
        self,
        _,
        lhs_shape,
        rhs_shape,
        lhs_op=lambda x: x,
        rhs_op=lambda x: x,
        op=torch.bmm,
    ):
        """
        Fusion permute while keep permute node which has more than one consumers
        """

        class TestModule(torch.nn.Module):
            def forward(self, x, y):
                z = lhs_op(x)
                return op(z, rhs_op(y)), z

        inputs = [torch.randn(*lhs_shape), torch.randn(*rhs_shape)]
        self.run_test(
            TestModule(),
            inputs,
            {trt_transposed_matmul, acc_ops.permute},
            apply_passes=[fuse_permute_matmul],
        )

    @parameterized.expand(
        [
            ("permute_both_bmm", (3, 3, 2), (3, 4, 3), (3, 4, 3)),
        ]
    )
    def test_multifuse_permute_matmul(
        self,
        _,
        x_shape,
        y_shape,
        z_shape,
    ):
        """
        Test cases when we have multiple bmm users of one permute
        """

        class TestModule(torch.nn.Module):
            def forward(self, x, y, z):
                x = permute021(x)
                y = permute021(y)
                z = permute021(z)
                return torch.bmm(x, y) + torch.bmm(x, z)

        inputs = [torch.randn(*x_shape), torch.randn(*y_shape), torch.randn(*z_shape)]
        self.run_test(
            TestModule(),
            inputs,
            {trt_transposed_matmul},
            apply_passes=[fuse_permute_matmul],
        )

if __name__ == '__main__':
    run_tests()
