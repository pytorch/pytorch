import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
from torch.testing._internal.common_fx2trt import AccTestCase
from parameterized import parameterized, param


class TestLinalgNormConverter(AccTestCase):
    @parameterized.expand(
        [
            param("1_matrix", input_shape=[2, 2], ord=1, dim=[1], keepdims=False),
            param("2_matrix", input_shape=[2, 2], ord=2, dim=[1], keepdims=False),
            param("-1_matrix", input_shape=[2, 10], ord=-1, dim=[1], keepdims=False),
            param("-2_matrix", input_shape=[2, 10], ord=2, dim=[1], keepdims=False),
            param(
                "1_matrix_keepdim", input_shape=[2, 2], ord=1, dim=[1], keepdims=True
            ),
            param(
                "2_matrix_keepdim", input_shape=[2, 2], ord=2, dim=[1], keepdims=True
            ),
            param(
                "fro_matrix",
                input_shape=[2, 2, 2],
                ord="fro",
                dim=[1, 2],
                keepdims=False,
            ),
            param(
                "nuc_matrix",
                input_shape=[2, 2, 2],
                ord="nuc",
                dim=[1, 2],
                keepdims=False,
            ),
            param(
                "nuc_matrix_keepdim",
                input_shape=[2, 2, 2],
                ord="nuc",
                dim=[1, 2],
                keepdims=True,
            ),
            param(
                "inf_matrix",
                input_shape=[2, 10],
                ord=float("inf"),
                dim=[1],
                keepdims=False,
            ),
            param(
                "-inf_matrix",
                input_shape=[2, 10],
                ord=float("-inf"),
                dim=[1],
                keepdims=False,
            ),
            param(
                "-inf_matrix_keepdim",
                input_shape=[2, 10],
                ord=float("-inf"),
                dim=[1],
                keepdims=True,
            ),
        ]
    )
    def test_linalg_norm(
        self, test_name, input_shape, ord=None, dim=None, keepdims=False
    ):
        class LinalgNorm(torch.nn.Module):
            def __init__(self, ord, dim, keepdims):
                super().__init__()

            def forward(self, x):
                return torch.linalg.norm(x, ord, dim, keepdims)

        inputs = [torch.randn(input_shape)]

        self.run_test(
            LinalgNorm(ord, dim, keepdims),
            inputs,
            expected_ops={acc_ops.linalg_norm},
            test_implicit_batch_dim=False,
        )

    # @parameterized.expand(
    #     [
    #         param(
    #             "l2_norm_dim_3",
    #             input_shape=[1, 512, 40, 40],
    #             ord=2,
    #             dim=3,
    #             keepdims=False,
    #         ),
    #         param(
    #             "l2_norm_dim_2",
    #             input_shape=[1, 512, 40, 40],
    #             ord=2,
    #             dim=2,
    #             keepdims=False,
    #         ),
    #         param(
    #             "l2_norm_dim_1",
    #             input_shape=[1, 512, 40, 40],
    #             ord=2,
    #             dim=1,
    #             keepdims=True,
    #         ),
    #     ]
    # )
    # def test_l2_norm(self, test_name, input_shape, ord=None, dim=None, keepdims=False):
    #     class LinalgNorm(torch.nn.Module):
    #         def __init__(self, ord, dim, keepdims):
    #             super().__init__()

    #         def forward(self, x):
    #             return torch.linalg.norm(x, ord, dim, keepdims)

    #     inputs = [torch.randn(input_shape)]

    #     self.run_test(
    #         LinalgNorm(ord, dim, keepdims),
    #         inputs,
    #         expected_ops={acc_ops.linalg_norm},
    #     )
