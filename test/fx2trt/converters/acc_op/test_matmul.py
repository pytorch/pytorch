# Owner(s): ["oncall: fx"]

import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
import torch.nn as nn
from torch.testing._internal.common_fx2trt import AccTestCase
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests


class TestMatMulConverter(AccTestCase):
    @parameterized.expand(
        [
            ("2_2", (2, 3), (3, 2)),
            ("2_1", (2, 3), (3,)),
            ("4_2", (1, 2, 2, 3), (3, 2)),
            ("1_2", (3,), (3, 2)),
        ]
    )
    def test_matmul_other_constant(self, _, input_shape, other_shape):
        class MatMul(nn.Module):
            def __init__(self):
                super().__init__()
                self.other = nn.Parameter(torch.randn(*other_shape))

            def forward(self, input):
                return torch.matmul(input, self.other)

        inputs = [torch.randn(*input_shape)]
        self.run_test(
            MatMul(),
            inputs,
            expected_ops={acc_ops.matmul},
            test_implicit_batch_dim=(len(input_shape) > 1),
        )

    @parameterized.expand(
        [
            ("2_2", (2, 3), (3, 2)),
            ("1_2", (3,), (3, 2)),
            ("3_4", (2, 2, 3), (3, 1, 3, 3)),
        ]
    )
    def test_matmul_input_constant(self, _, input_shape, other_shape):
        class MatMul(nn.Module):
            def __init__(self):
                super().__init__()
                self.input = nn.Parameter(torch.randn(*input_shape))

            def forward(self, other):
                return torch.matmul(self.input, other)

        inputs = [torch.randn(*other_shape)]
        self.run_test(
            MatMul(),
            inputs,
            expected_ops={acc_ops.matmul},
            test_implicit_batch_dim=(len(other_shape) > 2),
        )

    @parameterized.expand(
        [
            ("4_4", (2, 2, 2, 3), (2, 1, 3, 2)),
            ("4_2", (2, 1, 2, 3), (3, 2)),
            ("2_3", (2, 3), (2, 3, 4)),
            ("2_2", (2, 3), (3, 2)),
            ("2_1", (2, 3), (3,)),
            ("1_2", (3,), (3, 2)),
            ("1_1", (3,), (3,)),
        ]
    )
    def test_matmul(self, _, input_shape, other_shape):
        class MatMul(nn.Module):
            def forward(self, input, other):
                return torch.matmul(input, other)

        inputs = [torch.randn(*input_shape), torch.randn(*other_shape)]
        test_implicit_batch_dim = (
            input_shape[0] == other_shape[0]
            and len(input_shape) > 2
            and len(other_shape) > 2
        )
        self.run_test(
            MatMul(),
            inputs,
            expected_ops={acc_ops.matmul},
            test_implicit_batch_dim=test_implicit_batch_dim,
        )

if __name__ == '__main__':
    run_tests()
