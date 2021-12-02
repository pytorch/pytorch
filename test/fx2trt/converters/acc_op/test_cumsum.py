import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
import torch.nn as nn
from torch.testing._internal.common_fx2trt import AccTestCase
from parameterized import parameterized
from torch.fx.experimental.fx2trt.fx2trt import InputTensorSpec


class TestCumsumConverter(AccTestCase):
    @parameterized.expand(
        [
            ("cumsum", 0),
            ("cumsum", -2),
            ("cumsum", 2),
        ]
    )
    def test_cumsum(self, _, dim):
        class Cumsum(nn.Module):
            def forward(self, x):
                return torch.cumsum(x, dim)[0]

        inputs = [torch.rand(3, 5, 10)]
        self.run_test(
            Cumsum(),
            inputs,
            expected_ops={acc_ops.cumsum},
            test_implicit_batch_dim=False,
        )

    def test_elementwise_op_with_dynamic_shape(self):
        class Op(nn.Module):
            def forward(self, x):
                return torch.cumsum(x, dim=0)

        input_specs = [
            InputTensorSpec(
                shape=(-1, 2, 3),
                dtype=torch.float32,
                shape_ranges=[((1, 2, 3), (4, 2, 3), (100, 2, 3))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            Op(), input_specs, expected_ops={acc_ops.cumsum}
        )
