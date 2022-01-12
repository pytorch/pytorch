# Owner(s): ["oncall: fx"]

from typing import Callable

import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
import torch.nn as nn
from torch.testing._internal.common_fx2trt import AccTestCase, InputTensorSpec
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

elementwise_ops = [
    ((lambda x, y: x + y), acc_ops.add),
    ((lambda x, y: x - y), acc_ops.sub),
    ((lambda x, y: x / y), acc_ops.div),
    ((lambda x, y: x // y), acc_ops.floor_div),
    ((lambda x, y: torch.div(x, y, rounding_mode="trunc")), acc_ops.trunc_div),
    ((lambda x, y: torch.div(x, y, rounding_mode="floor")), acc_ops.floor_div),
    ((lambda x, y: torch.div(x, y)), acc_ops.div),
    # torch.floor_divide rounds result toward zero, rather than -Inf.
    # https://github.com/pytorch/pytorch/issues/43874
    ((lambda x, y: torch.floor_divide(x, y)), acc_ops.trunc_div),
    ((lambda x, y: x * y), acc_ops.mul),
    (torch.pow, acc_ops.pow),
]


class TestBinaryOpConverters(AccTestCase):
    @parameterized.expand([(op[1].__name__, op[0], op[1]) for op in elementwise_ops])
    def test_elementwise_ops(self, name, orig_op: Callable, expected_op):
        class TestModule(nn.Module):
            def __init__(self, orig_op):
                super().__init__()
                self.orig_op = orig_op

            def forward(self, x):
                return self.orig_op(x, x)

        m = TestModule(orig_op)
        # Avoid dividing by 0.
        inputs = [torch.rand(1, 1) + 1]
        self.run_test(m, inputs, expected_ops={expected_op})

    @parameterized.expand([(op[1].__name__, op[0], op[1]) for op in elementwise_ops])
    def test_elementwise_ops_constant(self, name, orig_op: Callable, expected_op):
        class TestModule(nn.Module):
            def __init__(self, orig_op):
                super().__init__()
                self.constant = torch.randn(1)
                self.orig_op = orig_op

            def forward(self, x):
                x = self.orig_op(x, self.constant)
                return self.orig_op(x, -2)

        m = TestModule(orig_op)
        inputs = [torch.randn(2, 2)]
        self.run_test(m, inputs, expected_ops={expected_op})

    @parameterized.expand(
        [
            (
                f"no_broadcast_{op[1].__name__}",
                (-1, -1),
                ((1, 1), (2, 2), (3, 3)),
                (-1, -1),
                ((1, 1), (2, 2), (3, 3)),
                op[0],
                op[1],
            )
            for op in elementwise_ops
        ]
        + [
            (
                f"broadcast_{op[1].__name__}",
                (-1, -1, -1),
                ((1, 1, 1), (2, 2, 2), (3, 3, 3)),
                (-1, -1),
                ((1, 1), (2, 2), (3, 3)),
                op[0],
                op[1],
            )
            for op in elementwise_ops
        ]
    )
    def test_elementwise_op_with_dynamic_shape(
        self, _, x_shape, x_shape_ranges, y_shape, y_shape_ranges, orig_op, expected_op
    ):
        class Op(nn.Module):
            def forward(self, x, y):
                return orig_op(x, y)

        input_specs = [
            InputTensorSpec(
                shape=x_shape,
                dtype=torch.float32,
                shape_ranges=[x_shape_ranges],
            ),
            InputTensorSpec(
                shape=y_shape,
                dtype=torch.float32,
                shape_ranges=[y_shape_ranges],
            ),
        ]

        self.run_test_with_dynamic_shape(Op(), input_specs, expected_ops={expected_op})

if __name__ == '__main__':
    run_tests()
