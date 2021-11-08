# Owner(s): ["oncall: fx"]

from typing import Callable

import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
import torch.nn as nn
from caffe2.torch.fb.fx2trt.tests.test_utils import AccTestCase
from parameterized import parameterized

unary_ops = [
    (torch.sin, acc_ops.sin),
    (torch.cos, acc_ops.cos),
    (torch.tan, acc_ops.tan),
    (torch.sinh, acc_ops.sinh),
    (torch.cosh, acc_ops.cosh),
    (torch.asin, acc_ops.asin),
    (torch.acos, acc_ops.acos),
    (torch.atan, acc_ops.atan),
    (torch.abs, acc_ops.abs),
    (torch.neg, acc_ops.neg),
    (torch.reciprocal, acc_ops.reciprocal),
    (torch.sqrt, acc_ops.sqrt),
    (torch.log, acc_ops.log),
    (torch.exp, acc_ops.exp),
    (torch.floor, acc_ops.floor),
    (torch.ceil, acc_ops.ceil),
]


class TestUnaryOpConverters(AccTestCase):
    @parameterized.expand([(op[1].__name__, op[0], op[1]) for op in unary_ops])
    def test_unary_ops(self, name, orig_op: Callable, expected_op):
        class TestModule(nn.Module):
            def __init__(self, orig_op):
                super().__init__()
                self.orig_op = orig_op

            def forward(self, x):
                return self.orig_op(x)

        m = TestModule(orig_op)
        inputs = [torch.randn(2, 2, 3)]
        self.run_test(m, inputs, expected_ops={expected_op})
