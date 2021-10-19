# Owner(s): ["oncall: fx"]

import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
import torch.nn as nn
from caffe2.torch.fb.fx2trt.tests.test_utils import AccTestCase, InputTensorSpec
from parameterized import parameterized


class TestSoftmaxConverter(AccTestCase):
    @parameterized.expand(
        [
            ("none_dim", None),
            ("basic", 1),
            ("batch_dim", 0),
            ("negative_dim", -2)
        ]
    )
    def test_softmax(self, _, dim):
        class Softmax(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, x):
                return nn.functional.softmax(x, dim=self.dim)

        inputs = [torch.randn(1, 2, 3, 4)]
        self.run_test(
            Softmax(dim),
            inputs,
            expected_ops={acc_ops.softmax},
            test_implicit_batch_dim=(dim % len(inputs[0].shape) != 0),
        )

    def test_softmax_with_dynamic_shape(self):
        class Softmax(nn.Module):
            def forward(self, x):
                return nn.functional.softmax(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1), (1, 2, 3), (3, 3, 3))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            Softmax(), input_specs, expected_ops={acc_ops.softmax}
        )

    def test_softmax_with_implicit_batch_dim0_fail(self):
        class Softmax(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return nn.functional.softmax(x, dim=0)

        inputs = [torch.randn(1, 2, 3, 4)]
        self.run_test_with_assert_error(
            Softmax(),
            inputs,
            expect_error=AssertionError,
            test_explicit_batch_dim=False,
        )
