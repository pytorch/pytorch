# Owner(s): ["oncall: fx"]

import torch
import torch.fx
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
import torch.nn as nn
from torch.testing._internal.common_fx2trt import AccTestCase, InputTensorSpec
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests


class TestUnsqueeze(AccTestCase):
    @parameterized.expand(
        [
            ("negative_dim", -2),
            ("positive_dim", 2),
        ]
    )
    def test_unsqueeze(self, _, dim):
        class Unsqueeze(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, x):
                return torch.unsqueeze(x, self.dim)

        inputs = [torch.randn(1, 2, 3)]
        self.run_test(Unsqueeze(dim), inputs, expected_ops={acc_ops.unsqueeze})

    @parameterized.expand(
        [
            ("negative_dim_dynamic", -4),
            ("positive_dim_dynamic", 1),
        ]
    )
    def test_unsqueeze_with_dynamic_shape(self, _, dim):
        class Unsqueeze(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, x):
                return torch.unsqueeze(x, self.dim)

        input_specs = [
            InputTensorSpec(
                shape=(-1, 2, 3),
                dtype=torch.float32,
                shape_ranges=[((1, 2, 3), (2, 2, 3), (3, 2, 3))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            Unsqueeze(dim), input_specs, expected_ops={acc_ops.unsqueeze}
        )

if __name__ == '__main__':
    run_tests()
