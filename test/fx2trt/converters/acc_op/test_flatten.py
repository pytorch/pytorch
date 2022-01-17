# Owner(s): ["oncall: fx"]

import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
import torch.nn as nn
from torch.testing._internal.common_fx2trt import AccTestCase, InputTensorSpec
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests


class TestFlattenConverter(AccTestCase):
    @parameterized.expand(
        [
            ("flatten_middle_dims", 1, 2),
            ("flatten_last_3_dims", 1, 3),
            ("flatten_last_1", 3, 3),
            ("flatten_all", 0, 3),
        ]
    )
    def test_flatten(self, _, start_dim, end_dim):
        class Flatten(nn.Module):
            def __init__(self, start, end):
                super().__init__()
                self.start = start
                self.end = end

            def forward(self, x):
                return torch.flatten(x, self.start, self.end)

        inputs = [torch.randn(1, 2, 3, 1)]
        self.run_test(
            Flatten(start_dim, end_dim),
            inputs,
            expected_ops={acc_ops.flatten},
            test_implicit_batch_dim=(start_dim != 0),
        )

    @parameterized.expand(
        [
            ("flatten_middle_dims", 1, 2),
            ("flatten_last_3_dims", 2, 4),
            ("flatten_last_1", 4, 4),
            ("flatten_first_2", 0, 1),
            ("flatten_all", 0, 4),
        ]
    )
    def test_flatten_with_dynamic_shape(self, _, start_dim, end_dim):
        class Flatten(nn.Module):
            def __init__(self, start, end):
                super().__init__()
                self.start = start
                self.end = end

            def forward(self, x):
                return torch.flatten(x, self.start, self.end)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1, 1), (1, 2, 3, 2, 1), (3, 3, 3, 3, 3))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            Flatten(start_dim, end_dim),
            input_specs,
            expected_ops={acc_ops.flatten},
        )

if __name__ == '__main__':
    run_tests()
