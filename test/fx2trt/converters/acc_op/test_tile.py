# Owner(s): ["oncall: fx"]

import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
import torch.nn as nn
from torch.testing._internal.common_fx2trt import AccTestCase, InputTensorSpec
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests


class TestTile(AccTestCase):
    @parameterized.expand(
        [
            ("same_num_dims", (2, 2, 3), (1, 2, 2)),
            ("less_dims", (2, 2, 3), (2,)),
            ("more_dims", (2, 3), (1, 2, 2, 1)),
        ]
    )
    def test_tile(self, _, input_shape, dims):
        class Tile(nn.Module):
            def __init__(self, dims):
                super().__init__()
                self.dims = dims

            def forward(self, x):
                return torch.tile(x, self.dims)

        inputs = [torch.randn(*input_shape)]
        self.run_test(
            Tile(dims),
            inputs,
            expected_ops={acc_ops.tile},
            test_implicit_batch_dim=(
                len(input_shape) > len(dims)
                or (len(input_shape) == len(dims) and dims[0] == 1)
            ),
        )

    @parameterized.expand(
        [
            ("same_num_dims", (-1, 2, 3), (1, 2, 2)),
            ("less_dims", (-1, 2, 3), (2,)),
            ("more_dims", (-1, 3), (1, 2, 2, 1)),
            ("all_dynamic_dim", (-1, -1), (1, 2, 2, 1)),
        ]
    )
    def test_tile_with_dynamic_shape(self, _, shape, dims):
        class Tile(nn.Module):
            def __init__(self, dims):
                super().__init__()
                self.dims = dims

            def forward(self, x):
                return torch.tile(x, self.dims)

        input_specs = [
            InputTensorSpec(
                shape=shape,
                dtype=torch.float32,
                shape_ranges=[
                    (
                        tuple(i if i != -1 else 1 for i in shape),
                        tuple(i if i != -1 else 2 for i in shape),
                        tuple(i if i != -1 else 3 for i in shape),
                    )
                ],
            ),
        ]
        self.run_test_with_dynamic_shape(
            Tile(dims), input_specs, expected_ops={acc_ops.tile}
        )

if __name__ == '__main__':
    run_tests()
