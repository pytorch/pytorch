# Owner(s): ["oncall: fx"]

import torch
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
import torch.nn as nn
from torch.testing._internal.common_fx2trt import AccTestCase, InputTensorSpec
from parameterized import parameterized, param
from torch.testing._internal.common_utils import run_tests


class TestLayerNormConverter(AccTestCase):
    @parameterized.expand(
        [
            param("1d_normalized_shape", [10], [2, 10]),
            param("2d_normalized_shape", [5, 10], [2, 5, 10]),
            param("4d_input_shape", [5, 10], [2, 8, 5, 10]),
        ]
    )
    def test_layer_norm(self, _, normalized_shape, input_shape):
        class LayerNorm(nn.Module):
            def __init__(self, normalized_shape):
                super().__init__()
                self.mod = nn.LayerNorm(normalized_shape, eps=1e-02)

            def forward(self, x):
                return self.mod(x)

        inputs = [torch.randn(input_shape)]
        self.run_test(
            LayerNorm(normalized_shape),
            inputs,
            expected_ops={acc_ops.layer_norm},
        )

    @parameterized.expand(
        [
            param("1d_normalized_shape", [10], (10,)),
            param("2d_normalized_shape", [5, 10], (5, 10)),
            param("4d_input_shape", [5, 10], (8, 5, 10)),
        ]
    )
    def test_layer_norm_with_dynamic_shape(self, _, normalized_shape, input_shape):
        class LayerNorm(nn.Module):
            def __init__(self, normalized_shape):
                super().__init__()
                self.mod = nn.LayerNorm(normalized_shape, eps=1e-02)

            def forward(self, x):
                return self.mod(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1,) + input_shape,
                dtype=torch.float32,
                shape_ranges=[
                    ((1,) + input_shape, (4,) + input_shape, (10,) + input_shape)
                ],
            ),
        ]
        self.run_test_with_dynamic_shape(
            LayerNorm(normalized_shape), input_specs, expected_ops={acc_ops.layer_norm}
        )

if __name__ == '__main__':
    run_tests()
