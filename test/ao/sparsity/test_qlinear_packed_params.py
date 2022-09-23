#!/usr/bin/env python3
# Owner(s): ["oncall: mobile"]

import tempfile
import torch
from torch.ao.nn.sparse.quantized.dynamic.linear import Linear
from torch.testing._internal.common_quantized import (
    qengine_is_qnnpack,
    override_quantized_engine,
    override_cpu_allocator_for_qnnpack
)
from torch.testing._internal.common_utils import TestCase

class TestQlinearPackedParams(TestCase):
    def test_qlinear_packed_params(self, allow_non_zero_zero_points=False):
        # copied from https://pytorch.org/docs/stable/sparse.html#csr-tensor-operations,
        # so row/col block indices match that example, but with blocks and
        # scaled rows
        weight_fp32 = torch.Tensor([
            [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0],
            [6, 6, 6, 6, 12, 12, 12, 12, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])

        row_block_size = 1
        col_block_size = 4
        out_features = weight_fp32.shape[0]
        in_features = weight_fp32.shape[1]

        scales = [2.0, 6.0, 12.0]
        zero_points = [
            ((i + 1) if allow_non_zero_zero_points else 0) for i in range(out_features)
        ]
        dtype = torch.qint8

        wide_weight_fp32 = torch.zeros((3, 4008))  # 4000 is tile width for Fbgemm
        wide_weight_fp32[0][0] = 4
        wide_weight_fp32[0][4004] = 6
        wide_weight_fp32[1][0] = 8

        per_tensor_small = (
            torch.quantize_per_tensor(
                weight_fp32,
                scales[0],
                zero_points[0],
                dtype
            ),
            True,
            [0, 1, 3, 3],
            [2, 0, 1],
            [x + (1 if allow_non_zero_zero_points else 0) for x in [
                1, 1, 1, 1, 3, 3, 3, 3, 6, 6, 6, 6
            ]],
        )

        per_channel_small = (
            torch.quantize_per_channel(
                weight_fp32,
                torch.Tensor(scales),
                torch.Tensor(zero_points).to(torch.int),
                0,  # axis = 0
                dtype,
            ),
            False,
            [0, 1, 3, 3],
            [2, 0, 1],
            [x + ([1, 2, 2][i // 4] if allow_non_zero_zero_points else 0) for (i, x) in enumerate([
                1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2
            ])],
        )

        per_tensor_large = (
            torch.quantize_per_tensor(
                wide_weight_fp32,
                scales[0],
                zero_points[0],
                dtype,
            ),
            True,
            [0, 2, 3, 3],
            [0, 1001, 0],
            [x + (1 if allow_non_zero_zero_points else 0) for x in [
                2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0
            ]],
        )

        for (weight, is_per_tensor_quantized, expected_row_block_indices, expected_col_block_indices, expected_weights) in [
            per_tensor_small, per_channel_small, per_tensor_large
        ]:
            lin = Linear(
                out_features=weight.shape[0],
                in_features=weight.shape[1],
                row_block_size=row_block_size,
                col_block_size=col_block_size,
                bias=True,
                dtype=dtype,
            )

            bias = torch.ones(size=(weight.shape[0],))

            lin.set_weight_bias(weight, bias, row_block_size, col_block_size)

            serialized = lin._packed_params._packed_params.__getstate__()

            (
                _,  # version
                bias_,
                out_features_block_size_,
                in_features_block_size_,
                weight_scales_,
                weight_zero_points_,
                quantization_scheme_,
                row_block_indices_,
                col_block_indices_,
                weights_,
                output_channels_,
                input_channels_
            ) = serialized[0]

            # Test Serialization
            self.assertEqual(bias_, bias)
            self.assertEqual(out_features_block_size_, row_block_size)
            self.assertEqual(in_features_block_size_, col_block_size)
            self.assertEqual(weight_scales_, [scales[0]] if is_per_tensor_quantized else scales)
            self.assertEqual(weight_zero_points_, [zero_points[0]] if is_per_tensor_quantized else zero_points)
            self.assertEqual(quantization_scheme_, is_per_tensor_quantized)
            self.assertEqual(row_block_indices_, expected_row_block_indices)
            self.assertEqual(col_block_indices_, expected_col_block_indices)
            self.assertEqual(weights_.tolist(), [v + 128 for v in expected_weights])  # weights are serialized as +128
            self.assertEqual(output_channels_, weight.shape[0])
            self.assertEqual(input_channels_, weight.shape[1])

            # Test Unpacking
            (weights_, bias_, out_features_block_size_, in_features_block_size_) = lin._weight_bias()
            self.assertEqual(torch.dequantize(weights_), torch.dequantize(weight))
            self.assertEqual(bias_, bias)
            self.assertEqual(out_features_block_size_, row_block_size)
            self.assertEqual(in_features_block_size_, col_block_size)

            # Test Deserialization
            with tempfile.TemporaryFile() as file_buff:
                torch.save(lin, file_buff)
                file_buff.seek(0)
                lin2 = torch.load(file_buff)
                self.assertEqual(lin._weight_bias(), lin2._weight_bias())
                # Serialize -> Deserialize -> Serialize should match Serialize
                self.assertEqual(serialized, lin2._packed_params._packed_params.__getstate__())

                # Test that op output is preserved by serialize -> deserialize
                if qengine_is_qnnpack():
                    x = torch.rand(size=(1, weight.shape[1]))
                    y1 = lin(x)
                    y2 = lin2(x)
                    self.assertEqual(y1, y2)


    def test_qlinear_packed_params_qnnpack(self):
        torch.manual_seed(0)
        with override_quantized_engine('qnnpack'):
            with override_cpu_allocator_for_qnnpack(qengine_is_qnnpack()):
                self.test_qlinear_packed_params(allow_non_zero_zero_points=True)
