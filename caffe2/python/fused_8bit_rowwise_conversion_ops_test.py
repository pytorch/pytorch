from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu

import numpy as np
import struct
from hypothesis import given

# Eigen/Python round 0.5 away from 0, Numpy rounds to even
round_to_nearest = np.vectorize(round)


def bytes_to_floats(byte_matrix):
    floats = np.empty([np.shape(byte_matrix)[0], 1], dtype=np.float32)
    for i, byte_values in enumerate(byte_matrix):
        floats[i], = struct.unpack('f', bytearray(byte_values))
    return floats


def floats_to_bytes(floats):
    byte_matrix = np.empty([np.shape(floats)[0], 4], dtype=np.uint8)
    for i, value in enumerate(floats):
        assert isinstance(value, np.float32), (value, floats)
        as_bytes = struct.pack('f', value)
        # In Python3 bytes will be a list of int, in Python2 a list of string
        if isinstance(as_bytes[0], int):
            byte_matrix[i] = list(as_bytes)
        else:
            byte_matrix[i] = list(map(ord, as_bytes))
    return byte_matrix


def fused_rowwise_8bit_quantize_reference(data):
    minimum = np.min(data, axis=-1, keepdims=True)
    maximum = np.max(data, axis=-1, keepdims=True)
    span = maximum - minimum
    bias = minimum
    scale = span / 255.0
    inverse_scale = 255.0 / (span + 1e-8)
    quantized_data = round_to_nearest((data - bias) * inverse_scale)
    scale_bytes = floats_to_bytes(scale.reshape(-1))
    scale_bytes = scale_bytes.reshape(data.shape[:-1] + (scale_bytes.shape[-1],))
    bias_bytes = floats_to_bytes(bias.reshape(-1))
    bias_bytes = bias_bytes.reshape(data.shape[:-1] + (bias_bytes.shape[-1],))
    print(quantized_data.shape, scale.shape, scale_bytes.shape, bias.shape, bias_bytes.shape)
    return np.concatenate([quantized_data, scale_bytes, bias_bytes], axis=-1)


def fused_rowwise_8bit_quantize_dequantize_reference(data):
    fused_quantized = fused_rowwise_8bit_quantize_reference(data)
    scale = bytes_to_floats(fused_quantized[..., -8:-4].astype(np.uint8).reshape(-1, 4))
    scale = scale.reshape(fused_quantized.shape[:-1] + (scale.shape[-1],))
    bias = bytes_to_floats(fused_quantized[..., -4:].astype(np.uint8).reshape(-1, 4))
    bias = bias.reshape(fused_quantized.shape[:-1] + (bias.shape[-1],))
    quantized_data = fused_quantized[..., :-8]
    return quantized_data * scale + bias


class TestFused8BitRowwiseQuantizationConversion(hu.HypothesisTestCase):
    @given(input_data=hu.tensor(min_dim=1, max_dim=3, max_value=33))
    def test_quantize_op(self, input_data):
        quantize = core.CreateOperator(
            'FloatToFused8BitRowwiseQuantized',
            ['input_data'],
            ['quantized_data'],
        )
        workspace.FeedBlob('input_data', input_data)
        workspace.RunOperatorOnce(quantize)

        quantized_data = workspace.FetchBlob('quantized_data')

        reference = fused_rowwise_8bit_quantize_reference(
            input_data.astype(np.float32)
        )
        np.testing.assert_array_almost_equal(quantized_data, reference)

    @given(input_data=hu.tensor(min_dim=1, max_dim=3, max_value=33))
    def test_quantize_and_dequantize_op(self, input_data):
        quantize = core.CreateOperator(
            'FloatToFused8BitRowwiseQuantized',
            ['input_data'],
            ['quantized_data'],
        )
        workspace.FeedBlob('input_data', input_data)
        workspace.RunOperatorOnce(quantize)

        quantized_data = workspace.FetchBlob('quantized_data')

        dequantize = core.CreateOperator(
            'Fused8BitRowwiseQuantizedToFloat',
            ['quantized_data'],
            ['dequantized_data'],
        )
        workspace.FeedBlob('quantized_data', quantized_data)
        workspace.RunOperatorOnce(dequantize)

        dequantized_data = workspace.FetchBlob('dequantized_data')

        reference = fused_rowwise_8bit_quantize_dequantize_reference(input_data)
        np.testing.assert_array_almost_equal(dequantized_data, reference)
