

import math
import struct

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, workspace
from caffe2.python.operator_test.fused_nbit_rowwise_test_helper import (
    _compress_uniform_simplified,
    param_search_greedy,
)
from hypothesis import assume, given, settings


# Eigen/Python round 0.5 away from 0, Numpy rounds to even
round_to_nearest = np.vectorize(round)


def bytes_to_half_floats(byte_matrix):
    floats = np.empty([np.shape(byte_matrix)[0], 1], dtype=np.float16)
    for i, byte_values in enumerate(byte_matrix):
        (floats[i],) = np.frombuffer(
            memoryview(byte_values).tobytes(), dtype=np.float16
        )
    return floats


def half_floats_to_bytes(floats):
    byte_matrix = np.empty([np.shape(floats)[0], 2], dtype=np.uint8)
    for i, value in enumerate(floats):
        assert isinstance(value, np.float16), (value, floats)
        byte_matrix[i] = np.frombuffer(
            memoryview(np.array([value])).tobytes(), dtype=np.uint8
        )
    return byte_matrix


def int8_to_bytes(int8s):
    byte_matrix = np.empty([np.shape(int8s)[0], 1], dtype=np.uint8)
    for i, value in enumerate(int8s):
        assert isinstance(value, np.int8), (value, int8s)
        as_bytes = struct.pack("b", value)
        # In Python3 bytes will be a list of int, in Python2 a list of string
        if isinstance(as_bytes[0], int):
            byte_matrix[i] = list(as_bytes)
        else:
            byte_matrix[i] = list(map(ord, as_bytes))
    return byte_matrix


def fused_rowwise_nbit_quantize_reference(data, bit):
    minimum = np.min(data, axis=1).astype(np.float16).astype(np.float32)
    maximum = np.max(data, axis=1)
    span = maximum - minimum
    qmax = (1 << bit) - 1
    scale = (span / qmax).astype(np.float16).astype(np.float32)
    bias = np.zeros(data.shape[0])
    quantized_data = np.zeros(data.shape).astype(np.uint8)

    for i in range(data.shape[0]):
        bias[i] = minimum[i]
        inverse_scale = 1.0 if scale[i] == 0.0 else 1 / scale[i]
        if scale[i] == 0.0 or math.isinf(inverse_scale):
            scale[i] = 1.0
            inverse_scale = 1.0
        quantized_data[i] = np.clip(
            np.round((data[i, :] - minimum[i]) * inverse_scale), 0, qmax
        )

    # pack
    assert 8 % bit == 0
    num_elem_per_byte = 8 // bit
    packed_dim = (data.shape[1] + num_elem_per_byte - 1) // num_elem_per_byte
    packed_data = np.zeros([data.shape[0], packed_dim]).astype(np.uint8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if j % num_elem_per_byte == 0:
                packed_data[i, j // num_elem_per_byte] = quantized_data[i, j]
            else:
                packed_data[i, j // num_elem_per_byte] += quantized_data[i, j] << (
                    (j % num_elem_per_byte) * bit
                )

    scale_bytes = half_floats_to_bytes(scale.astype(np.float16))
    bias_bytes = half_floats_to_bytes(bias.astype(np.float16))
    return np.concatenate([packed_data, scale_bytes, bias_bytes], axis=1)


def fused_rowwise_nbit_quantize_dequantize_reference(data, bit):
    fused_quantized = fused_rowwise_nbit_quantize_reference(data, bit)
    scale = bytes_to_half_floats(fused_quantized[:, -4:-2].astype(np.uint8)).astype(
        np.float32
    )
    bias = bytes_to_half_floats(fused_quantized[:, -2:].astype(np.uint8)).astype(
        np.float32
    )
    quantized_data = fused_quantized[:, :-4]

    # unpack
    packed_dim = fused_quantized.shape[1] - 4
    assert 8 % bit == 0
    num_elem_per_byte = 8 // bit
    assert packed_dim == ((data.shape[1] + num_elem_per_byte - 1) // num_elem_per_byte)
    unpacked_data = np.zeros(data.shape).astype(np.uint8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            unpacked_data[i, j] = (
                quantized_data[i, j // num_elem_per_byte]
                >> ((j % num_elem_per_byte) * bit)
            ) & ((1 << bit) - 1)

    return scale * unpacked_data + bias


class TestFusedNBitRowwiseQuantizationConversion(hu.HypothesisTestCase):
    @given(input_data=hu.tensor(min_dim=2, max_dim=2), bit_rate=st.sampled_from([2, 4]))
    def test_quantize_op(self, input_data, bit_rate):
        assert 8 % bit_rate == 0
        num_elem_per_byte = 8 // bit_rate
        assume(input_data.shape[1] % num_elem_per_byte == 0)
        quantize = core.CreateOperator(
            "FloatToFused" + str(bit_rate) + "BitRowwiseQuantized",
            ["input_data"],
            ["quantized_data"],
        )
        workspace.FeedBlob("input_data", input_data)
        workspace.RunOperatorOnce(quantize)

        quantized_data = workspace.FetchBlob("quantized_data")

        reference = fused_rowwise_nbit_quantize_reference(
            input_data.astype(np.float32), bit_rate
        )

        interleaved_dim = input_data.shape[1] // num_elem_per_byte
        # compare quantized data
        np.testing.assert_array_equal(
            quantized_data[:, :interleaved_dim], reference[:, :interleaved_dim]
        )
        # compare scales
        np.testing.assert_array_almost_equal(
            bytes_to_half_floats(
                quantized_data[:, interleaved_dim : interleaved_dim + 2]
            ),
            bytes_to_half_floats(reference[:, interleaved_dim : interleaved_dim + 2]),
        )
        # compare zero points
        np.testing.assert_array_equal(
            quantized_data[:, interleaved_dim + 2], reference[:, interleaved_dim + 2]
        )

    @given(
        batch_size=st.integers(1, 100),
        block_size=st.integers(1, 100),
        bit_rate=st.sampled_from([2, 4]),
    )
    def test_quantize_and_dequantize_op(self, batch_size, block_size, bit_rate):
        assert 8 % bit_rate == 0
        num_elem_per_byte = 8 // bit_rate
        input_data = np.random.rand(batch_size, block_size).astype(np.float32)
        assume(input_data.shape[1] % num_elem_per_byte == 0)
        quantize = core.CreateOperator(
            "FloatToFused" + str(bit_rate) + "BitRowwiseQuantized",
            ["input_data"],
            ["quantized_data"],
        )
        workspace.FeedBlob("input_data", input_data)
        workspace.RunOperatorOnce(quantize)

        quantized_data = workspace.FetchBlob("quantized_data")

        dequantize = core.CreateOperator(
            "Fused" + str(bit_rate) + "BitRowwiseQuantizedToFloat",
            ["quantized_data"],
            ["dequantized_data"],
        )
        workspace.FeedBlob("quantized_data", quantized_data)
        workspace.RunOperatorOnce(dequantize)

        dequantized_data = workspace.FetchBlob("dequantized_data")

        reference = fused_rowwise_nbit_quantize_dequantize_reference(
            input_data, bit_rate
        )
        np.testing.assert_array_almost_equal(dequantized_data, reference)


def ErrorThresholdRow(X, bit_rate):
    # minimum representable error in bit_rate per row
    min_elem = np.min(X, axis=1)
    max_elem = np.max(X, axis=1)

    bias = np.float16(min_elem)
    scale = np.float16((max_elem - bias) / ((1 << bit_rate) - 1))

    max_round_error = scale / 2
    max_clip_error = np.maximum(
        np.abs(min_elem - bias), np.abs(scale * ((1 << bit_rate) - 1) + bias - max_elem)
    )
    thres = np.maximum(max_round_error, max_clip_error) * 1.1
    return thres


class TestNBitFakeFused(hu.HypothesisTestCase):
    @given(bit_rate=st.sampled_from([2, 4]))
    @settings(deadline=1000)
    def testNBit(self, bit_rate):
        # uncomment for debugging
        # np.random.seed(0)
        net = core.Net("bench")
        batchsize = np.random.randint(2, 1000)
        blocksize = np.random.randint(2, 1000)
        input_data = np.random.rand(batchsize, blocksize).astype(np.float32)
        op = core.CreateOperator(
            "FloatToFused" + str(bit_rate) + "BitFakeRowwiseQuantized",
            "input_data",
            "minmax_quantized_data",
        )
        net.Proto().op.extend([op])
        net.Fused8BitRowwiseQuantizedToFloat(
            "minmax_quantized_data", "minmax_dequantized_data"
        )
        op = core.CreateOperator(
            "FloatToFused" + str(bit_rate) + "BitFakeRowwiseQuantized",
            "input_data",
            "greedy_quantized_data",
            engine="GREEDY",
        )
        net.Proto().op.extend([op])
        net.Fused8BitRowwiseQuantizedToFloat(
            "greedy_quantized_data", "greedy_dequantized_data"
        )
        workspace.FeedBlob("input_data", input_data)

        workspace.GlobalInit(["caffe2", "--caffe2_log_level=0"])
        workspace.RunNetOnce(net)

        minmax_dequantized_data = workspace.FetchBlob("minmax_dequantized_data")
        greedy_dequantized_data = workspace.FetchBlob("greedy_dequantized_data")

        err_thres = ErrorThresholdRow(input_data, bit_rate)
        diff_minmax = np.abs(input_data - minmax_dequantized_data)
        diff_greedy = np.abs(input_data - greedy_dequantized_data)
        for i in range(err_thres.size):
            # Check error from minmax quantization is within the bound derived from the range
            assert (
                np.sum(diff_minmax[i, :] > err_thres[i]) == 0
            ), "error at row {} too high (diff_minmax[i, :] {} diff_minmax[i, :] > err_thres[i] {} err_thres[i] {}".format(
                i, diff_minmax[i, :], diff_minmax[i, :] > err_thres[i], err_thres[i]
            )

            # Check error from greedy quantization is smaller than minmax quantization
            # Multiply by a margin 1.03 to consider inexactness of
            # floating-point operations and from binning (in exact math,
            # l2_greedy should be no less than l2_minmax).
            l2_minmax_err = np.linalg.norm(diff_minmax[i, :])
            l2_greedy_err = np.linalg.norm(diff_greedy[i, :])
            assert (
                l2_greedy_err <= l2_minmax_err * 1.03
            ), "L2 quantization error using greedy algorithm {} at row {} is bigger than error using minmax {} (input_data[i,:] {} minmax_dequantized_data[i,:] {} greedy_dequantized_data[i,:] {}".format(  # noqa
                l2_greedy_err,
                i,
                l2_minmax_err,
                input_data[i, :],
                minmax_dequantized_data[i, :],
                greedy_dequantized_data[i, :],
            )


class TestNBitGreedyFused(hu.HypothesisTestCase):
    @given(bit_rate=st.sampled_from([2, 4]))
    @settings(deadline=None, max_examples=50)
    def testNBit(self, bit_rate):
        # uncomment for debugging
        # np.random.seed(0)
        net = core.Net("bench")
        batchsize = np.random.randint(2, 1000)
        assert 8 % bit_rate == 0
        num_elem_per_byte = 8 // bit_rate
        blocksize = np.random.randint(2, 500) * num_elem_per_byte
        input_data = np.random.rand(batchsize, blocksize).astype(np.float32)

        op = core.CreateOperator(
            "FloatToFused" + str(bit_rate) + "BitRowwiseQuantized",
            "input_data",
            "minmax_quantized_data",
        )
        net.Proto().op.extend([op])
        op = core.CreateOperator(
            "Fused" + str(bit_rate) + "BitRowwiseQuantizedToFloat",
            "minmax_quantized_data",
            "minmax_dequantized_data",
        )
        net.Proto().op.extend([op])
        op = core.CreateOperator(
            "FloatToFused" + str(bit_rate) + "BitRowwiseQuantized",
            "input_data",
            "greedy_quantized_data",
            engine="GREEDY",
        )
        net.Proto().op.extend([op])
        op = core.CreateOperator(
            "Fused" + str(bit_rate) + "BitRowwiseQuantizedToFloat",
            "greedy_quantized_data",
            "greedy_dequantized_data",
        )
        net.Proto().op.extend([op])
        workspace.FeedBlob("input_data", input_data)

        workspace.GlobalInit(["caffe2", "--caffe2_log_level=0"])
        workspace.RunNetOnce(net)

        minmax_dequantized_data = workspace.FetchBlob("minmax_dequantized_data")
        greedy_dequantized_data = workspace.FetchBlob("greedy_dequantized_data")

        diff_minmax = np.abs(input_data - minmax_dequantized_data)
        l2_minmax = np.linalg.norm(input_data - minmax_dequantized_data, axis=1)
        diff_greedy = np.abs(input_data - greedy_dequantized_data)
        l2_greedy = np.linalg.norm(input_data - greedy_dequantized_data, axis=1)

        for i in range(input_data.shape[0]):
            # Compare with Python reference greedy search implementation
            xmin, xmax = param_search_greedy(
                input_data[i, :], bit_rate, n_bins=200, ratio=0.16
            )
            X_q_ref, l2_greedy_ref = _compress_uniform_simplified(
                input_data[i, :], bit_rate, xmin, xmax, fp16_scale_bias=True
            )
            l2_discrepancy = np.abs(l2_greedy[i] - l2_greedy_ref) / input_data.shape[1]
            # C++ implementation has a different accumulation order when
            # computing norm in compress_uniform_simplified_ so we shouldn't
            # use too small tolerance.
            assert (
                l2_discrepancy < 1e-5
            ), "l2_discrepancy between C++ and Python greedy algorithm {} at row {} is too high (actual l2 err {} ref l2 err {} actual {} ref {})".format(  # noqa
                l2_discrepancy,
                i,
                l2_greedy[i],
                l2_greedy_ref,
                greedy_dequantized_data[i, :],
                X_q_ref,
            )

            # Check error from greedy quantization is smaller than minmax quantization
            # Multiply by a margin 1.03 to consider inexactness of
            # floating-point operations and from binning (in exact math,
            # l2_greedy should be no less than l2_minmax).
            assert (
                l2_greedy[i] <= l2_minmax[i] * 1.03
            ), "L2 quantization error using greedy algorithm {} at row {} is bigger than error using minmax {}".format(
                l2_greedy[i], i, l2_minmax[i]
            )
