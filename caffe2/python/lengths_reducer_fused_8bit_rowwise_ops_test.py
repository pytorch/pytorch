from __future__ import absolute_import, division, print_function, unicode_literals

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, workspace
from hypothesis import given


class TestLengthsReducerOpsFused8BitRowwise(hu.HypothesisTestCase):
    @given(
        batchsize=st.integers(1, 20),
        blocksize=st.sampled_from([8, 16, 32, 64, 85, 96, 128, 163]),
        weighted=st.booleans(),
        seed=st.integers(0, 2 ** 32 - 1),
        empty_indices=st.booleans(),
    )
    def test_sparse_lengths_sum(
        self, batchsize, blocksize, weighted, seed, empty_indices
    ):
        net = core.Net("bench")

        np.random.seed(seed)

        input_data = np.random.rand(batchsize, blocksize).astype(np.float32)
        if empty_indices:
            lengths = np.zeros(batchsize, dtype=np.int32)
            num_indices = 0
        else:
            num_indices = np.random.randint(len(input_data))
            num_lengths = np.clip(1, num_indices // 2, 10)
            lengths = (
                np.ones([num_indices // num_lengths], dtype=np.int32) * num_lengths
            )
            # readjust num_indices when num_lengths doesn't divide num_indices
            num_indices = num_indices // num_lengths * num_lengths
        indices = np.random.randint(
            low=0,
            high=len(input_data),
            size=[num_indices],
            dtype=np.int32,
        )
        weights = np.random.uniform(size=[len(indices)]).astype(np.float32)

        quantized_data = net.FloatToFused8BitRowwiseQuantized(
            "input_data", "quantized_data"
        )
        dequantized_data = net.Fused8BitRowwiseQuantizedToFloat(
            quantized_data, "dequantized_data"
        )

        if weighted:
            net.SparseLengthsWeightedSum(
                [dequantized_data, "weights", "indices", "lengths"],
                "sum_reference",
            )
            net.SparseLengthsWeightedSumFused8BitRowwise(
                [quantized_data, "weights", "indices", "lengths"], "sum_quantized"
            )
        else:
            net.SparseLengthsSum(
                [dequantized_data, "indices", "lengths"], "sum_reference",
            )
            net.SparseLengthsSumFused8BitRowwise(
                [quantized_data, "indices", "lengths"], "sum_quantized"
            )

        workspace.FeedBlob("input_data", input_data)
        workspace.FeedBlob("weights", weights)
        workspace.FeedBlob("indices", indices)
        workspace.FeedBlob("lengths", lengths)

        workspace.GlobalInit(["caffe2", "--caffe2_log_level=0"])
        workspace.CreateNet(net)
        workspace.RunNetOnce(net)

        sum_reference = workspace.FetchBlob("sum_reference")
        sum_quantized = workspace.FetchBlob("sum_quantized")
        np.testing.assert_array_almost_equal(sum_reference, sum_quantized)

    @given(
        batchsize=st.integers(1, 20),
        blocksize=st.sampled_from([8, 16, 32, 64, 85, 96, 128, 163]),
        seed=st.integers(0, 2 ** 32 - 1),
        empty_indices=st.booleans(),
    )
    def test_sparse_lengths_mean(self, batchsize, blocksize, seed, empty_indices):
        net = core.Net("bench")

        np.random.seed(seed)

        input_data = np.random.rand(batchsize, blocksize).astype(np.float32)
        if empty_indices:
            lengths = np.zeros(batchsize, dtype=np.int32)
            num_indices = 0
        else:
            num_indices = np.random.randint(len(input_data))
            num_lengths = np.clip(1, num_indices // 2, 10)
            lengths = (
                np.ones([num_indices // num_lengths], dtype=np.int32) * num_lengths
            )
            # readjust num_indices when num_lengths doesn't divide num_indices
            num_indices = num_indices // num_lengths * num_lengths
        indices = np.random.randint(
            low=0,
            high=len(input_data),
            size=[num_indices],
            dtype=np.int32,
        )
        print(indices, lengths)

        quantized_data = net.FloatToFused8BitRowwiseQuantized(
            "input_data", "quantized_data"
        )
        dequantized_data = net.Fused8BitRowwiseQuantizedToFloat(
            quantized_data, "dequantized_data"
        )

        net.SparseLengthsMean(
            [dequantized_data, "indices", "lengths"], "mean_reference"
        )
        net.SparseLengthsMeanFused8BitRowwise(
            [quantized_data, "indices", "lengths"], "mean_quantized"
        )

        workspace.FeedBlob("input_data", input_data)
        workspace.FeedBlob("indices", indices)
        workspace.FeedBlob("lengths", lengths)

        workspace.GlobalInit(["caffe2", "--caffe2_log_level=0"])
        workspace.CreateNet(net)
        workspace.RunNetOnce(net)

        mean_reference = workspace.FetchBlob("mean_reference")
        mean_quantized = workspace.FetchBlob("mean_quantized")
        np.testing.assert_array_almost_equal(mean_reference, mean_quantized)
