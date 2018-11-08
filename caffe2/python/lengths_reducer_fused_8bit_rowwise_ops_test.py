from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu

import numpy as np
from hypothesis import given
import hypothesis.strategies as st


class TestLengthsReducerOpsFused8BitRowwise(hu.HypothesisTestCase):
    @given(
        input_data=hu.tensor(min_dim=2, max_dim=2),
        weighted=st.booleans(),
        seed=st.integers(0, 2**32 - 1),
    )
    def test_sparse_lengths_sum(self, input_data, weighted, seed):
        net = core.Net("bench")

        np.random.seed(seed)

        input_data = input_data.astype(np.float32)
        indices = np.random.randint(
            low=0,
            high=len(input_data),
            size=[np.random.randint(len(input_data))],
            dtype=np.int32
        )
        weights = np.random.uniform(size=[len(indices)]).astype(np.float32)
        lengths_split = np.clip(1, len(indices) // 2, 10)
        lengths = np.ones(
            [len(indices) // lengths_split], dtype=np.int32
        ) * lengths_split
        print(indices, weights, lengths)

        quantized_data = net.FloatToFused8BitRowwiseQuantized(
            'input_data', 'quantized_data'
        )
        dequantized_data = net.Fused8BitRowwiseQuantizedToFloat(
            quantized_data, 'dequantized_data'
        )

        if weighted:
            net.SparseLengthsWeightedSum(
                [dequantized_data, 'weights', 'indices', 'lengths'],
                'sum_reference',
                engine='fp16'
            )
            net.SparseLengthsWeightedSumFused8BitRowwise(
                [quantized_data, 'weights', 'indices', 'lengths'],
                'sum_quantized'
            )
        else:
            net.SparseLengthsSum(
                [dequantized_data, 'indices', 'lengths'],
                'sum_reference',
                engine='fp16'
            )
            net.SparseLengthsSumFused8BitRowwise(
                [quantized_data, 'indices', 'lengths'], 'sum_quantized'
            )

        workspace.FeedBlob('input_data', input_data)
        workspace.FeedBlob('weights', weights)
        workspace.FeedBlob('indices', indices)
        workspace.FeedBlob('lengths', lengths)

        workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
        workspace.CreateNet(net)
        workspace.RunNetOnce(net)

        sum_reference = workspace.FetchBlob('sum_reference')
        sum_quantized = workspace.FetchBlob('sum_quantized')
        np.testing.assert_array_almost_equal(sum_reference, sum_quantized)

    @given(
        input_data=hu.tensor(min_dim=2, max_dim=2),
        seed=st.integers(0, 2**32 - 1),
    )
    def test_sparse_lengths_mean(self, input_data, seed):
        net = core.Net("bench")

        np.random.seed(seed)

        input_data = input_data.astype(np.float32)
        indices = np.random.randint(
            low=0,
            high=len(input_data),
            size=[np.random.randint(len(input_data))],
            dtype=np.int32
        )
        lengths_split = np.clip(1, len(indices) // 2, 10)
        lengths = np.ones(
            [len(indices) // lengths_split], dtype=np.int32
        ) * lengths_split
        print(indices, lengths)

        quantized_data = net.FloatToFused8BitRowwiseQuantized(
            'input_data', 'quantized_data'
        )
        dequantized_data = net.Fused8BitRowwiseQuantizedToFloat(
            quantized_data, 'dequantized_data'
        )

        net.SparseLengthsMean(
            [dequantized_data, 'indices', 'lengths'],
            'mean_reference',
            engine='fp16'
        )
        net.SparseLengthsMeanFused8BitRowwise(
            [quantized_data, 'indices', 'lengths'], 'mean_quantized'
        )

        workspace.FeedBlob('input_data', input_data)
        workspace.FeedBlob('indices', indices)
        workspace.FeedBlob('lengths', lengths)

        workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
        workspace.CreateNet(net)
        workspace.RunNetOnce(net)

        mean_reference = workspace.FetchBlob('mean_reference')
        mean_quantized = workspace.FetchBlob('mean_quantized')
        np.testing.assert_array_almost_equal(mean_reference, mean_quantized)
