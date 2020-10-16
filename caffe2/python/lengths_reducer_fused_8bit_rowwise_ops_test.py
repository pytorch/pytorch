

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, workspace
from hypothesis import given


def compare_rowwise(emb_orig, emb_reconstructed, fp16):
    # there is an absolute error introduced per row through int8 quantization
    # and a relative error introduced when quantizing back from fp32 to fp16
    assert emb_orig.shape == emb_reconstructed.shape
    rtol = 1e-8
    if fp16:
        rtol = 1e-3
    erange = np.amax(emb_orig, axis=1) - np.amin(emb_orig, axis=1)

    threshold = erange / 255.0 / 1.9

    for i in range(emb_orig.shape[0]):
        r_orig = emb_orig[i, :]
        r_reconstructed = emb_reconstructed[i, :]

        isclose = np.isclose(r_orig, r_reconstructed, atol=threshold[i], rtol=rtol)
        n_violated = isclose.size - isclose.sum()

        if n_violated > 0:
            print(isclose, threshold[i])
            print(i, r_orig, r_reconstructed, threshold[i], r_orig - r_reconstructed)
        assert n_violated == 0


class TestLengthsReducerOpsFused8BitRowwise(hu.HypothesisTestCase):
    @given(
        num_rows=st.integers(1, 20),
        blocksize=st.sampled_from([8, 16, 32, 64, 85, 96, 128, 163]),
        weighted=st.booleans(),
        seed=st.integers(0, 2 ** 32 - 1),
        empty_indices=st.booleans(),
        fp16=st.booleans(),
    )
    def test_sparse_lengths_sum(
        self, num_rows, blocksize, weighted, seed, empty_indices, fp16
    ):
        net = core.Net("bench")

        np.random.seed(seed)

        if fp16:
            input_data = np.random.rand(num_rows, blocksize).astype(np.float16)
        else:
            input_data = np.random.rand(num_rows, blocksize).astype(np.float32)
        if empty_indices:
            lengths = np.zeros(num_rows, dtype=np.int32)
            num_indices = 0
        else:
            num_indices = np.random.randint(len(input_data))
            # the number of indices per sample
            lengths_split = np.clip(num_indices // 2, 1, 10)
            lengths = (
                np.ones([num_indices // lengths_split], dtype=np.int32) * lengths_split
            )
            # readjust num_indices when lengths_split doesn't divide num_indices
            num_indices = num_indices // lengths_split * lengths_split
        indices = np.random.randint(
            low=0, high=len(input_data), size=[num_indices], dtype=np.int32
        )
        weights = np.random.uniform(size=[len(indices)]).astype(np.float32)

        if fp16:
            quantized_data = net.HalfFloatToFused8BitRowwiseQuantized(
                "input_data", "quantized_data"
            )
            dequantized_data = net.Fused8BitRowwiseQuantizedToHalfFloat(
                quantized_data, "dequantized_data"
            )
        else:
            quantized_data = net.FloatToFused8BitRowwiseQuantized(
                "input_data", "quantized_data"
            )
            dequantized_data = net.Fused8BitRowwiseQuantizedToFloat(
                quantized_data, "dequantized_data"
            )

        if weighted:
            net.SparseLengthsWeightedSum(
                [dequantized_data, "weights", "indices", "lengths"], "sum_reference"
            )
            net.SparseLengthsWeightedSumFused8BitRowwise(
                [quantized_data, "weights", "indices", "lengths"], "sum_quantized"
            )
        else:
            net.SparseLengthsSum(
                [dequantized_data, "indices", "lengths"], "sum_reference"
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

        dequantized_data = workspace.FetchBlob("dequantized_data")
        np.testing.assert_array_almost_equal(
            input_data, workspace.FetchBlob("input_data")
        )
        compare_rowwise(input_data, dequantized_data, fp16)

        sum_reference = workspace.FetchBlob("sum_reference")
        sum_quantized = workspace.FetchBlob("sum_quantized")
        if fp16:
            np.testing.assert_array_almost_equal(
                sum_reference, sum_quantized, decimal=3
            )
        else:
            np.testing.assert_array_almost_equal(sum_reference, sum_quantized)

    @given(
        num_rows=st.integers(1, 20),
        blocksize=st.sampled_from([8, 16, 32, 64, 85, 96, 128, 163]),
        seed=st.integers(0, 2 ** 32 - 1),
        empty_indices=st.booleans(),
        fp16=st.booleans(),
    )
    def test_sparse_lengths_mean(self, num_rows, blocksize, seed, empty_indices, fp16):
        net = core.Net("bench")

        np.random.seed(seed)

        if fp16:
            input_data = np.random.rand(num_rows, blocksize).astype(np.float16)
        else:
            input_data = np.random.rand(num_rows, blocksize).astype(np.float32)

        if empty_indices:
            lengths = np.zeros(num_rows, dtype=np.int32)
            num_indices = 0
        else:
            num_indices = np.random.randint(len(input_data))
            # the number of indices per sample
            lengths_split = np.clip(num_indices // 2, 1, 10)
            lengths = (
                np.ones([num_indices // lengths_split], dtype=np.int32) * lengths_split
            )
            # readjust num_indices when lengths_split doesn't divide num_indices
            num_indices = num_indices // lengths_split * lengths_split
        indices = np.random.randint(
            low=0, high=len(input_data), size=[num_indices], dtype=np.int32
        )
        print(indices, lengths)

        if fp16:
            quantized_data = net.HalfFloatToFused8BitRowwiseQuantized(
                "input_data", "quantized_data"
            )
            dequantized_data = net.Fused8BitRowwiseQuantizedToHalfFloat(
                quantized_data, "dequantized_data"
            )
        else:
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

        dequantized_data = workspace.FetchBlob("dequantized_data")
        np.testing.assert_array_almost_equal(
            input_data, workspace.FetchBlob("input_data")
        )
        compare_rowwise(input_data, dequantized_data, fp16)

        mean_reference = workspace.FetchBlob("mean_reference")
        mean_quantized = workspace.FetchBlob("mean_quantized")
        if fp16:
            np.testing.assert_array_almost_equal(
                mean_reference, mean_quantized, decimal=3
            )
        else:
            np.testing.assert_array_almost_equal(mean_reference, mean_quantized)
