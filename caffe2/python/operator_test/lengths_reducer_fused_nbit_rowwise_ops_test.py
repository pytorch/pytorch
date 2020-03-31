from __future__ import absolute_import, division, print_function, unicode_literals

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, dyndep, workspace
from hypothesis import given


class TestLengthsReducerOpsFusedNBitRowwise(hu.HypothesisTestCase):
    @given(
        num_rows=st.integers(1, 20),
        blocksize=st.sampled_from([8, 12, 16, 32, 64, 96, 128]),
        weighted=st.booleans(),
        seed=st.integers(0, 2 ** 32 - 1),
        empty_indices=st.booleans(),
        engine=st.sampled_from(["", "GREEDY"]),
        bit_rate=st.sampled_from([2, 4]),
    )
    def test_sparse_lengths_sum(
        self, num_rows, blocksize, weighted, seed, empty_indices, engine, bit_rate
    ):
        net = core.Net("bench")

        np.random.seed(seed)

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
            low=0, high=len(input_data), size=[num_indices], dtype=np.int64
        )
        weights = np.random.uniform(size=[len(indices)]).astype(np.float32)

        op = core.CreateOperator(
            "FloatToFused" + str(bit_rate) + "BitRowwiseQuantized",
            "input_data",
            "quantized_data",
            engine=engine,
        )
        net.Proto().op.extend([op])
        op = core.CreateOperator(
            "Fused" + str(bit_rate) + "BitRowwiseQuantizedToFloat",
            "quantized_data",
            "dequantized_data",
        )
        net.Proto().op.extend([op])
        op = core.CreateOperator(
            "FloatToFused" + str(bit_rate) + "BitFakeRowwiseQuantized",
            "input_data",
            "fake_quantized_data",
            engine=engine,
        )
        net.Proto().op.extend([op])

        if weighted:
            net.SparseLengthsWeightedSum(
                ["dequantized_data", "weights", "indices", "lengths"], "sum_reference"
            )
            net.SparseLengthsWeightedSumFused8BitRowwise(
                ["fake_quantized_data", "weights", "indices", "lengths"],
                "sum_fake_quantized",
            )
            op = core.CreateOperator(
                "SparseLengthsWeightedSumFused" + str(bit_rate) + "BitRowwise",
                ["quantized_data", "weights", "indices", "lengths"],
                "sum_quantized",
            )
            net.Proto().op.extend([op])
        else:
            net.SparseLengthsSum(
                ["dequantized_data", "indices", "lengths"], "sum_reference"
            )
            net.SparseLengthsSumFused8BitRowwise(
                ["fake_quantized_data", "indices", "lengths"], "sum_fake_quantized"
            )
            op = core.CreateOperator(
                "SparseLengthsSumFused" + str(bit_rate) + "BitRowwise",
                ["quantized_data", "indices", "lengths"],
                "sum_quantized",
            )
            net.Proto().op.extend([op])
        net.Proto().external_input.extend(["input_data"])

        workspace.FeedBlob("input_data", input_data)
        workspace.FeedBlob("weights", weights)
        workspace.FeedBlob("indices", indices)
        workspace.FeedBlob("lengths", lengths)

        workspace.GlobalInit(["caffe2", "--caffe2_log_level=0"])
        workspace.RunNetOnce(net)

        sum_reference = workspace.FetchBlob("sum_reference")
        sum_fake_quantized = workspace.FetchBlob("sum_fake_quantized")
        sum_quantized = workspace.FetchBlob("sum_quantized")

        np.testing.assert_array_almost_equal(sum_reference, sum_quantized)
        np.testing.assert_array_equal(sum_fake_quantized, sum_quantized)

    @given(
        num_rows=st.integers(1, 20),
        blocksize=st.sampled_from([8, 12, 16, 32, 64, 96, 128]),
        seed=st.integers(0, 2 ** 32 - 1),
        empty_indices=st.booleans(),
        engine=st.sampled_from(["", "GREEDY"]),
        bit_rate=st.sampled_from([2, 4]),
    )
    def test_sparse_lengths_mean(
        self, num_rows, blocksize, seed, empty_indices, engine, bit_rate
    ):
        net = core.Net("bench")

        np.random.seed(seed)

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
        #  Use int32 here because int64 is covered by test_sparse_lengths_sum
        indices = np.random.randint(
            low=0, high=len(input_data), size=[num_indices], dtype=np.int32
        )

        op = core.CreateOperator(
            "FloatToFused" + str(bit_rate) + "BitRowwiseQuantized",
            "input_data",
            "quantized_data",
            engine=engine,
        )
        net.Proto().op.extend([op])
        op = core.CreateOperator(
            "Fused" + str(bit_rate) + "BitRowwiseQuantizedToFloat",
            "quantized_data",
            "dequantized_data",
        )
        net.Proto().op.extend([op])
        op = core.CreateOperator(
            "FloatToFused" + str(bit_rate) + "BitFakeRowwiseQuantized",
            "input_data",
            "fake_quantized_data",
            engine=engine,
        )
        net.Proto().op.extend([op])

        net.SparseLengthsMean(
            ["dequantized_data", "indices", "lengths"], "mean_reference"
        )
        net.SparseLengthsMeanFused8BitRowwise(
            ["fake_quantized_data", "indices", "lengths"], "mean_fake_quantized"
        )
        op = core.CreateOperator(
            "SparseLengthsMeanFused" + str(bit_rate) + "BitRowwise",
            ["quantized_data", "indices", "lengths"],
            "mean_quantized",
        )
        net.Proto().op.extend([op])
        net.Proto().external_input.extend(["input_data"])

        workspace.FeedBlob("input_data", input_data)
        workspace.FeedBlob("indices", indices)
        workspace.FeedBlob("lengths", lengths)

        workspace.GlobalInit(["caffe2", "--caffe2_log_level=0"])
        workspace.RunNetOnce(net)

        mean_reference = workspace.FetchBlob("mean_reference")
        mean_fake_quantized = workspace.FetchBlob("mean_fake_quantized")
        mean_quantized = workspace.FetchBlob("mean_quantized")

        np.testing.assert_array_almost_equal(mean_reference, mean_quantized)
        np.testing.assert_array_equal(mean_fake_quantized, mean_quantized)

    @given(
        num_rows=st.integers(1, 20),
        blocksize=st.sampled_from([8, 12, 16, 32, 64, 96, 128]),
        weighted=st.booleans(),
        empty_indices=st.booleans(),
        bit_rate=st.sampled_from([2, 4, 8]),
        indices_64bit=st.booleans(),
    )
    def test_sparse_lengths_sum_rowwise_sparse(
        self, num_rows, blocksize, weighted, empty_indices, bit_rate, indices_64bit
    ):
        net = core.Net("bench")

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
        #  Use int32 here because int64 is covered by test_sparse_lengths_sum
        index_type = np.int64 if indices_64bit else np.int32
        indices = np.random.randint(
            low=0, high=len(input_data), size=[num_indices], dtype=index_type
        )
        weights = np.random.uniform(size=[len(indices)]).astype(np.float32)

        op = core.CreateOperator(
            "FloatToFused" + str(bit_rate) + "BitRowwiseQuantized",
            "input_data",
            "quantized_data",
        )
        workspace.FeedBlob("input_data", input_data)
        workspace.RunOperatorOnce(op)
        quantized_data = workspace.FetchBlob("quantized_data")

        # Prune and generate mapping table
        sparsity = 0.7
        mapping_table = np.zeros(num_rows, dtype=np.int32)
        num_compressed_rows = 0
        unpruned_ids = []
        for i in range(num_rows):
            if np.random.uniform() < sparsity:
                mapping_table[i] = -1
                quantized_data[i, :] = 0
            else:
                mapping_table[i] = num_compressed_rows
                num_compressed_rows += 1
                unpruned_ids.append(i)

        pruned_quantized_data = quantized_data[unpruned_ids]

        inputs = (
            ["quantized_data"]
            + (["weights"] if weighted else [])
            + ["indices", "lengths"]
        )
        op = core.CreateOperator(
            "SparseLengths"
            + ("Weighted" if weighted else "")
            + "SumFused"
            + str(bit_rate)
            + "BitRowwise",
            inputs,
            "sum_reference",
        )
        net.Proto().op.extend([op])

        inputs[0] = "pruned_quantized_data"
        op = core.CreateOperator(
            "SparseLengths"
            + ("Weighted" if weighted else "")
            + "Sum"
            + str(bit_rate)
            + "BitRowwiseSparse",
            inputs + ["mapping_table"],
            "sum_pruned",
        )
        net.Proto().op.extend([op])

        workspace.FeedBlob("quantized_data", quantized_data)
        workspace.FeedBlob("pruned_quantized_data", pruned_quantized_data)
        workspace.FeedBlob("weights", weights)
        workspace.FeedBlob("indices", indices)
        workspace.FeedBlob("lengths", lengths)
        workspace.FeedBlob("mapping_table", mapping_table)

        workspace.RunNetOnce(net)

        sum_reference = workspace.FetchBlob("sum_reference")
        sum_pruned = workspace.FetchBlob("sum_pruned")

        np.testing.assert_array_equal(sum_reference, sum_pruned)

    @given(
        num_rows=st.integers(1, 20),
        blocksize=st.sampled_from([8, 12, 16, 32, 64, 96, 128]),
        seed=st.integers(0, 2 ** 32 - 1),
        empty_indices=st.booleans(),
        engine=st.sampled_from(["", "GREEDY"]),
        bit_rate=st.sampled_from([2, 4]),
    )
    def test_sparse_lengths_mean_rowwise_sparse_with_skipped_pruning(
        self, num_rows, blocksize, seed, empty_indices, engine, bit_rate
    ):
        net = core.Net("bench")

        np.random.seed(seed)

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
        #  Use int32 here because int64 is covered by test_sparse_lengths_sum
        indices = np.random.randint(
            low=0, high=len(input_data), size=[num_indices], dtype=np.int32
        )

        op = core.CreateOperator(
            "FloatToFused" + str(bit_rate) + "BitRowwiseQuantized",
            "input_data",
            "quantized_data",
            engine=engine,
        )
        net.Proto().op.extend([op])
        op = core.CreateOperator(
            "Fused" + str(bit_rate) + "BitRowwiseQuantizedToFloat",
            "quantized_data",
            "dequantized_data",
        )
        net.Proto().op.extend([op])
        op = core.CreateOperator(
            "FloatToFused" + str(bit_rate) + "BitFakeRowwiseQuantized",
            "input_data",
            "fake_quantized_data",
            engine=engine,
        )
        net.Proto().op.extend([op])

        net.SparseLengthsMean(
            ["dequantized_data", "indices", "lengths"], "mean_reference"
        )
        net.SparseLengthsMeanFused8BitRowwise(
            ["fake_quantized_data", "indices", "lengths"], "mean_fake_quantized"
        )
        op1 = core.CreateOperator(
            "SparseLengthsMeanFused" + str(bit_rate) + "BitRowwise",
            ["quantized_data", "indices", "lengths"],
            "mean_quantized",
        )
        op2 = core.CreateOperator(
            "SparseLengthsMean" + str(bit_rate) + "BitRowwiseSparse",
            ["quantized_data", "indices", "lengths"] + ["mapping_table"],
            "mean_quantized_pruned",
        )
        net.Proto().op.extend([op1, op2])
        net.Proto().external_input.extend(["input_data", "mapping_table"])

        workspace.FeedBlob("input_data", input_data)
        workspace.FeedBlob("indices", indices)
        workspace.FeedBlob("lengths", lengths)
        mapping_table = np.array([0]).astype(dtype=np.int32)
        workspace.FeedBlob("mapping_table", mapping_table)

        workspace.GlobalInit(["caffe2", "--caffe2_log_level=0"])
        workspace.RunNetOnce(net)

        mean_reference = workspace.FetchBlob("mean_reference")
        mean_fake_quantized = workspace.FetchBlob("mean_fake_quantized")
        mean_quantized = workspace.FetchBlob("mean_quantized")
        mean_quantized_pruned = workspace.FetchBlob("mean_quantized_pruned")

        np.testing.assert_array_almost_equal(mean_reference, mean_quantized)
        np.testing.assert_array_equal(mean_fake_quantized, mean_quantized)
        np.testing.assert_array_equal(mean_quantized_pruned, mean_quantized)
