

import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import numpy as np
from caffe2.python import core, workspace
from hypothesis import given, settings, strategies as st


def batched_boarders_and_data(
    data_min_size=5,
    data_max_size=10,
    examples_min_number=1,
    examples_max_number=4,
    example_min_size=1,
    example_max_size=3,
    dtype=np.float32,
    elements=None,
):
    dims_ = st.tuples(
        st.integers(min_value=data_min_size, max_value=data_max_size),
        st.integers(min_value=examples_min_number, max_value=examples_max_number),
        st.integers(min_value=example_min_size, max_value=example_max_size),
    )
    return dims_.flatmap(
        lambda dims: st.tuples(
            hu.arrays(
                [dims[1], dims[2], 2],
                dtype=np.int32,
                elements=st.integers(min_value=0, max_value=dims[0]),
            ),
            hu.arrays([dims[0]], dtype, elements),
        )
    )


@st.composite
def _tensor_splits(draw):
    lengths = draw(st.lists(st.integers(1, 5), min_size=1, max_size=10))
    batch_size = draw(st.integers(1, 5))
    element_pairs = [
        (batch, r) for batch in range(batch_size) for r in range(len(lengths))
    ]
    perm = draw(st.permutations(element_pairs))
    perm = perm[:-1]  # skip one range
    ranges = [[(0, 0)] * len(lengths) for _ in range(batch_size)]
    offset = 0
    for pair in perm:
        ranges[pair[0]][pair[1]] = (offset, lengths[pair[1]])
        offset += lengths[pair[1]]

    data = draw(
        st.lists(
            st.floats(min_value=-1.0, max_value=1.0), min_size=offset, max_size=offset
        )
    )

    key = draw(st.permutations(range(offset)))

    return (
        np.array(data).astype(np.float32),
        np.array(ranges),
        np.array(lengths),
        np.array(key).astype(np.int64),
    )


@st.composite
def _bad_tensor_splits(draw):
    lengths = draw(st.lists(st.integers(4, 6), min_size=4, max_size=4))
    batch_size = 4
    element_pairs = [
        (batch, r) for batch in range(batch_size) for r in range(len(lengths))
    ]
    perm = draw(st.permutations(element_pairs))
    ranges = [[(0, 0)] * len(lengths) for _ in range(batch_size)]
    offset = 0

    # Inject some bad samples depending on the batch.
    # Batch 2: length is set to 0. This way, 25% of the samples are empty.
    # Batch 0-1: length is set to half the original length. This way, 50% of the
    # samples are of mismatched length.
    for pair in perm:
        if pair[0] == 2:
            length = 0
        elif pair[0] <= 1:
            length = lengths[pair[1]] // 2
        else:
            length = lengths[pair[1]]
        ranges[pair[0]][pair[1]] = (offset, length)
        offset += length

    data = draw(
        st.lists(
            st.floats(min_value=-1.0, max_value=1.0), min_size=offset, max_size=offset
        )
    )

    key = draw(st.permutations(range(offset)))

    return (
        np.array(data).astype(np.float32),
        np.array(ranges),
        np.array(lengths),
        np.array(key).astype(np.int64),
    )


def gather_ranges(data, ranges):
    lengths = []
    output = []
    for example_ranges in ranges:
        length = 0
        for range in example_ranges:
            assert len(range) == 2
            output.extend(data[range[0] : range[0] + range[1]])
            length += range[1]
        lengths.append(length)
    return output, lengths


def gather_ranges_to_dense(data, ranges, lengths):
    outputs = []
    assert len(ranges)
    batch_size = len(ranges)
    assert len(ranges[0])
    num_ranges = len(ranges[0])
    assert ranges.shape[2] == 2
    for i in range(num_ranges):
        out = []
        for j in range(batch_size):
            start, length = ranges[j][i]
            if not length:
                out.append([0] * lengths[i])
            else:
                assert length == lengths[i]
                out.append(data[start : start + length])
        outputs.append(np.array(out))
    return outputs


def gather_ranges_to_dense_with_key(data, ranges, key, lengths):
    outputs = []
    assert len(ranges)
    batch_size = len(ranges)
    assert len(ranges[0])
    num_ranges = len(ranges[0])
    assert ranges.shape[2] == 2
    for i in range(num_ranges):
        out = []
        for j in range(batch_size):
            start, length = ranges[j][i]
            if not length:
                out.append([0] * lengths[i])
            else:
                assert length == lengths[i]
                key_data_list = zip(
                    key[start : start + length], data[start : start + length]
                )
                sorted_key_data_list = sorted(key_data_list, key=lambda x: x[0])
                sorted_data = [d for (k, d) in sorted_key_data_list]
                out.append(sorted_data)
        outputs.append(np.array(out))
    return outputs


class TestGatherRanges(serial.SerializedTestCase):
    @given(boarders_and_data=batched_boarders_and_data(), **hu.gcs_cpu_only)
    @settings(deadline=1000)
    def test_gather_ranges(self, boarders_and_data, gc, dc):
        boarders, data = boarders_and_data

        def boarders_to_range(boarders):
            assert len(boarders) == 2
            boarders = sorted(boarders)
            return [boarders[0], boarders[1] - boarders[0]]

        ranges = np.apply_along_axis(boarders_to_range, 2, boarders)

        self.assertReferenceChecks(
            device_option=gc,
            op=core.CreateOperator(
                "GatherRanges", ["data", "ranges"], ["output", "lengths"]
            ),
            inputs=[data, ranges],
            reference=gather_ranges,
        )

    @given(tensor_splits=_tensor_splits(), **hu.gcs_cpu_only)
    @settings(deadline=1000)
    def test_gather_ranges_split(self, tensor_splits, gc, dc):
        data, ranges, lengths, _ = tensor_splits

        self.assertReferenceChecks(
            device_option=gc,
            op=core.CreateOperator(
                "GatherRangesToDense",
                ["data", "ranges"],
                ["X_{}".format(i) for i in range(len(lengths))],
                lengths=lengths,
            ),
            inputs=[data, ranges, lengths],
            reference=gather_ranges_to_dense,
        )

    @given(tensor_splits=_tensor_splits(), **hu.gcs_cpu_only)
    def test_gather_ranges_with_key_split(self, tensor_splits, gc, dc):
        data, ranges, lengths, key = tensor_splits

        self.assertReferenceChecks(
            device_option=gc,
            op=core.CreateOperator(
                "GatherRangesToDense",
                ["data", "ranges", "key"],
                ["X_{}".format(i) for i in range(len(lengths))],
                lengths=lengths,
            ),
            inputs=[data, ranges, key, lengths],
            reference=gather_ranges_to_dense_with_key,
        )

    def test_shape_and_type_inference(self):
        with hu.temp_workspace("shape_type_inf_int32"):
            net = core.Net("test_net")
            net.ConstantFill([], "ranges", shape=[3, 5, 2], dtype=core.DataType.INT32)
            net.ConstantFill([], "values", shape=[64], dtype=core.DataType.INT64)
            net.GatherRanges(["values", "ranges"], ["values_output", "lengths_output"])
            (shapes, types) = workspace.InferShapesAndTypes([net], {})

            self.assertEqual(shapes["values_output"], [64])
            self.assertEqual(types["values_output"], core.DataType.INT64)
            self.assertEqual(shapes["lengths_output"], [3])
            self.assertEqual(types["lengths_output"], core.DataType.INT32)

    @given(tensor_splits=_bad_tensor_splits(), **hu.gcs_cpu_only)
    @settings(deadline=10000)
    def test_empty_range_check(self, tensor_splits, gc, dc):
        data, ranges, lengths, key = tensor_splits

        workspace.FeedBlob("data", data)
        workspace.FeedBlob("ranges", ranges)
        workspace.FeedBlob("key", key)

        def getOpWithThreshold(
            min_observation=2, max_mismatched_ratio=0.5, max_empty_ratio=None
        ):
            return core.CreateOperator(
                "GatherRangesToDense",
                ["data", "ranges", "key"],
                ["X_{}".format(i) for i in range(len(lengths))],
                lengths=lengths,
                min_observation=min_observation,
                max_mismatched_ratio=max_mismatched_ratio,
                max_empty_ratio=max_empty_ratio,
            )

        workspace.RunOperatorOnce(getOpWithThreshold())

        workspace.RunOperatorOnce(
            getOpWithThreshold(max_mismatched_ratio=0.3, min_observation=50)
        )

        with self.assertRaises(RuntimeError):
            workspace.RunOperatorOnce(
                getOpWithThreshold(max_mismatched_ratio=0.3, min_observation=5)
            )

        with self.assertRaises(RuntimeError):
            workspace.RunOperatorOnce(
                getOpWithThreshold(min_observation=50, max_empty_ratio=0.01)
            )


if __name__ == "__main__":
    import unittest

    unittest.main()
