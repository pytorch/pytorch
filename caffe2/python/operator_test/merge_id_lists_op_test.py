




from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np


@st.composite
def id_list_batch(draw):
    num_inputs = draw(st.integers(1, 3))
    batch_size = draw(st.integers(5, 10))
    values_dtype = draw(st.sampled_from([np.int32, np.int64]))
    inputs = []
    for _ in range(num_inputs):
        size = draw(st.integers(5, 10))
        values = draw(hnp.arrays(values_dtype, size, st.integers(1, 10)))
        lengths = draw(hu.lengths(len(values),
                                  min_segments=batch_size,
                                  max_segments=batch_size))
        inputs.append(lengths)
        inputs.append(values)
    return inputs


def merge_id_lists_ref(*args):
    n = len(args)
    assert n > 0
    assert n % 2 == 0
    batch_size = len(args[0])
    num_inputs = int(n / 2)
    lengths = np.array([np.insert(args[2 * i], 0, 0)
                        for i in range(num_inputs)])
    values = [args[2 * i + 1] for i in range(num_inputs)]
    offsets = [np.cumsum(lengths[j]) for j in range(num_inputs)]

    def merge_arrays(vs, offs, j):
        concat = np.concatenate([vs[i][offs[i][j]:offs[i][j + 1]]
                                for i in range(num_inputs)])
        return np.sort(np.unique(concat))

    merged = [merge_arrays(values, offsets, j) for j in range(batch_size)]
    merged_lengths = np.array([len(x) for x in merged])
    merged_values = np.concatenate(merged)
    return merged_lengths, merged_values


class TestMergeIdListsOp(serial.SerializedTestCase):
    def test_merge_id_lists_ref(self):
        # Verify that the reference implementation is correct!
        lengths_0 = np.array([3, 0, 4], dtype=np.int32)
        values_0 = np.array([1, 5, 6, 2, 4, 5, 6], dtype=np.int64)
        lengths_1 = np.array([3, 2, 1], dtype=np.int32)
        values_1 = np.array([5, 8, 9, 14, 9, 5], dtype=np.int64)

        merged_lengths, merged_values = merge_id_lists_ref(
            lengths_0, values_0, lengths_1, values_1)
        expected_lengths = np.array([5, 2, 4], dtype=np.int32)
        expected_values = np.array([1, 5, 6, 8, 9, 9, 14, 2, 4, 5, 6], dtype=np.int64)

        np.testing.assert_array_equal(merged_lengths, expected_lengths)
        np.testing.assert_array_equal(merged_values, expected_values)

    @serial.given(inputs=id_list_batch(), **hu.gcs_cpu_only)
    def test_merge_id_lists_op(self, inputs, gc, dc):
        num_inputs = int(len(inputs) / 2)
        op = core.CreateOperator(
            "MergeIdLists",
            ["{prefix}_{i}".format(prefix=p, i=i)
                for i in range(num_inputs)
                for p in ["lengths", "values"]],
            ["merged_lengths", "merged_values"]
        )
        self.assertDeviceChecks(dc, op, inputs, [0])
        self.assertReferenceChecks(gc, op, inputs, merge_id_lists_ref)
