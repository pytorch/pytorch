




from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np


@st.composite
def id_list_batch(draw):
    batch_size = draw(st.integers(2, 2))
    values_dtype = np.float32
    inputs = []
    sample_size = draw(st.integers(5, 10))
    for _ in range(batch_size):
        values = draw(hnp.arrays(values_dtype, sample_size, st.integers(0, 1)))
        inputs += [values]
    return [np.array(inputs)]


def dense_vector_to_id_list_ref(*arg):
    arg = arg[0]
    batch_size = len(arg)
    assert batch_size > 0
    out_length = []
    out_values = []
    for row in arg:
        length = 0
        for idx, entry in enumerate(row):
            if entry != 0:
                out_values += [idx]
                length += 1
        out_length += [length]
    return (out_length, out_values)


class TestDenseVectorToIdList(hu.HypothesisTestCase):
    def test_dense_vector_to_id_list_ref(self):
        # Verify that the reference implementation is correct!
        dense_input = np.array(
            [[1, 0, 0, 1, 0, 0, 0, 1],
             [1, 0, 1, 0, 0, 0, 0, 1],
             [0, 1, 0, 0, 0, 1, 0, 1]],
            dtype=np.float32)
        sparse_lengths, sparse_values = dense_vector_to_id_list_ref(dense_input)
        expected_lengths = np.array([3, 3, 3], dtype=np.int32)
        expected_values = np.array([0, 3, 7, 0, 2, 7, 1, 5, 7], dtype=np.int64)

        np.testing.assert_array_equal(sparse_lengths, expected_lengths)
        np.testing.assert_array_equal(sparse_values, expected_values)

    @given(inputs=id_list_batch(), **hu.gcs_cpu_only)
    def test_dense_vector_to_id_list_op(self, inputs, gc, dc):
        op = core.CreateOperator(
            "DenseVectorToIdList",
            ["values"],
            ["out_lengths", "out_values"]
        )
        self.assertDeviceChecks(dc, op, inputs, [0])
        self.assertReferenceChecks(gc, op, inputs, dense_vector_to_id_list_ref)
