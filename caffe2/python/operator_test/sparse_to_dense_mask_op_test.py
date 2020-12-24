




from caffe2.python import core
from hypothesis import given, settings
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


class TestFcOperator(hu.HypothesisTestCase):

    @given(n=st.integers(1, 10), k=st.integers(1, 5),
           use_length=st.booleans(), **hu.gcs_cpu_only)
    @settings(deadline=1000)
    def test_sparse_to_dense_mask(self, n, k, use_length, gc, dc):
        lengths = np.random.randint(k, size=n).astype(np.int32) + 1
        N = sum(lengths)
        indices = np.random.randint(5, size=N)
        values = np.random.rand(N, 2).astype(np.float32)
        default = np.random.rand(2).astype(np.float32)
        mask = np.arange(3)
        np.random.shuffle(mask)

        input_str = ['indices', 'values', 'default']
        input_data = [indices, values, default]
        if use_length and n > 1:
            input_str.append('lengths')
            input_data.append(lengths)
        output_str = ['output']

        op = core.CreateOperator(
            'SparseToDenseMask',
            input_str,
            output_str,
            mask=mask,
        )

        # Check over multiple devices
        self.assertDeviceChecks(
            dc, op, input_data, [0])
        # Gradient check for values
        self.assertGradientChecks(
            gc, op, input_data, 1, [0])

    @given(n=st.integers(1, 10), k=st.integers(1, 5),
           use_length=st.booleans(), **hu.gcs_cpu_only)
    @settings(deadline=1000)
    def test_sparse_to_dense_mask_with_int64(self, n, k, use_length, gc, dc):
        lengths = np.random.randint(k, size=n).astype(np.int32) + 1
        N = sum(lengths)
        int64_mask = 10000000000
        indices = np.random.randint(5, size=N) + int64_mask
        values = np.random.rand(N, 2).astype(np.float32)
        default = np.random.rand(2).astype(np.float32)
        mask = np.arange(3) + int64_mask
        np.random.shuffle(mask)

        input_str = ['indices', 'values', 'default']
        input_data = [indices, values, default]
        if use_length and n > 1:
            input_str.append('lengths')
            input_data.append(lengths)
        output_str = ['output']

        op = core.CreateOperator(
            'SparseToDenseMask',
            input_str,
            output_str,
            mask=mask,
        )

        # Check over multiple devices
        self.assertDeviceChecks(
            dc, op, input_data, [0])
        # Gradient check for values
        self.assertGradientChecks(
            gc, op, input_data, 1, [0])

    @given(n=st.integers(1, 10), k=st.integers(1, 5),
           dim=st.integers(1, 3), **hu.gcs_cpu_only)
    @settings(deadline=10000)
    def test_sparse_to_dense_mask_high_dim(self, n, k, dim, gc, dc):
        lengths = np.random.randint(k, size=n).astype(np.int32) + 1
        N = sum(lengths)
        indices = np.random.randint(5, size=N)
        shape = np.random.randint(5, size=dim).astype(np.int32) + 1
        values = np.random.rand(*((N,) + tuple(shape))).astype(np.float32)
        default = np.random.rand(*shape).astype(np.float32)
        mask = np.arange(3)
        np.random.shuffle(mask)

        op = core.CreateOperator(
            'SparseToDenseMask',
            ['indices', 'values', 'default', 'lengths'],
            ['output'],
            mask=mask,
        )

        # Check over multiple devices
        self.assertDeviceChecks(
            dc, op, [indices, values, default, lengths], [0])
        # Gradient check for values
        self.assertGradientChecks(
            gc, op, [indices, values, default, lengths], 1, [0])


if __name__ == "__main__":
    import unittest
    unittest.main()
