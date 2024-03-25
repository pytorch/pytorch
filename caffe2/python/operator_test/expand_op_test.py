




from caffe2.python import core
from hypothesis import given, settings

import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np


class TestExpandOp(serial.SerializedTestCase):
    def _rand_shape(self, X_shape, max_length):
        length = np.random.randint(max_length)
        shape = np.ones(length, dtype=np.int64)
        i = len(X_shape) - 1
        for j in reversed(range(length)):
            if i >= 0:
                k = np.random.choice([1, X_shape[i]])
                i -= 1
            else:
                k = np.random.randint(3) + 1
            shape[j] = k
        return shape

    def _run_expand_op_test(self, X, shape, gc, dc):
        shape = np.array(shape)
        op = core.CreateOperator(
            'Expand',
            ["X", "shape"],
            ["Y"],
        )
        def ref(X, shape):
            return (X * np.ones(abs(shape)),)

        self.assertReferenceChecks(gc, op, [X, shape], ref)
        self.assertDeviceChecks(dc, op, [X, shape], [0])
        self.assertGradientChecks(gc, op, [X, shape], 0, [0])

    @serial.given(X=hu.tensor(max_dim=5, dtype=np.float32),
           **hu.gcs)
    def test_expand_rand_shape(self, X, gc, dc):
        shape = self._rand_shape(X.shape, 5)
        self._run_expand_op_test(X, shape, gc, dc)

    @given(X=st.sampled_from([np.ones([1, 3, 1]),
                             np.ones([3, 1, 3]),
                             np.ones([1, 3])]),
           **hu.gcs)
    def test_expand_nonrand_shape1(self, X, gc, dc):
        self._run_expand_op_test(X, [3, 1, 3], gc, dc)
        self._run_expand_op_test(X, [3, -1, 3], gc, dc)


    @given(X=st.sampled_from([np.ones([4, 4, 2, 1]),
                             np.ones([1, 4, 1, 2]),
                             np.ones([4, 1, 2])]),
           **hu.gcs)
    @settings(deadline=10000)
    def test_expand_nonrand_shape2(self, X, gc, dc):
        self._run_expand_op_test(X, [4, 1, 2, 2], gc, dc)
        self._run_expand_op_test(X, [4, -1, 2, 2], gc, dc)
