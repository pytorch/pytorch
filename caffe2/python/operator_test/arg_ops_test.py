from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import hypothesis.strategies as st
import numpy as np

from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial


class TestArgOps(serial.SerializedTestCase):
    @serial.given(
        X=hu.tensor(dtype=np.float32), axis=st.integers(-1, 5),
        keepdims=st.booleans(), **hu.gcs)
    def test_argmax(self, X, axis, keepdims, gc, dc):
        if axis >= len(X.shape):
            axis %= len(X.shape)
        op = core.CreateOperator(
            "ArgMax", ["X"], ["Indices"], axis=axis, keepdims=keepdims,
            device_option=gc)

        def argmax_ref(X):
            indices = np.argmax(X, axis=axis)
            if keepdims:
                out_dims = list(X.shape)
                out_dims[axis] = 1
                indices = indices.reshape(tuple(out_dims))
            return [indices]

        self.assertReferenceChecks(gc, op, [X], argmax_ref)
        self.assertDeviceChecks(dc, op, [X], [0])

    @serial.given(
        X=hu.tensor(dtype=np.float32), axis=st.integers(-1, 5),
        keepdims=st.booleans(), **hu.gcs)
    def test_argmin(self, X, axis, keepdims, gc, dc):
        if axis >= len(X.shape):
            axis %= len(X.shape)
        op = core.CreateOperator(
            "ArgMin", ["X"], ["Indices"], axis=axis, keepdims=keepdims,
            device_option=gc)

        def argmin_ref(X):
            indices = np.argmin(X, axis=axis)
            if keepdims:
                out_dims = list(X.shape)
                out_dims[axis] = 1
                indices = indices.reshape(tuple(out_dims))
            return [indices]

        self.assertReferenceChecks(gc, op, [X], argmin_ref)
        self.assertDeviceChecks(dc, op, [X], [0])


if __name__ == "__main__":
    import unittest
    unittest.main()
