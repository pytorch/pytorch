from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given
import hypothesis.strategies as st
import caffe2.python.hypothesis_test_util as hu
import numpy as np

import unittest


class TestGivenTensorFillOps(hu.HypothesisTestCase):
    @given(X=hu.tensor(min_dim=1, max_dim=4, dtype=np.int32),
           t=st.sampled_from([
               (core.DataType.BOOL, np.bool_, "GivenTensorFill"),
               (core.DataType.INT32, np.int32, "GivenTensorFill"),
               (core.DataType.FLOAT, np.float32, "GivenTensorFill"),
               (core.DataType.INT32, np.int32, "GivenTensorIntFill"),
               (core.DataType.INT64, np.int64, "GivenTensorInt64Fill"),
               (core.DataType.BOOL, np.bool_, "GivenTensorBoolFill"),
               (core.DataType.DOUBLE, np.double, "GivenTensorDoubleFill"),
               (core.DataType.INT32, np.double, "GivenTensorDoubleFill"),
           ]),
           **hu.gcs)
    def test_given_tensor_fill(self, X, t, gc, dc):
        X = X.astype(t[1])
        print('X: ', str(X))
        op = core.CreateOperator(
            t[2], [], ["Y"],
            shape=X.shape,
            dtype=t[0],
            values=X.reshape((1, X.size)),
        )

        def constant_fill(*args, **kw):
            return [X]

        self.assertReferenceChecks(gc, op, [], constant_fill)
        self.assertDeviceChecks(dc, op, [], [0])


if __name__ == "__main__":
    unittest.main()
