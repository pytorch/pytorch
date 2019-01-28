from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import numpy as np

import unittest


class TestGivenTensorByteStringToUInt8FillOps(hu.HypothesisTestCase):
    @given(X=hu.tensor(min_dim=1, max_dim=4, dtype=np.int32),
           **hu.gcs)
    def test_given_tensor_byte_string_to_uint8_fill(self, X, gc, dc):
        X = X.astype(np.uint8)
        print('X: ', str(X))
        op = core.CreateOperator(
            "GivenTensorByteStringToUInt8Fill",
            [], ["Y"],
            shape=X.shape,
            dtype=core.DataType.STRING,
            values=[X.tobytes()],
        )

        def constant_fill(*args, **kw):
            return [X]

        self.assertReferenceChecks(gc, op, [], constant_fill)
        self.assertDeviceChecks(dc, op, [], [0])

    @given(**hu.gcs)
    def test_empty_given_tensor_byte_string_to_uint8_fill(self, gc, dc):
        X = np.array([], dtype=np.uint8)
        print('X: ', str(X))
        op = core.CreateOperator(
            "GivenTensorByteStringToUInt8Fill",
            [], ["Y"],
            shape=X.shape,
            values=[X.tobytes()],
        )

        def constant_fill(*args, **kw):
            return [X]

        self.assertReferenceChecks(gc, op, [], constant_fill)
        self.assertDeviceChecks(dc, op, [], [0])


if __name__ == "__main__":
    unittest.main()
