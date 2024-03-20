




import math

from caffe2.python import core
from hypothesis import given, settings
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial

import numpy as np
import unittest


class TestErfOp(serial.SerializedTestCase):
    @given(
        X=hu.tensor(elements=hu.floats(min_value=-0.7, max_value=0.7)),
        **hu.gcs)
    @settings(deadline=10000)
    def test_erf(self, X, gc, dc):
        op = core.CreateOperator('Erf', ["X"], ["Y"])
        self.assertReferenceChecks(gc, op, [X], lambda x: (np.vectorize(math.erf)(X),))
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0])


if __name__ == "__main__":
    unittest.main()
