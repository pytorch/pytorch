




from caffe2.python import core
from hypothesis import given, settings
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial

import numpy as np
import unittest


class TestTrigonometricOp(serial.SerializedTestCase):
    @given(
        X=hu.tensor(elements=hu.floats(min_value=-0.7, max_value=0.7)),
        **hu.gcs)
    @settings(deadline=None, max_examples=50)
    def test_acos(self, X, gc, dc):
        self.assertTrigonometricChecks("Acos", X, lambda x: (np.arccos(X),), gc, dc)

    @given(
        X=hu.tensor(elements=hu.floats(min_value=-0.7, max_value=0.7)),
        **hu.gcs)
    @settings(deadline=None, max_examples=50)
    def test_asin(self, X, gc, dc):
        self.assertTrigonometricChecks("Asin", X, lambda x: (np.arcsin(X),), gc, dc)

    @given(
        X=hu.tensor(elements=hu.floats(min_value=-100, max_value=100)),
        **hu.gcs)
    @settings(deadline=None, max_examples=50)
    def test_atan(self, X, gc, dc):
        self.assertTrigonometricChecks("Atan", X, lambda x: (np.arctan(X),), gc, dc)

    @given(
        X=hu.tensor(elements=hu.floats(min_value=-0.5, max_value=0.5)),
        **hu.gcs)
    @settings(deadline=None, max_examples=50)
    def test_tan(self, X, gc, dc):
        self.assertTrigonometricChecks("Tan", X, lambda x: (np.tan(X),), gc, dc)

    def assertTrigonometricChecks(self, op_name, input, reference, gc, dc):
        op = core.CreateOperator(op_name, ["X"], ["Y"])
        self.assertReferenceChecks(gc, op, [input], reference)
        self.assertDeviceChecks(dc, op, [input], [0])
        self.assertGradientChecks(gc, op, [input], 0, [0])


if __name__ == "__main__":
    unittest.main()
