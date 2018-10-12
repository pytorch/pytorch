from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given
from hypothesis import strategies as st
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial

import numpy as np
import unittest


class TestTrigonometricOp(serial.SerializedTestCase):
    @serial.given(
        X=hu.tensor(elements=st.floats(min_value=-0.7, max_value=0.7)),
        **hu.gcs)
    def test_acos(self, X, gc, dc):
        self.assertTrigonometricChecks("Acos", X, lambda x: (np.arccos(X),), gc, dc)

    @serial.given(
        X=hu.tensor(elements=st.floats(min_value=-0.7, max_value=0.7)),
        **hu.gcs)
    def test_asin(self, X, gc, dc):
        self.assertTrigonometricChecks("Asin", X, lambda x: (np.arcsin(X),), gc, dc)

    @serial.given(
        X=hu.tensor(elements=st.floats(min_value=-100, max_value=100)),
        **hu.gcs)
    def test_atan(self, X, gc, dc):
        self.assertTrigonometricChecks("Atan", X, lambda x: (np.arctan(X),), gc, dc)

    @serial.given(
        X=hu.tensor(elements=st.floats(min_value=-0.5, max_value=0.5)),
        **hu.gcs)
    def test_tan(self, X, gc, dc):
        self.assertTrigonometricChecks("Tan", X, lambda x: (np.tan(X),), gc, dc)

    def assertTrigonometricChecks(self, op_name, input, reference, gc, dc):
        op = core.CreateOperator(op_name, ["X"], ["Y"])
        self.assertReferenceChecks(gc, op, [input], reference)
        self.assertDeviceChecks(dc, op, [input], [0])
        self.assertGradientChecks(gc, op, [input], 0, [0])


if __name__ == "__main__":
    unittest.main()
