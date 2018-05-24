from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given
from hypothesis import strategies as st
import caffe2.python.hypothesis_test_util as hu

import numpy as np
import unittest


class TestTrigonometricOp(hu.HypothesisTestCase):
    @given(X=hu.tensor(elements=st.floats(min_value=-0.7, max_value=0.7)))
    def test_acos(self, X):
        self.assertTrigonometricChecks("Acos", X, lambda x: (np.arccos(X),))

    @given(X=hu.tensor(elements=st.floats(min_value=-0.7, max_value=0.7)))
    def test_asin(self, X):
        self.assertTrigonometricChecks("Asin", X, lambda x: (np.arcsin(X),))

    @given(X=hu.tensor(elements=st.floats(min_value=-100, max_value=100)))
    def test_atan(self, X):
        self.assertTrigonometricChecks("Atan", X, lambda x: (np.arctan(X),))

    @given(X=hu.tensor(elements=st.floats(min_value=-0.5, max_value=0.5)))
    def test_tan(self, X):
        self.assertTrigonometricChecks("Tan", X, lambda x: (np.tan(X),))

    @given(**hu.gcs)
    def assertTrigonometricChecks(self, op_name, input, reference, gc, dc):
        op = core.CreateOperator(op_name, ["X"], ["Y"])
        self.assertReferenceChecks(gc, op, [input], reference)
        self.assertDeviceChecks(dc, op, [input], [0])
        self.assertGradientChecks(gc, op, [input], 0, [0])


if __name__ == "__main__":
    unittest.main()
