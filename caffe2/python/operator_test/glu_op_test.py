from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
from hypothesis import assume, given, settings, HealthCheck
import hypothesis.strategies as st
import numpy as np

import unittest


@st.composite
def _glu_old_input(draw):
    dims = draw(st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=3))
    axis = draw(st.integers(min_value=0, max_value=len(dims)))
    # The axis dimension must be divisible by two
    axis_dim = 2 * draw(st.integers(min_value=1, max_value=2))
    dims.insert(axis, axis_dim)
    X = draw(hu.arrays(dims, np.float32, None))
    return (X, axis)


class TestGlu(serial.SerializedTestCase):
    @serial.given(
        X_axis=_glu_old_input(),
        **hu.gcs
    )
    def test_glu_old(self, X_axis, gc, dc):
        X, axis = X_axis

        def glu_ref(X):
            x1, x2 = np.split(X, [X.shape[axis] // 2], axis=axis)
            Y = x1 * (1. / (1. + np.exp(-x2)))
            return [Y]

        op = core.CreateOperator("Glu", ["X"], ["Y"], dim=axis)
        self.assertReferenceChecks(gc, op, [X], glu_ref)

if __name__ == "__main__":
    unittest.main()
