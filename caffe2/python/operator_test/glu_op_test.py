from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import assume, given, settings, HealthCheck
import hypothesis.strategies as st
import caffe2.python.hypothesis_test_util as hu
import numpy as np

import unittest


class TestGlu(hu.HypothesisTestCase):
    # Suppress filter_too_much health check.
    # Reproduce by commenting @settings and uncommenting @seed.
    # @seed(302934307671667531413257853548643485645)
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    @given(
        X=hu.tensor(),
        axis=st.integers(min_value=0, max_value=3),
        **hu.gcs
    )
    def test_glu_old(self, X, axis, gc, dc):
        def glu_ref(X):
            x1, x2 = np.split(X, [X.shape[axis] // 2], axis=axis)
            Y = x1 * (1. / (1. + np.exp(-x2)))
            return [Y]

        # Test only valid tensors.
        assume(axis < X.ndim)
        assume(X.shape[axis] % 2 == 0)
        op = core.CreateOperator("Glu", ["X"], ["Y"], dim=axis)
        self.assertReferenceChecks(gc, op, [X], glu_ref)

if __name__ == "__main__":
    unittest.main()
