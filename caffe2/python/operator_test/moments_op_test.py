from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
import itertools as it


class TestMomentsOp(hu.HypothesisTestCase):
    def run_moments_test(self, X, axes, keepdims, gc, dc):
        if axes is None:
            op = core.CreateOperator(
                "Moments",
                ["X"],
                ["mean", "variance"],
                keepdims=keepdims,
            )
        else:
            op = core.CreateOperator(
                "Moments",
                ["X"],
                ["mean", "variance"],
                axes=axes,
                keepdims=keepdims,
            )

        def ref(X):
            mean = np.mean(X, axis=None if axes is None else tuple(
                axes), keepdims=keepdims)
            variance = np.var(X, axis=None if axes is None else tuple(
                axes), keepdims=keepdims)
            return [mean, variance]

        self.assertReferenceChecks(gc, op, [X], ref)
        self.assertDeviceChecks(dc, op, [X], [0, 1])
        self.assertGradientChecks(gc, op, [X], 0, [0, 1])

    @given(X=hu.tensor(dtype=np.float32), keepdims=st.booleans(),
           num_axes=st.integers(1, 4), **hu.gcs)
    def test_moments(self, X, keepdims, num_axes, gc, dc):
        self.run_moments_test(X, None, keepdims, gc, dc)
        num_dims = len(X.shape)
        if num_dims < num_axes:
            self.run_moments_test(X, range(num_dims), keepdims, gc, dc)
        else:
            for axes in it.combinations(range(num_dims), num_axes):
                self.run_moments_test(X, axes, keepdims, gc, dc)
