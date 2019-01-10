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


class TestPiecewiseLinearTransform(hu.HypothesisTestCase):
    def constrain(self, v, min_val, max_val):
        def constrain_internal(x):
            return min(max(x, min_val), max_val)
        return np.array([constrain_internal(x) for x in v])

    def transform(self, x, bounds, slopes, intercepts):
        n = len(slopes)
        x_ = self.constrain(x, bounds[0], bounds[-1])
        index = np.minimum(
            np.maximum(
                np.searchsorted(bounds, x_) - 1,
                0
            ),
            n - 1
        )
        y = slopes[index] * x_ + intercepts[index]
        return y

    @given(n=st.integers(1, 100), **hu.gcs)
    def test_multi_predictions_params_from_arg(self, n, gc, dc):
        slopes = np.random.uniform(-1, 1, (2, n)).astype(np.float32)
        intercepts = np.random.uniform(-1, 1, (2, n)).astype(np.float32)
        bounds = np.random.uniform(0.1, 0.9,
                                   (2, n + 1)).astype(np.float32)
        bounds.sort()
        X = np.random.uniform(0, 1, (n, 2)).astype(np.float32)

        op = core.CreateOperator(
            "PiecewiseLinearTransform", ["X"], ["Y"],
            bounds=bounds.flatten().tolist(),
            slopes=slopes.flatten().tolist(),
            intercepts=intercepts.flatten().tolist(),
        )

        def piecewise(x, *args, **kw):
            x_0 = self.transform(
                x[:, 0], bounds[0, :], slopes[0, :], intercepts[0, :])
            x_1 = self.transform(
                x[:, 1], bounds[1, :], slopes[1, :], intercepts[1, :])

            return [np.vstack((x_0, x_1)).transpose()]

        self.assertReferenceChecks(gc, op, [X], piecewise)
        self.assertDeviceChecks(dc, op, [X], [0])

    @given(n=st.integers(1, 100), **hu.gcs)
    def test_binary_predictions_params_from_arg(self, n, gc, dc):
        slopes = np.random.uniform(-1, 1, size=n).astype(np.float32)
        intercepts = np.random.uniform(-1, 1, size=n).astype(np.float32)
        bounds = np.random.uniform(0.1, 0.9, n + 1).astype(np.float32)
        bounds.sort()

        X = np.random.uniform(0, 1, (n, 2)).astype(np.float32)
        X[:, 0] = 1 - X[:, 1]

        op = core.CreateOperator(
            "PiecewiseLinearTransform", ["X"], ["Y"],
            bounds=bounds.flatten().tolist(),
            slopes=slopes.flatten().tolist(),
            intercepts=intercepts.flatten().tolist(),
            pieces=n,
            binary=True,
        )

        def piecewise(x):
            x_ = self.transform(x[:, 1], bounds, slopes, intercepts)
            return [np.vstack((1 - x_, x_)).transpose()]

        self.assertReferenceChecks(gc, op, [X], piecewise)
        self.assertDeviceChecks(dc, op, [X], [0])

    @given(n=st.integers(1, 100), **hu.gcs)
    def test_multi_predictions_params_from_input(self, n, gc, dc):
        slopes = np.random.uniform(-1, 1, (2, n)).astype(np.float32)
        intercepts = np.random.uniform(-1, 1, (2, n)).astype(np.float32)
        bounds = np.random.uniform(0.1, 0.9,
                                   (2, n + 1)).astype(np.float32)
        bounds.sort()
        X = np.random.uniform(0, 1, (n, 2)).astype(np.float32)

        op = core.CreateOperator(
            "PiecewiseLinearTransform",
            ["X", "bounds", "slopes", "intercepts"],
            ["Y"],
        )

        def piecewise(x, bounds, slopes, intercepts):
            x_0 = self.transform(
                x[:, 0], bounds[0, :], slopes[0, :], intercepts[0, :])
            x_1 = self.transform(
                x[:, 1], bounds[1, :], slopes[1, :], intercepts[1, :])

            return [np.vstack((x_0, x_1)).transpose()]

        self.assertReferenceChecks(
            gc, op, [X, bounds, slopes, intercepts], piecewise)
        self.assertDeviceChecks(dc, op, [X, bounds, slopes, intercepts], [0])

    @given(n=st.integers(1, 100), **hu.gcs)
    def test_binary_predictions_params_from_input(self, n, gc, dc):
        slopes = np.random.uniform(-1, 1, size=n).astype(np.float32)
        intercepts = np.random.uniform(-1, 1, size=n).astype(np.float32)
        bounds = np.random.uniform(0.1, 0.9, n + 1).astype(np.float32)
        bounds.sort()

        X = np.random.uniform(0, 1, (n, 2)).astype(np.float32)
        X[:, 0] = 1 - X[:, 1]

        op = core.CreateOperator(
            "PiecewiseLinearTransform",
            ["X", "bounds", "slopes", "intercepts"],
            ["Y"],
            binary=True,
        )

        def piecewise(x, bounds, slopes, intercepts):
            x_ = self.transform(x[:, 1], bounds, slopes, intercepts)
            return [np.vstack((1 - x_, x_)).transpose()]

        self.assertReferenceChecks(
            gc, op, [X, bounds, slopes, intercepts], piecewise)
        self.assertDeviceChecks(dc, op, [X, bounds, slopes, intercepts], [0])

    @given(n=st.integers(1, 100), **hu.gcs)
    def test_1D_predictions_params_from_input(self, n, gc, dc):
        slopes = np.random.uniform(-1, 1, size=n).astype(np.float32)
        intercepts = np.random.uniform(-1, 1, size=n).astype(np.float32)
        bounds = np.random.uniform(0.1, 0.9, n + 1).astype(np.float32)
        bounds.sort()

        X = np.random.uniform(0, 1, size=n).astype(np.float32)

        op = core.CreateOperator(
            "PiecewiseLinearTransform",
            ["X", "bounds", "slopes", "intercepts"],
            ["Y"],
            binary=True,
        )

        def piecewise(x, bounds, slopes, intercepts):
            x_ = self.transform(x, bounds, slopes, intercepts)
            return [x_]

        self.assertReferenceChecks(
            gc, op, [X, bounds, slopes, intercepts], piecewise)
        self.assertDeviceChecks(dc, op, [X, bounds, slopes, intercepts], [0])


if __name__ == "__main__":
    unittest.main()
