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
    @given(n=st.integers(1, 100), **hu.gcs_cpu_only)
    def test_piecewise_linear_transform_general(self, n, gc, dc):
        W = np.random.uniform(-1, 1, (2, n)).astype(np.float32)
        b = np.random.uniform(-1, 1, (2, n)).astype(np.float32)
        # make sure bucket range are increating!
        bucket_range = np.random.uniform(0.1, 0.9,
                                         (2, n + 1)).astype(np.float32)
        bucket_base = np.array(list(range(n + 1)))
        bucket_range[0, :] = bucket_range[0, :] + bucket_base
        bucket_range[1, :] = bucket_range[1, :] + bucket_base
        # make x[i] inside bucket i, for the ease of testing
        X = np.random.uniform(0, 0.9, (n, 2)).astype(np.float32)
        for i in range(len(X)):
            X[i][0] = X[i][0] * bucket_range[0][i] + \
                (1 - X[i][0]) * bucket_range[0][i + 1]
            X[i][1] = X[i][1] * bucket_range[1][i] + \
                (1 - X[i][1]) * bucket_range[1][i + 1]

        op = core.CreateOperator(
            "PiecewiseLinearTransform", ["X"], ["Y"],
            bounds=bucket_range.flatten().tolist(),
            slopes=W.flatten().tolist(),
            intercepts=b.flatten().tolist(),
            pieces=n
        )

        def piecewise(x, *args, **kw):
            return [W.transpose() * x + b.transpose()]

        self.assertReferenceChecks(gc, op, [X], piecewise)
        self.assertDeviceChecks(dc, op, [X], [0])

    @given(n=st.integers(1, 100), **hu.gcs_cpu_only)
    def test_piecewise_linear_transform_binary(self, n, gc, dc):
        W = np.random.uniform(-1, 1, size=n).astype(np.float32)
        b = np.random.uniform(-1, 1, size=n).astype(np.float32)
        bucket_range = np.random.uniform(
            0, 1, n + 1).astype(np.float32)
        bucket_range.sort()

        # make x[i] inside bucket i, for the ease of testing
        X = np.random.uniform(0, 0.9, (n, 2)).astype(np.float32)
        for i in range(len(X)):
            X[i][1] = X[i][1] * bucket_range[i] + \
                (1 - X[i][1]) * bucket_range[i + 1]
        X[:, 0] = 1 - X[:, 1]

        op = core.CreateOperator(
            "PiecewiseLinearTransform", ["X"], ["Y"],
            bounds=bucket_range.flatten().tolist(),
            slopes=W.flatten().tolist(),
            intercepts=b.flatten().tolist(),
            pieces=n,
            binary=True,
        )

        def piecewise(x):
            positive = W.transpose() * x[:, 1] + b.transpose()
            return [np.vstack((1 - positive, positive)).transpose()]

        self.assertReferenceChecks(gc, op, [X], piecewise)
        self.assertDeviceChecks(dc, op, [X], [0])


if __name__ == "__main__":
    unittest.main()
