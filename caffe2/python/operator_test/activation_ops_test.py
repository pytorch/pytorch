from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from hypothesis import given
import hypothesis.strategies as st

from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu


class TestActivations(hu.HypothesisTestCase):
    @given(X=hu.tensor(),
           alpha=st.floats(min_value=0.1, max_value=2.0),
           inplace=st.booleans(),
           **hu.gcs)
    def test_elu(self, X, alpha, inplace, gc, dc):
        # go away from the origin point to avoid kink problems
        X += 0.04 * np.sign(X)
        X[X == 0.0] += 0.04

        def elu_ref(X):
            Y = X.copy()
            neg_indices = X <= 0
            Y[neg_indices] = alpha * (np.exp(Y[neg_indices]) - 1)
            return (Y,)

        op = core.CreateOperator(
            "Elu",
            ["X"], ["Y" if not inplace else "X"],
            alpha=alpha)
        self.assertReferenceChecks(gc, op, [X], elu_ref)
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [X], [0])
        # Gradient check wrt X
        self.assertGradientChecks(gc, op, [X], 0, [0])

    @given(X=hu.tensor(min_dim=4, max_dim=4),
           alpha=st.floats(min_value=0.1, max_value=2.0),
           inplace=st.booleans(),
           shared=st.booleans(),
           order=st.sampled_from(["NCHW", "NHWC"]),
           seed=st.sampled_from([20, 100]),
           **hu.gcs)
    def test_prelu(self, X, alpha, inplace, shared, order, seed, gc, dc):
        np.random.seed(seed)
        W = np.random.randn(
            X.shape[1] if order == "NCHW" else X.shape[3]).astype(np.float32)

        if shared:
            W = np.random.randn(1).astype(np.float32)

        # go away from the origin point to avoid kink problems
        X += 0.04 * np.sign(X)
        X[X == 0.0] += 0.04

        def prelu_ref(X, W):
            Y = X.copy()
            W = W.reshape(1, -1, 1, 1) if order == "NCHW" \
                else W.reshape(1, 1, 1, -1)
            assert len(X.shape) == 4
            neg_indices = X <= 0
            assert len(neg_indices.shape) == 4
            assert X.shape == neg_indices.shape
            Y[neg_indices] = (Y * W)[neg_indices]
            return (Y,)

        op = core.CreateOperator(
            "PRelu", ["X", "W"], ["Y" if not inplace else "X"],
            alpha=alpha, order=order)
        self.assertReferenceChecks(gc, op, [X, W], prelu_ref)
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [X, W], [0])

        if not inplace:
            # Gradient check wrt X
            self.assertGradientChecks(gc, op, [X, W], 0, [0], stepsize=1e-2)
            # Gradient check wrt W
            self.assertGradientChecks(gc, op, [X, W], 1, [0], stepsize=1e-2)

    @given(X=hu.tensor(),
           alpha=st.floats(min_value=0.1, max_value=2.0),
           inplace=st.booleans(),
           **hu.gcs)
    def test_leaky_relu(self, X, alpha, inplace, gc, dc):
        # go away from the origin point to avoid kink problems
        X += 0.04 * np.sign(X)
        X[X == 0.0] += 0.04

        def leaky_relu_ref(X):
            Y = X.copy()
            neg_indices = X <= 0
            Y[neg_indices] = Y[neg_indices] * alpha
            return (Y,)

        op = core.CreateOperator(
            "LeakyRelu",
            ["X"], ["Y" if not inplace else "X"],
            alpha=alpha)
        self.assertReferenceChecks(gc, op, [X], leaky_relu_ref)
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [X], [0])

    @given(X=hu.tensor(),
           inplace=st.booleans(),
           **hu.gcs)
    def test_leaky_relu_default(self, X, inplace, gc, dc):
        # go away from the origin point to avoid kink problems
        X += 0.04 * np.sign(X)
        X[X == 0.0] += 0.04

        def leaky_relu_ref(X):
            Y = X.copy()
            neg_indices = X <= 0
            Y[neg_indices] = Y[neg_indices] * 0.01
            return (Y,)

        op = core.CreateOperator(
            "LeakyRelu",
            ["X"], ["Y" if not inplace else "X"])
        self.assertReferenceChecks(gc, op, [X], leaky_relu_ref)
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [X], [0])


if __name__ == "__main__":
    import unittest
    unittest.main()
