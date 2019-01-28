from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial

from hypothesis import given
import hypothesis.strategies as st
import numpy as np


class TestGroupNormOp(serial.SerializedTestCase):
    def group_norm_nchw_ref(self, X, gamma, beta, group, epsilon):
        dims = X.shape
        N = dims[0]
        C = dims[1]
        G = group
        D = int(C / G)
        X = X.reshape(N, G, D, -1)
        mu = np.mean(X, axis=(2, 3), keepdims=True)
        std = np.sqrt((np.var(X, axis=(2, 3), keepdims=True) + epsilon))
        gamma = gamma.reshape(G, D, 1)
        beta = beta.reshape(G, D, 1)
        Y = gamma * (X - mu) / std + beta
        return [Y.reshape(dims), mu.reshape(N, G), (1.0 / std).reshape(N, G)]

    def group_norm_nhwc_ref(self, X, gamma, beta, group, epsilon):
        dims = X.shape
        N = dims[0]
        C = dims[-1]
        G = group
        D = int(C / G)
        X = X.reshape(N, -1, G, D)
        mu = np.mean(X, axis=(1, 3), keepdims=True)
        std = np.sqrt((np.var(X, axis=(1, 3), keepdims=True) + epsilon))
        gamma = gamma.reshape(G, D)
        beta = beta.reshape(G, D)
        Y = gamma * (X - mu) / std + beta
        return [Y.reshape(dims), mu.reshape(N, G), (1.0 / std).reshape(N, G)]

    @serial.given(
        N=st.integers(1, 5), G=st.integers(1, 5), D=st.integers(1, 5),
        H=st.integers(2, 5), W=st.integers(2, 5),
        epsilon=st.floats(min_value=1e-5, max_value=1e-4),
        order=st.sampled_from(["NCHW", "NHWC"]), **hu.gcs)
    def test_group_norm_2d(
            self, N, G, D, H, W, epsilon, order, gc, dc):
        op = core.CreateOperator(
            "GroupNorm",
            ["X", "gamma", "beta"],
            ["Y", "mean", "inv_std"],
            group=G,
            epsilon=epsilon,
            order=order,
        )

        C = G * D
        if order == "NCHW":
            X = np.random.randn(N, C, H, W).astype(np.float32) + 1.0
        else:
            X = np.random.randn(N, H, W, C).astype(np.float32) + 1.0
        gamma = np.random.randn(C).astype(np.float32)
        beta = np.random.randn(C).astype(np.float32)
        inputs = [X, gamma, beta]

        def ref_op(X, gamma, beta):
            if order == "NCHW":
                return self.group_norm_nchw_ref(X, gamma, beta, G, epsilon)
            else:
                return self.group_norm_nhwc_ref(X, gamma, beta, G, epsilon)
        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=inputs,
            reference=ref_op,
            threshold=5e-3,
        )
        self.assertDeviceChecks(dc, op, inputs, [0, 1, 2])

    @given(N=st.integers(1, 5), G=st.integers(1, 3), D=st.integers(2, 3),
           T=st.integers(2, 4), H=st.integers(2, 4), W=st.integers(2, 4),
           epsilon=st.floats(min_value=1e-5, max_value=1e-4),
           order=st.sampled_from(["NCHW", "NHWC"]), **hu.gcs)
    def test_group_norm_3d(
            self, N, G, D, T, H, W, epsilon, order, gc, dc):
        op = core.CreateOperator(
            "GroupNorm",
            ["X", "gamma", "beta"],
            ["Y", "mean", "inv_std"],
            group=G,
            epsilon=epsilon,
            order=order,
        )

        C = G * D
        if order == "NCHW":
            X = np.random.randn(N, C, T, H, W).astype(np.float32) + 1.0
        else:
            X = np.random.randn(N, T, H, W, C).astype(np.float32) + 1.0
        gamma = np.random.randn(C).astype(np.float32)
        beta = np.random.randn(C).astype(np.float32)
        inputs = [X, gamma, beta]

        def ref_op(X, gamma, beta):
            if order == "NCHW":
                return self.group_norm_nchw_ref(X, gamma, beta, G, epsilon)
            else:
                return self.group_norm_nhwc_ref(X, gamma, beta, G, epsilon)
        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=inputs,
            reference=ref_op,
            threshold=5e-3,
        )
        self.assertDeviceChecks(dc, op, inputs, [0, 1, 2])

    @given(N=st.integers(1, 5), G=st.integers(1, 5), D=st.integers(2, 2),
           H=st.integers(2, 5), W=st.integers(2, 5),
           epsilon=st.floats(min_value=1e-5, max_value=1e-4),
           order=st.sampled_from(["NCHW", "NHWC"]), **hu.gcs)
    def test_group_norm_grad(
            self, N, G, D, H, W, epsilon, order, gc, dc):
        op = core.CreateOperator(
            "GroupNorm",
            ["X", "gamma", "beta"],
            ["Y", "mean", "inv_std"],
            group=G,
            epsilon=epsilon,
            order=order,
        )

        C = G * D
        X = np.arange(N * C * H * W).astype(np.float32)
        np.random.shuffle(X)
        if order == "NCHW":
            X = X.reshape((N, C, H, W))
        else:
            X = X.reshape((N, H, W, C))
        gamma = np.random.randn(C).astype(np.float32)
        beta = np.random.randn(C).astype(np.float32)
        inputs = [X, gamma, beta]
        for i in range(len(inputs)):
            self.assertGradientChecks(gc, op, inputs, i, [0])
