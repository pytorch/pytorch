from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
from hypothesis import given
import hypothesis.strategies as st
import numpy as np


class TestBatchMomentsOp(serial.SerializedTestCase):
    def batch_moments_nchw_ref(self, X):
        dims = X.shape
        N = dims[0]
        C = dims[1]
        X = X.reshape(N, C, -1)
        mu = np.mean(X, axis=(0, 2))
        var = np.mean(np.square(X), axis=(0, 2))
        return [mu, var]

    def batch_moments_nhwc_ref(self, X):
        dims = X.shape
        C = dims[-1]
        X = X.reshape(-1, C)
        mu = np.mean(X, axis=0)
        var = np.mean(np.square(X), axis=0)
        return [mu, var]

    @serial.given(N=st.integers(1, 5), C=st.integers(1, 5),
            H=st.integers(1, 5), W=st.integers(1, 5),
            order=st.sampled_from(["NCHW", "NHWC"]), **hu.gcs)
    def test_batch_moments_2d(self, N, C, H, W, order, gc, dc):
        op = core.CreateOperator(
            "BatchMoments",
            ["X"],
            ["mu", "var"],
            order=order,
        )

        if order == "NCHW":
            X = np.random.randn(N, C, H, W).astype(np.float32)
        else:
            X = np.random.randn(N, H, W, C).astype(np.float32)

        def ref(X):
            if order == "NCHW":
                return self.batch_moments_nchw_ref(X)
            else:
                return self.batch_moments_nhwc_ref(X)

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=ref,
        )
        self.assertDeviceChecks(dc, op, [X], [0, 1])
        self.assertGradientChecks(gc, op, [X], 0, [0, 1])

    @given(N=st.integers(1, 5), C=st.integers(1, 5), T=st.integers(1, 3),
           H=st.integers(1, 3), W=st.integers(1, 3),
           order=st.sampled_from(["NCHW", "NHWC"]), **hu.gcs)
    def test_batch_moments_3d(self, N, C, T, H, W, order, gc, dc):
        op = core.CreateOperator(
            "BatchMoments",
            ["X"],
            ["mu", "var"],
            order=order,
        )

        if order == "NCHW":
            X = np.random.randn(N, C, T, H, W).astype(np.float32)
        else:
            X = np.random.randn(N, T, H, W, C).astype(np.float32)

        def ref(X):
            if order == "NCHW":
                return self.batch_moments_nchw_ref(X)
            else:
                return self.batch_moments_nhwc_ref(X)

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=ref,
        )
        self.assertDeviceChecks(dc, op, [X], [0, 1])
        self.assertGradientChecks(gc, op, [X], 0, [0, 1])
