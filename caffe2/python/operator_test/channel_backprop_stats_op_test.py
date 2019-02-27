from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial

from hypothesis import given
import hypothesis.strategies as st
import numpy as np
import unittest


class TestChannelBackpropStatsOp(serial.SerializedTestCase):
    def channel_backprop_stats_nchw_ref(self, dY, X, mean, rstd):
        dims = X.shape
        N = dims[0]
        C = dims[1]
        dY = dY.reshape(N, C, -1)
        X = X.reshape(N, C, -1)
        mean = mean.reshape(C, 1)
        rstd = rstd.reshape(C, 1)
        dscale = np.sum(dY * (X - mean) * rstd, axis=(0, 2), keepdims=False)
        dbias = np.sum(dY, axis=(0, 2), keepdims=False)
        return (dscale, dbias)

    def channel_backprop_stats_nhwc_ref(self, dY, X, mean, rstd):
        dims = X.shape
        N = dims[0]
        C = dims[-1]
        dY = dY.reshape(-1, C)
        X = X.reshape(-1, C)
        dscale = np.sum(dY * (X - mean) * rstd, axis=0, keepdims=False)
        dbias = np.sum(dY, axis=0, keepdims=False)
        return (dscale, dbias)

    @serial.given(
        N=st.integers(1, 5), C=st.integers(1, 10), H=st.integers(1, 12),
        W=st.integers(1, 12), order=st.sampled_from(["NCHW", "NHWC"]), **hu.gcs)
    def test_channel_backprop_stats_2d(self, N, C, H, W, order, gc, dc):
        op = core.CreateOperator(
            "ChannelBackpropStats",
            ["X", "mean", "rstd", "dY"],
            ["dscale", "dbias"],
            order=order,
        )

        def ref_op(X, mean, rstd, dY):
            if order == "NCHW":
                return self.channel_backprop_stats_nchw_ref(dY, X, mean, rstd)
            else:
                return self.channel_backprop_stats_nhwc_ref(dY, X, mean, rstd)

        X = np.random.randn(N, C, H, W).astype(np.float32)
        mean = np.mean(X, axis=(0, 2, 3), keepdims=False)
        rstd = 1.0 / np.sqrt(np.var(X, axis=(0, 2, 3), keepdims=False) + 1e-5)
        dY = np.random.randn(N, C, H, W).astype(np.float32)
        if order == "NHWC":
            dY = np.transpose(dY, [0, 2, 3, 1])
            X = np.transpose(X, [0, 2, 3, 1])

        self.assertReferenceChecks(
            gc, op, [X, mean, rstd, dY], reference=ref_op)
        self.assertDeviceChecks(dc, op, [X, mean, rstd, dY], [0, 1])

    @serial.given(
        N=st.integers(1, 5), C=st.integers(1, 10), D=st.integers(1, 6),
        H=st.integers(1, 6), W=st.integers(1, 6),
        order=st.sampled_from(["NCHW", "NHWC"]), **hu.gcs)
    def test_channel_backprop_stats_3d(self, N, C, D, H, W, order, gc, dc):
        op = core.CreateOperator(
            "ChannelBackpropStats",
            ["X", "mean", "rstd", "dY"],
            ["dscale", "dbias"],
            order=order,
        )

        def ref_op(X, mean, rstd, dY):
            if order == "NCHW":
                return self.channel_backprop_stats_nchw_ref(dY, X, mean, rstd)
            else:
                return self.channel_backprop_stats_nhwc_ref(dY, X, mean, rstd)

        X = np.random.randn(N, C, D, H, W).astype(np.float32)
        mean = np.mean(X, axis=(0, 2, 3, 4), keepdims=False)
        rstd = 1.0 / np.sqrt(np.var(X, axis=(0, 2, 3, 4),
                                    keepdims=False) + 1e-5)
        dY = np.random.randn(N, C, D, H, W).astype(np.float32)
        if order == "NHWC":
            dY = np.transpose(dY, [0, 2, 3, 4, 1])
            X = np.transpose(X, [0, 2, 3, 4, 1])

        self.assertReferenceChecks(
            gc, op, [X, mean, rstd, dY], reference=ref_op)
        self.assertDeviceChecks(dc, op, [X, mean, rstd, dY], [0, 1])


if __name__ == "__main__":
    unittest.main()
