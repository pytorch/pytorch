



from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial

from hypothesis import given, settings
import hypothesis.strategies as st
import numpy as np

import unittest


class TestChannelStatsOp(serial.SerializedTestCase):
    def channel_stats_nchw_ref(self, X):
        dims = X.shape
        N = dims[0]
        C = dims[1]
        X = X.reshape(N, C, -1)
        sum1 = np.sum(X, axis=(0, 2), keepdims=False)
        sum2 = np.sum(X**2, axis=(0, 2), keepdims=False)
        return (sum1, sum2)

    def channel_stats_nhwc_ref(self, X):
        dims = X.shape
        N = dims[0]
        C = dims[-1]
        X = X.reshape(N, -1, C)
        sum1 = np.sum(X, axis=(0, 1), keepdims=False)
        sum2 = np.sum(X**2, axis=(0, 1), keepdims=False)
        return (sum1, sum2)

    @given(
        N=st.integers(1, 5), C=st.integers(1, 10), H=st.integers(1, 12),
        W=st.integers(1, 12), order=st.sampled_from(["NCHW", "NHWC"]), **hu.gcs)
    @settings(deadline=10000)
    def test_channel_stats_2d(self, N, C, H, W, order, gc, dc):
        op = core.CreateOperator(
            "ChannelStats",
            ["X"],
            ["sum", "sumsq"],
            order=order,
        )

        def ref_op(X):
            if order == "NCHW":
                return self.channel_stats_nchw_ref(X)
            else:
                return self.channel_stats_nhwc_ref(X)

        X = np.random.randn(N, C, H, W).astype(np.float32)
        if order == "NHWC":
            X = np.transpose(X, [0, 2, 3, 1])

        self.assertReferenceChecks(gc, op, [X], reference=ref_op)
        self.assertDeviceChecks(dc, op, [X], [0, 1])

    @given(
        N=st.integers(1, 5), C=st.integers(1, 10), D=st.integers(1, 6),
        H=st.integers(1, 6), W=st.integers(1, 6),
        order=st.sampled_from(["NCHW", "NHWC"]), **hu.gcs)
    @settings(deadline=10000)
    def test_channel_stats_3d(self, N, C, D, H, W, order, gc, dc):
        op = core.CreateOperator(
            "ChannelStats",
            ["X"],
            ["sum", "sumsq"],
            order=order,
        )

        def ref_op(X):
            if order == "NCHW":
                return self.channel_stats_nchw_ref(X)
            else:
                return self.channel_stats_nhwc_ref(X)

        X = np.random.randn(N, C, D, H, W).astype(np.float32)
        if order == "NHWC":
            X = np.transpose(X, [0, 2, 3, 4, 1])

        self.assertReferenceChecks(gc, op, [X], reference=ref_op)
        self.assertDeviceChecks(dc, op, [X], [0, 1])

if __name__ == "__main__":
    unittest.main()
