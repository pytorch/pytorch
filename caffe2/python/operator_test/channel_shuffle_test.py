from __future__ import absolute_import, division, print_function, unicode_literals

import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core


class ChannelShuffleOpsTest(serial.SerializedTestCase):
    def _channel_shuffle_nchw_ref(self, X, group):
        dims = X.shape
        N = dims[0]
        C = dims[1]
        G = group
        K = int(C / G)
        X = X.reshape(N, G, K, np.prod(dims[2:]))
        Y = np.transpose(X, axes=(0, 2, 1, 3))
        return [Y.reshape(dims)]

    def _channel_shuffle_nhwc_ref(self, X, group):
        dims = X.shape
        N = dims[0]
        C = dims[-1]
        G = group
        K = int(C / G)
        X = X.reshape(N, np.prod(dims[1:-1]), G, K)
        Y = np.transpose(X, axes=(0, 1, 3, 2))
        return [Y.reshape(dims)]

    @serial.given(
        N=st.integers(0, 5),
        G=st.integers(1, 5),
        K=st.integers(1, 5),
        H=st.integers(1, 5),
        W=st.integers(1, 5),
        order=st.sampled_from(["NCHW", "NHWC"]),
        **hu.gcs
    )
    def test_channel_shuffle(self, N, G, K, H, W, order, gc, dc):
        C = G * K
        if order == "NCHW":
            X = np.random.randn(N, C, H, W).astype(np.float32)
        else:
            X = np.random.randn(N, H, W, C).astype(np.float32)

        op = core.CreateOperator("ChannelShuffle", ["X"], ["Y"], group=G, order=order)

        def channel_shuffle_ref(X):
            if order == "NCHW":
                return self._channel_shuffle_nchw_ref(X, G)
            else:
                return self._channel_shuffle_nhwc_ref(X, G)

        self.assertReferenceChecks(gc, op, [X], channel_shuffle_ref)
        self.assertGradientChecks(gc, op, [X], 0, [0])
        self.assertDeviceChecks(dc, op, [X], [0])
