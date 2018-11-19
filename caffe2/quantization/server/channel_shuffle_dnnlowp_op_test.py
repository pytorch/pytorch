from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import caffe2.python.hypothesis_test_util as hu
from caffe2.python import core, dyndep, workspace
from hypothesis import given
import hypothesis.strategies as st

dyndep.InitOpsLibrary("//caffe2/caffe2/quantization/server:dnnlowp_ops")


class DNNLowPChannelShuffleOpsTest(hu.HypothesisTestCase):
    @given(
        channels_per_group=st.integers(min_value=1, max_value=5),
        groups=st.sampled_from([1, 4, 8, 9]),
        n=st.integers(min_value=1, max_value=2),
        order=st.sampled_from(["NCHW", "NHWC"]),
        **hu.gcs_cpu_only)
    def test_channel_shuffle(self, channels_per_group, groups, n, order, gc, dc):
        X = np.round(np.random.rand(
            n, channels_per_group * groups, 5, 6) * 255).astype(np.float32)
        X[0, 0, 0, 0] = 0
        X[0, 0, 0, 1] = 255
        if order == "NHWC":
            # NCHW -> NHWC
            X = X.transpose((0, 2, 3, 1))

        net = core.Net("test_net")

        quantize = core.CreateOperator(
            "Quantize",
            ["X"],
            ["X_q"],
            engine="DNNLOWP",
        )

        channel_shuffle = core.CreateOperator(
            "ChannelShuffle", ["X_q"], ["Y_q"],
            group=groups,
            kernel=1,
            order=order,
            engine="DNNLOWP",
        )

        dequantize = core.CreateOperator(
            "Dequantize",
            ["Y_q"],
            ["Y"],
            engine="DNNLOWP",
        )

        net.Proto().op.extend([quantize, channel_shuffle, dequantize])
        workspace.FeedBlob("X", X)
        workspace.RunNetOnce(net)
        Y = workspace.FetchBlob("Y")

        def channel_shuffle_ref(X):
            if order == "NHWC":
                # NHWC -> NCHW
                X = X.transpose((0, 3, 1, 2))
            Y_r = X.reshape(X.shape[0],
                            groups,
                            X.shape[1] // groups,
                            X.shape[2],
                            X.shape[3])
            Y_trns = Y_r.transpose((0, 2, 1, 3, 4))
            Y_reshaped = Y_trns.reshape(X.shape)
            if order == "NHWC":
                # NCHW -> NHWC
                Y_reshaped = Y_reshaped.transpose((0, 2, 3, 1))
            return Y_reshaped

        Y_ref = channel_shuffle_ref(X)
        np.testing.assert_allclose(Y, Y_ref)

    @given(
        channels_per_group=st.integers(min_value=32, max_value=128),
        n=st.integers(min_value=1, max_value=2),
        **hu.gcs_cpu_only)
    def test_channel_shuffle_fast_path(self, channels_per_group, n, gc, dc):
        order = "NHWC"
        groups = 4
        X = np.round(np.random.rand(
            n, channels_per_group * groups, 5, 6) * 255).astype(np.float32)
        X[0, 0, 0, 0] = 0
        X[0, 0, 0, 1] = 255
        # NCHW -> NHWC
        X = X.transpose((0, 2, 3, 1))

        net = core.Net("test_net")

        quantize = core.CreateOperator(
            "Quantize",
            ["X"],
            ["X_q"],
            engine="DNNLOWP",
        )

        channel_shuffle = core.CreateOperator(
            "ChannelShuffle", ["X_q"], ["Y_q"],
            group=groups,
            kernel=1,
            order=order,
            engine="DNNLOWP",
        )

        dequantize = core.CreateOperator(
            "Dequantize",
            ["Y_q"],
            ["Y"],
            engine="DNNLOWP",
        )

        net.Proto().op.extend([quantize, channel_shuffle, dequantize])
        workspace.FeedBlob("X", X)
        workspace.RunNetOnce(net)
        Y = workspace.FetchBlob("Y")

        def channel_shuffle_ref(X):
            if order == "NHWC":
                # NHWC -> NCHW
                X = X.transpose((0, 3, 1, 2))
            Y_r = X.reshape(X.shape[0],
                            groups,
                            X.shape[1] // groups,
                            X.shape[2],
                            X.shape[3])
            Y_trns = Y_r.transpose((0, 2, 1, 3, 4))
            Y_reshaped = Y_trns.reshape(X.shape)
            if order == "NHWC":
                # NCHW -> NHWC
                Y_reshaped = Y_reshaped.transpose((0, 2, 3, 1))
            return Y_reshaped

        Y_ref = channel_shuffle_ref(X)
        np.testing.assert_allclose(Y, Y_ref)
