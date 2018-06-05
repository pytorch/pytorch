from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import caffe2.python.hypothesis_test_util as hu
from caffe2.python import core
from hypothesis import given
import hypothesis.strategies as st


class ChannelShuffleOpsTest(hu.HypothesisTestCase):
    @given(
        channels_per_group=st.integers(min_value=1, max_value=5),
        groups=st.integers(min_value=1, max_value=5),
        n=st.integers(min_value=1, max_value=2),
        order=st.sampled_from(["NCHW", "NHWC"]),
        **hu.gcs)
    def test_channel_shuffle(self, channels_per_group, groups, n, order, gc, dc):
        X = np.random.randn(
            n, channels_per_group * groups, 5, 6).astype(np.float32)
        if order == "NHWC":
            # NCHW -> NHWC
            X = X.transpose((0, 2, 3, 1))

        op = core.CreateOperator("ChannelShuffle", ["X"], ["Y"],
                                 group=groups, kernel=1, order=order,
                                 device_option=gc)

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
            return (Y_reshaped,)

        self.assertReferenceChecks(gc, op, [X], channel_shuffle_ref)
        self.assertGradientChecks(gc, op, [X], 0, [0])
        self.assertDeviceChecks(dc, op, [X], [0])
