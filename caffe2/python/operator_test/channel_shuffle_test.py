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
        **hu.gcs)
    def test_channel_shuffle(self, channels_per_group, groups, n, gc, dc):
        X = np.random.randn(
            n, channels_per_group * groups, 5, 6).astype(np.float32)

        op = core.CreateOperator("ChannelShuffle", ["X"], ["Y"],
                                 group=groups, kernel=1)

        def channel_shuffle_ref(X):
            Y_r = X.reshape(X.shape[0],
                            groups,
                            X.shape[1] // groups,
                            X.shape[2],
                            X.shape[3])
            Y_trns = Y_r.transpose((0, 2, 1, 3, 4))
            return (Y_trns.reshape(X.shape),)

        self.assertReferenceChecks(gc, op, [X], channel_shuffle_ref)
        self.assertGradientChecks(gc, op, [X], 0, [0])
        self.assertDeviceChecks(dc, op, [X], [0])
