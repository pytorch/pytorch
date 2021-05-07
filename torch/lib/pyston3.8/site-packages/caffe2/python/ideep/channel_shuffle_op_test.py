




import unittest
import hypothesis.strategies as st
from hypothesis import given, settings
import numpy as np
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.ideep_test_util as mu


@unittest.skipIf(not workspace.C.use_mkldnn, "No MKLDNN support.")
class ChannelShuffleTest(hu.HypothesisTestCase):
    @given(size=st.integers(8, 10),
           input_channels=st.integers(1, 3),
           batch_size=st.integers(1, 32),
           group=st.integers(2, 4),
           stride=st.integers(1, 3),
           pad=st.integers(0, 3),
           kernel=st.integers(3, 5),
           **mu.gcs)
    @settings(max_examples=10, deadline=None)
    def test_channel_shuffle(self, size, input_channels, batch_size, group, stride, pad, kernel, gc, dc):
        op = core.CreateOperator(
            "ChannelShuffle",
            ["X"],
            ["Y"],
            group=group,
            stride=stride,
            pad=pad,
            kernel=kernel,
        )
        X = np.random.rand(
            batch_size, input_channels * group, size, size).astype(np.float32) - 0.5

        self.assertDeviceChecks(dc, op, [X], [0])

        self.assertGradientChecks(gc, op, [X], 0, [0])


if __name__ == "__main__":
    unittest.main()
