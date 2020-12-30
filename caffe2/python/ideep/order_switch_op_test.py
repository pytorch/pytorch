




import unittest
import numpy as np
import hypothesis.strategies as st
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.ideep_test_util as mu

from hypothesis import given, settings
from caffe2.python import core, workspace


@unittest.skipIf(not workspace.C.use_mkldnn, "No MKLDNN support.")
class OrderSwitchTest(hu.HypothesisTestCase):
    @given(n=st.integers(1, 128),
           c=st.integers(1, 64),
           h=st.integers(1, 128),
           w=st.integers(1, 128),
           **mu.gcs)
    @settings(max_examples=10, deadline=None)
    def test_nchw2nhwc(self, n, c, h, w, gc, dc):
        op = core.CreateOperator(
            "NCHW2NHWC",
            ["X"],
            ["Y"],
        )
        X = np.random.rand(n, c, h, w).astype(np.float32) - 0.5

        self.assertDeviceChecks(dc, op, [X], [0])

    @given(n=st.integers(1, 128),
           c=st.integers(1, 64),
           h=st.integers(1, 128),
           w=st.integers(1, 128),
           **mu.gcs)
    @settings(deadline=None, max_examples=50)
    def test_nhwc2nchw(self, n, c, h, w, gc, dc):
        op0 = core.CreateOperator(
            "NCHW2NHWC",
            ["X"],
            ["Y"],
        )
        op1 = core.CreateOperator(
            "NHWC2NCHW",
            ["Y"],
            ["Z"],
        )

        X = np.random.rand(n, c, h, w).astype(np.float32) - 0.5

        old_ws_name = workspace.CurrentWorkspace()
        workspace.SwitchWorkspace("_device_check_", True)
        workspace.FeedBlob('X', X, dc[0])
        op0.device_option.CopyFrom(dc[0])
        op1.device_option.CopyFrom(dc[0])
        workspace.RunOperatorOnce(op0)
        workspace.RunOperatorOnce(op1)
        Z0 = workspace.FetchBlob("Z")

        workspace.ResetWorkspace()
        workspace.FeedBlob('X', X, dc[1])
        op0.device_option.CopyFrom(dc[1])
        op1.device_option.CopyFrom(dc[1])
        workspace.RunOperatorOnce(op0)
        workspace.RunOperatorOnce(op1)
        Z1 = workspace.FetchBlob("Z")

        if not np.allclose(Z0, Z1, atol=0.01, rtol=0.01):
            print(Z1.flatten())
            print(Z0.flatten())
            print(np.max(np.abs(Z1 - Z0)))
            self.assertTrue(False)

        workspace.SwitchWorkspace(old_ws_name)


if __name__ == "__main__":
    unittest.main()
