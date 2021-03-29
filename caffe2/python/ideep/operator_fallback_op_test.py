




import unittest
import hypothesis.strategies as st
from hypothesis import given
import numpy as np
from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.ideep_test_util as mu


@unittest.skipIf(not workspace.C.use_mkldnn, "No MKLDNN support.")
class TestFallbackOps(hu.HypothesisTestCase):
    @given(stride=st.integers(1, 3),
           pad=st.integers(0, 3),
           kernel=st.integers(3, 5),
           size=st.integers(8, 10),
           input_channels=st.integers(1, 3),
           output_channels=st.integers(1, 5),
           batch_size=st.integers(1, 3),
           use_bias=st.booleans(),
           **mu.gcs)
    def test_in_place(self, stride, pad, kernel, size,
                             input_channels, output_channels,
                             batch_size, use_bias, gc, dc):
        # To expose fallback in-place potential issue, the fallback op
        # following ideep op must be run at least two iterations.
        conv = core.CreateOperator(
            "Conv",
            ["X", "w", "b"] if use_bias else ["X", "w"],
            ["Y"],
            stride=stride,
            pad=pad,
            kernel=kernel,
            device_option=dc[0]
        )
        X = np.random.rand(
            batch_size, input_channels, size, size).astype(np.float32) - 0.5
        w = np.random.rand(output_channels, input_channels, kernel, kernel) \
            .astype(np.float32) - 0.5
        b = np.random.rand(output_channels).astype(np.float32) - 0.5

        old_ws_name = workspace.CurrentWorkspace()
        workspace.SwitchWorkspace("_device_check_", True)
        workspace.FeedBlob('X', X, dc[0])
        workspace.FeedBlob('w', w, dc[0])
        workspace.FeedBlob('b', b, dc[0])
        workspace.RunOperatorOnce(conv)
        Y = workspace.FetchBlob('Y')

        scale = np.random.randn(Y.shape[1]).astype(np.float32)
        bias = np.random.randn(Y.shape[1]).astype(np.float32)
        ac = core.CreateOperator(
            "AffineChannel",
            ["Y", "scale", "bias"],
            ["Y"],
            is_learnable=False,
            device_option=dc[0]
        )
        workspace.FeedBlob('scale', scale, dc[0])
        workspace.FeedBlob('bias', bias, dc[0])
        workspace.RunOperatorOnce(ac)
        workspace.RunOperatorOnce(conv)
        workspace.RunOperatorOnce(ac)
        Y0 = workspace.FetchBlob('Y')

        workspace.ResetWorkspace()
        dev_net = caffe2_pb2.NetDef()
        conv_dev = caffe2_pb2.OperatorDef()
        conv_dev.CopyFrom(conv)
        conv_dev.device_option.CopyFrom(dc[1])
        ac_dev = caffe2_pb2.OperatorDef()
        ac_dev.CopyFrom(ac)
        ac_dev.device_option.CopyFrom(dc[1])
        dev_net.op.extend([conv_dev, ac_dev])
        workspace.FeedBlob('X', X, dc[1])
        workspace.FeedBlob('w', w, dc[1])
        workspace.FeedBlob('b', b, dc[1])
        workspace.FeedBlob('scale', scale, dc[1])
        workspace.FeedBlob('bias', bias, dc[1])
        workspace.RunNetOnce(dev_net)
        workspace.RunNetOnce(dev_net)
        Y1 = workspace.FetchBlob('Y')

        if not np.allclose(Y0, Y1, atol=0.01, rtol=0.01):
            print(Y1.flatten())
            print(Y0.flatten())
            print(np.max(np.abs(Y1 - Y0)))
            self.assertTrue(False)

        workspace.SwitchWorkspace(old_ws_name)


if __name__ == "__main__":
    unittest.main()
