from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, dyndep, workspace
from hypothesis import given
import caffe2.python.ideep_test_util as mu

dyndep.InitOpsLibrary("@/caffe2/modules/detectron:detectron_ops")


class TestUpsampleNearestOp(hu.HypothesisTestCase):
    @given(
        N=st.integers(1, 3),
        H=st.integers(10, 300),
        W=st.integers(10, 300),
        scale=st.integers(1, 3),
        **hu.gcs
    )
    def test_upsample_nearest_op_cpu(self, N, H, W, scale, gc, dc):
        C = 32
        X = np.random.randn(N, C, H, W).astype(np.float32)
        op = core.CreateOperator("UpsampleNearest", ["X"], ["Y"], scale=scale)

        def ref(X):
            outH = H * scale
            outW = W * scale
            outH_idxs, outW_idxs = np.meshgrid(
                np.arange(outH), np.arange(outW), indexing="ij"
            )
            inH_idxs = (outH_idxs / scale).astype(np.int32)
            inW_idxs = (outW_idxs / scale).astype(np.int32)
            Y = X[:, :, inH_idxs, inW_idxs]
            return [Y]

        self.assertReferenceChecks(device_option=gc, op=op, inputs=[X], reference=ref)
    @unittest.skipIf(not workspace.C.use_mkldnn, "No MKLDNN support.")
    @given(stride=st.integers(1, 3),
           pad=st.integers(0, 3),
           kernel=st.integers(3, 5),
           size=st.integers(8, 10),
           input_channels=st.integers(3, 32),
           output_channels=st.integers(3, 32),
           batch_size=st.integers(1, 3),
           use_bias=st.booleans(),
           scale=st.integers(1, 3),
           **mu.gcs
    )
    def test_upsample_nearest_op_ideep(self, stride, pad, kernel, size,
                                           input_channels, output_channels,
                                           batch_size, use_bias,
                                           scale, gc, dc):
        X = np.random.rand(
            batch_size, input_channels, size, size).astype(np.float32) - 0.5
        w = np.random.rand(
                output_channels, input_channels, kernel, kernel) \
            .astype(np.float32) - 0.5
        b = np.random.rand(output_channels).astype(np.float32) - 0.5
         
        conv_op = core.CreateOperator(
                 "Conv",
                 ["X", "w", "b"],
                 ["Y0"],
                 stride=stride,
                 pad=pad,
                 kernel=kernel,
                 device_option=dc[0]
        )
        upsample_op = core.CreateOperator("UpsampleNearest",
                ["Y0"],
                ["Y1"],
                scale=scale,
                device_option=dc[0]
        )
        conv_op1 = core.CreateOperator(
                 "Conv",
                 ["X", "w", "b"],
                 ["Y2"],
                 stride=stride,
                 pad=pad,
                 kernel=kernel,
                 device_option=dc[1]
        )
        upsample_op1 = core.CreateOperator("UpsampleNearest",
                ["Y2"],
                ["Y3"],
                scale=scale,
                device_option=dc[1]
        )

        workspace.SwitchWorkspace("_device_check_", True)
        workspace.FeedBlob('X', X, dc[0])
        workspace.FeedBlob('w', w, dc[0])
        workspace.FeedBlob('b', b, dc[0])
        workspace.RunOperatorOnce(conv_op)
        workspace.RunOperatorOnce(upsample_op)
        Y1 = workspace.FetchBlob('Y1')
   
        workspace.ResetWorkspace()
        workspace.FeedBlob('X', X, dc[1])
        workspace.FeedBlob('w', w, dc[1])
        workspace.FeedBlob('b', b, dc[1])
        workspace.RunOperatorOnce(conv_op1)
        workspace.RunOperatorOnce(upsample_op1)
        Y3 = workspace.FetchBlob('Y3')
   
        if not np.allclose(Y1, Y3, atol=0.01, rtol=0.01):
            print(Y3.flatten())
            print(Y1.flatten())
            print(np.max(np.abs(Y3 - Y1)))
            self.assertTrue(False)

if __name__ == "__main__":
    unittest.main()
