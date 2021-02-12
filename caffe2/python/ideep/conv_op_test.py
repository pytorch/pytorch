




import unittest
import hypothesis.strategies as st
from hypothesis import given, settings
import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from caffe2.python.transformations import optimizeForMKLDNN
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.ideep_test_util as mu


@unittest.skipIf(not workspace.C.use_mkldnn, "No MKLDNN support.")
class ConvTest(hu.HypothesisTestCase):
    @given(stride=st.integers(1, 3),
           pad=st.integers(0, 3),
           kernel=st.integers(3, 5),
           size=st.integers(8, 10),
           input_channels=st.integers(1, 3),
           output_channels=st.integers(1, 5),
           batch_size=st.integers(1, 3),
           use_bias=st.booleans(),
           training_mode=st.booleans(),
           group=st.integers(1, 2),
           **mu.gcs)
    @settings(max_examples=10, deadline=None)
    def test_convolution(self, stride, pad, kernel, size,
                         input_channels, output_channels,
                         batch_size, use_bias, training_mode, group, gc, dc):
        training = 1 if training_mode else 0
        op = core.CreateOperator(
            "Conv",
            ["X", "w", "b"] if use_bias else ["X", "w"],
            ["Y"],
            stride=stride,
            pad=pad,
            kernel=kernel,
            group=group,
            training_mode=training,
        )
        X = np.random.rand(
            batch_size, input_channels * group, size, size).astype(np.float32) - 0.5
        w = np.random.rand(output_channels * group, input_channels, kernel, kernel) \
            .astype(np.float32) - 0.5
        b = np.random.rand(output_channels * group).astype(np.float32) - 0.5

        inputs = [X, w, b] if use_bias else [X, w]
        self.assertDeviceChecks(dc, op, inputs, [0])

        if training_mode:
            for i in range(len(inputs)):
                self.assertGradientChecks(gc, op, inputs, i, [0], threshold=0.01)

    @settings(max_examples=10, deadline=None)
    @given(stride=st.integers(1, 3),
           pad=st.integers(0, 3),
           size=st.integers(8, 10),
           input_channels=st.integers(16, 32),
           output_channels=st.integers(16, 32),
           batch_size=st.integers(1, 3),
           use_bias=st.booleans(),
           training_mode=st.booleans(),
           **mu.gcs)
    def test_winograd_convolution(self, stride, pad, size,
                             input_channels, output_channels,
                             batch_size, use_bias, training_mode, gc, dc):
        training = 1 if training_mode else 0
        conv3x3_winograd_algorithm = 1
        kernel = 3
        op = core.CreateOperator(
            "Conv",
            ["X", "w", "b"] if use_bias else ["X", "w"],
            ["Y"],
            stride=stride,
            pad=pad,
            kernel=kernel,
            training_mode=training,
            algorithm=conv3x3_winograd_algorithm
        )
        X = np.random.rand(
            batch_size, input_channels, size, size).astype(np.float32) - 0.5
        w = np.random.rand(
                output_channels, input_channels, kernel, kernel) \
            .astype(np.float32) - 0.5
        b = np.random.rand(output_channels).astype(np.float32) - 0.5

        inputs = [X, w, b] if use_bias else [X, w]
        self.assertDeviceChecks(dc, op, inputs, [0])

        if training_mode:
            for i in range(len(inputs)):
                self.assertGradientChecks(gc, op, inputs, i, [0], threshold=0.01)

    @given(batch_size=st.integers(1, 3), **mu.gcs)
    def test_depthwise_convolution(self, batch_size, gc, dc):
        op = core.CreateOperator(
            "Conv",
            ["X", "w", "b"],
            ["Y"],
            stride=1,
            pad=0,
            kernel=1,
            group=4,
            device_option=dc[0]
        )
        op1 = core.CreateOperator(
            "Conv",
            ["X", "w", "b"],
            ["Y"],
            stride=1,
            pad=0,
            kernel=1,
            group=4,
            device_option=dc[1]
        )
        X = np.random.rand(batch_size, 544, 14, 14).astype(np.float32)
        w = np.random.rand(544, 136, 1, 1).astype(np.float32)
        b = np.random.rand(544).astype(np.float32)

        workspace.SwitchWorkspace("_device_check_", True)
        workspace.FeedBlob('X', X, dc[0])
        workspace.FeedBlob('w', w, dc[0])
        workspace.FeedBlob('b', b, dc[0])
        workspace.RunOperatorOnce(op)
        Y0 = workspace.FetchBlob('Y')

        workspace.ResetWorkspace()
        workspace.FeedBlob('X', X, dc[1])
        workspace.FeedBlob('w', w, dc[1])
        workspace.FeedBlob('b', b, dc[1])
        net = core.Net("net")
        old_net = caffe2_pb2.NetDef()
        old_net.op.extend([op1])
        net.Proto().CopyFrom(old_net)
        optimizeForMKLDNN(net)
        workspace.RunOperatorOnce(net.Proto().op[0])
        Y1 = workspace.FetchBlob('Y')

        if not np.allclose(Y0, Y1, atol=0.01, rtol=0.01):
            print(Y1.flatten())
            print(Y0.flatten())
            print(np.max(np.abs(Y1 - Y0)))
            self.assertTrue(False)

        workspace.ResetWorkspace()
        workspace.FeedBlob('X', X, dc[1])
        workspace.FeedBlob('w', w, dc[1])
        workspace.FeedBlob('b', b, dc[1])
        workspace.RunOperatorOnce(op1)
        Y2 = workspace.FetchBlob('Y')

        if not np.allclose(Y0, Y2, atol=0.01, rtol=0.01):
            print(Y2.flatten())
            print(Y0.flatten())
            print(np.max(np.abs(Y2 - Y0)))
            self.assertTrue(False)



if __name__ == "__main__":
    unittest.main()
