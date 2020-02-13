from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import sys
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
        w = np.random.rand(
                output_channels * group, input_channels, kernel, kernel) \
            .astype(np.float32) - 0.5
        b = np.random.rand(output_channels * group).astype(np.float32) - 0.5

        inputs = [X, w, b] if use_bias else [X, w]
        self.assertDeviceChecks(dc, op, inputs, [0])

        if training_mode:
            for i in range(len(inputs)):
                self.assertGradientChecks(gc, op, inputs, i, [0], threshold=0.01)
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

    @unittest.skipIf(sys.version_info.major > 2, "broken in python 3")
    @given(stride=st.integers(1, 3),
           pad=st.integers(0, 3),
           kernel=st.integers(3, 5),
           size=st.integers(8, 10),
           input_channels=st.integers(1, 3),
           output_channels=st.integers(1, 5),
           batch_size=st.integers(1, 3),
           use_bias=st.booleans(),
           **mu.gcs)
    def test_int8_convolution(self, stride, pad, kernel, size,
                             input_channels, output_channels,
                             batch_size, use_bias, gc, dc):
        X = np.random.rand(
            batch_size, input_channels, size, size).astype(np.float32) - 0.5
        w = np.random.rand(
                output_channels, input_channels, kernel, kernel) .astype(np.float32) - 0.5
        b = np.random.rand(output_channels).astype(np.float32) - 0.5

        old_ws_name = workspace.CurrentWorkspace()
        workspace.SwitchWorkspace("_device_check_", True)
        conv_fp32 = core.CreateOperator(
            "Conv",
            ["X_fp32", "w_fp32", "b_fp32"] if use_bias else ["X_fp32", "w_fp32"],
            ["Y_fp32"],
            stride=stride,
            pad=pad,
            kernel=kernel,
            training_mode=0,
            device_option=dc[0],
        )
        workspace.FeedBlob('X_fp32', X, dc[0])
        workspace.FeedBlob('w_fp32', w, dc[0])
        workspace.FeedBlob('b_fp32', b, dc[0])
        workspace.RunOperatorOnce(conv_fp32)
        Y = workspace.FetchBlob('Y_fp32')

        workspace.ResetWorkspace()

        Y_absmax = np.array([np.absolute(Y).max()]).astype(np.float32)
        if Y.min() >= 0:
            Y_scale = Y_absmax / 0xFF
            Y_zero_point = 0
        else:
            Y_scale = Y_absmax / 0x7F
            Y_zero_point = 128

        X_absmax = np.array([np.absolute(X).max()]).astype(np.float32)
        if X.min() >= 0:
            X_scale = X_absmax / 0xFF
            X_zero_point = 0
        else:
            X_scale = X_absmax / 0x7F
            X_zero_point = 128

        w_absmax = np.array([np.absolute(w[i, ...]).max() for i in range(w.shape[0])]).astype(np.float32)
        w_scale = w_absmax / 0x7F
        w_zero_point = 128
        w = np.transpose(w, (0, 2, 3, 1)).astype(np.float32)
        w_bytes = np.rint([w[i, ...] / w_scale[i] for i in range(w.shape[0])]).astype(np.int8) + w_zero_point

        w_filler = core.CreateOperator(
            "Int8GivenTensorFill",
            [], ["w"],
            shape=w.shape,
            values=w_bytes.astype(np.uint8).tobytes(),
            Y_zero_point=w_zero_point,
            Y_scales=w_scale,
            device_option=dc[1],
        )

        b_scale = w_scale * X_scale
        b_zero_point = 0
        b_bytes = np.rint([b[i] / b_scale[i] for i in range(b.shape[0])]).astype(np.int32)
        b_filler = core.CreateOperator(
            "Int8GivenIntTensorFill",
            [], ["b"],
            shape=b.shape,
            values=b_bytes,
            Y_zero_point=b_zero_point,
            Y_scales=b_scale,
            device_option=dc[1],
        )

        sw2nhwc = core.CreateOperator(
            "NCHW2NHWC",
            ["X"],
            ["X_nhwc"],
            device_option=dc[1]
        )

        quantize_X = core.CreateOperator(
            "Int8Quantize",
            ["X_nhwc"],
            ["X_quantized"],
            engine="DNNLOWP",
            device_option=dc[1],
            Y_zero_point=X_zero_point,
            Y_scale=X_scale[0],
        )

        conv = core.CreateOperator(
            "Int8Conv",
            ["X_quantized", "w", "b"] if use_bias else ["X_quantized", "w"],
            ["Y_quantized"],
            stride=stride,
            pad=pad,
            kernel=kernel,
            engine="DNNLOWP",
            device_option=dc[1],
            Y_zero_point=Y_zero_point,
            Y_scale=Y_scale[0],
        )

        dequantize_Y = core.CreateOperator(
            "Int8Dequantize",
            ["Y_quantized"],
            ["Y_nhwc"],
            engine="DNNLOWP",
            device_option=dc[1],
        )

        sw2nchw = core.CreateOperator(
            "NHWC2NCHW",
            ["Y_nhwc"],
            ["Y_out"],
            device_option=dc[1]
        )

        net = caffe2_pb2.NetDef()
        net.op.extend([w_filler, b_filler, sw2nhwc, quantize_X, conv, dequantize_Y, sw2nchw])

        workspace.FeedBlob("X", X, dc[1])
        workspace.RunNetOnce(net)
        Y_out = workspace.FetchBlob("Y_out")

        MSE = np.square(np.subtract(Y, Y_out)).mean()
        if MSE > 0.005:
            print(Y.flatten())
            print(Y_out.flatten())
            print(np.max(np.abs(Y_out - Y)))
            print("MSE", MSE)
            self.assertTrue(False)

        workspace.SwitchWorkspace(old_ws_name)



if __name__ == "__main__":
    unittest.main()
