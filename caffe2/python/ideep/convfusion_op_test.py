from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import hypothesis.strategies as st
from hypothesis import given, settings
import copy
import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from caffe2.python.transformations import optimizeForIDEEP
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.ideep_test_util as mu


@unittest.skipIf(not workspace.C.use_mkldnn, "No MKLDNN support.")
class ConvFusionTest(hu.HypothesisTestCase):
    @given(stride=st.integers(1, 3),
           pad=st.integers(0, 3),
           kernel=st.integers(3, 5),
           size=st.integers(8, 20),
           input_channels=st.integers(1, 16),
           output_channels=st.integers(1, 16),
           batch_size=st.integers(1, 3),
           use_bias=st.booleans(),
           group=st.integers(1, 1),
           **mu.gcs)
    def test_convolution_relu_fusion(self, stride, pad, kernel, size,
                             input_channels, output_channels,
                             batch_size, use_bias, group, gc, dc):
        conv = core.CreateOperator(
            "Conv",
            ["X0", "w0", "b0"] if use_bias else ["X0", "w0"],
            ["Y0"],
            stride=stride,
            pad=pad,
            kernel=kernel,
            group=group,
            device_option=dc[0]
        )
        relu = core.CreateOperator(
            "Relu",
            ["Y0"],
            ["Y0"],
            device_option=dc[0]
        )

        # Manual fusion for Conv + ReLU
        conv_fusion = core.CreateOperator(
            "ConvFusion",
            ["X1", "w1", "b1"] if use_bias else ["X1", "w1"],
            ["Y1"],
            stride=stride,
            pad=pad,
            kernel=kernel,
            group=group,
            fusion_type = 1,
            device_option=dc[1]
        )

        X = np.random.rand(
            batch_size, input_channels * group, size, size).astype(np.float32) - 0.5
        w = np.random.rand(
                output_channels * group, input_channels, kernel, kernel) \
            .astype(np.float32) - 0.5
        b = np.random.rand(output_channels * group).astype(np.float32) - 0.5

        old_ws_name = workspace.CurrentWorkspace()
        workspace.SwitchWorkspace("_device_check_", True)
        workspace.FeedBlob('X0', X, dc[0])
        workspace.FeedBlob('w0', w, dc[0])
        workspace.FeedBlob('b0', b, dc[0])
        workspace.RunOperatorOnce(conv)
        workspace.RunOperatorOnce(relu)
        Y0 = workspace.FetchBlob('Y0')

        workspace.ResetWorkspace()
        workspace.FeedBlob('X1', X, dc[1])
        workspace.FeedBlob('w1', w, dc[1])
        workspace.FeedBlob('b1', b, dc[1])
        workspace.RunOperatorOnce(conv_fusion)
        Y1 = workspace.FetchBlob('Y1')
        if not np.allclose(Y0, Y1, atol=0.01, rtol=0.01):
            print(Y1.flatten())
            print(Y0.flatten())
            print(np.max(np.abs(Y1 - Y0)))
            self.assertTrue(False)

        # Auto fusion for Conv + ReLU
        workspace.ResetWorkspace()
        old_net = caffe2_pb2.NetDef()
        conv_old = caffe2_pb2.OperatorDef()
        conv_old.CopyFrom(conv)
        conv_old.device_option.CopyFrom(dc[1])
        relu_old = caffe2_pb2.OperatorDef()
        relu_old.CopyFrom(relu)
        relu_old.device_option.CopyFrom(dc[1])
        old_net.op.extend([conv_old, relu_old])
        workspace.FeedBlob('X0', X, dc[1])
        workspace.FeedBlob('w0', w, dc[1])
        workspace.FeedBlob('b0', b, dc[1])
        net = core.Net("net")
        net.Proto().CopyFrom(old_net)
        optimizeForIDEEP(net)
        self.assertTrue(len(net.Proto().op) == 1)
        self.assertTrue(net.Proto().op[0].type == "ConvFusion")
        workspace.RunOperatorOnce(net.Proto().op[0])
        Y2 = workspace.FetchBlob('Y0')
        if not np.allclose(Y0, Y2, atol=0.01, rtol=0.01):
            print(Y2.flatten())
            print(Y0.flatten())
            print(np.max(np.abs(Y2 - Y0)))
            self.assertTrue(False)

        workspace.SwitchWorkspace(old_ws_name)

    @given(stride=st.integers(1, 3),
           pad=st.integers(0, 3),
           kernel=st.integers(3, 5),
           size=st.integers(8, 20),
           input_channels=st.integers(1, 16),
           output_channels=st.integers(1, 16),
           batch_size=st.integers(1, 3),
           use_bias=st.booleans(),
           group=st.integers(1, 1),
           **mu.gcs)
    def test_convolution_sum_fusion(self, stride, pad, kernel, size,
                             input_channels, output_channels,
                             batch_size, use_bias, group, gc, dc):
        conv_S0 = core.CreateOperator(
            "Conv",
            ["SX0", "Sw0", "Sb0"] if use_bias else ["SX0", "Sw0"],
            ["S0"],
            stride=stride,
            pad=pad,
            kernel=kernel,
            group=group,
            device_option=dc[0]
        )
        conv = core.CreateOperator(
            "Conv",
            ["X0", "w0", "b0"] if use_bias else ["X0", "w0"],
            ["Y0"],
            stride=stride,
            pad=pad,
            kernel=kernel,
            group=group,
            device_option=dc[0]
        )
        sum = core.CreateOperator(
            "Sum",
            ["S0", "Y0"],
            ["S0"],
            device_option=dc[0]
        )

        # Manual fusion for Conv + Sum
        conv_S1 = core.CreateOperator(
            "Conv",
            ["SX1", "Sw1", "Sb1"] if use_bias else ["SX1", "Sw1"],
            ["S1"],
            stride=stride,
            pad=pad,
            kernel=kernel,
            group=group,
            device_option=dc[1]
        )
        conv_fusion = core.CreateOperator(
            "ConvFusion",
            ["X1", "w1", "b1", "S1"] if use_bias else ["X1", "w1", "S1"],
            ["S1"],
            stride=stride,
            pad=pad,
            kernel=kernel,
            group=group,
            fusion_type = 2,
            device_option=dc[1]
        )
        SX = np.random.rand(
            batch_size, input_channels * group, size, size).astype(np.float32) - 0.5
        Sw = np.random.rand(
                output_channels * group, input_channels, kernel, kernel) \
            .astype(np.float32) - 0.5
        Sb = np.random.rand(output_channels * group).astype(np.float32) - 0.5
        X = np.random.rand(
            batch_size, input_channels * group, size, size).astype(np.float32) - 0.5
        w = np.random.rand(
                output_channels * group, input_channels, kernel, kernel) \
            .astype(np.float32) - 0.5
        b = np.random.rand(output_channels * group).astype(np.float32) - 0.5

        old_ws_name = workspace.CurrentWorkspace()
        workspace.SwitchWorkspace("_device_check_", True)
        workspace.FeedBlob('SX0', SX, dc[0])
        workspace.FeedBlob('Sw0', Sw, dc[0])
        workspace.FeedBlob('Sb0', Sb, dc[0])
        workspace.FeedBlob('X0', X, dc[0])
        workspace.FeedBlob('w0', w, dc[0])
        workspace.FeedBlob('b0', b, dc[0])
        workspace.RunOperatorOnce(conv_S0)
        workspace.RunOperatorOnce(conv)
        workspace.RunOperatorOnce(sum)
        S0 = workspace.FetchBlob('S0')

        workspace.ResetWorkspace()
        workspace.FeedBlob('SX1', SX, dc[1])
        workspace.FeedBlob('Sw1', Sw, dc[1])
        workspace.FeedBlob('Sb1', Sb, dc[1])
        workspace.FeedBlob('X1', X, dc[1])
        workspace.FeedBlob('w1', w, dc[1])
        workspace.FeedBlob('b1', b, dc[1])
        workspace.RunOperatorOnce(conv_S1)
        workspace.RunOperatorOnce(conv_fusion)
        S1 = workspace.FetchBlob('S1')

        if not np.allclose(S0, S1, atol=0.01, rtol=0.01):
            print(S1.flatten())
            print(S0.flatten())
            print(np.max(np.abs(S1 - S0)))
            self.assertTrue(False)

        # Auto fusion for Conv + Sum
        workspace.ResetWorkspace()
        old_net = caffe2_pb2.NetDef()
        conv_S0_old = caffe2_pb2.OperatorDef()
        conv_S0_old.CopyFrom(conv_S0)
        conv_S0_old.device_option.CopyFrom(dc[1])
        conv_old = caffe2_pb2.OperatorDef()
        conv_old.CopyFrom(conv)
        conv_old.device_option.CopyFrom(dc[1])
        sum_old = caffe2_pb2.OperatorDef()
        sum_old.CopyFrom(sum)
        sum_old.device_option.CopyFrom(dc[1])
        old_net.op.extend([conv_S0_old, conv_old, sum_old])
        workspace.FeedBlob('SX0', SX, dc[1])
        workspace.FeedBlob('Sw0', Sw, dc[1])
        workspace.FeedBlob('Sb0', Sb, dc[1])
        workspace.FeedBlob('X0', X, dc[1])
        workspace.FeedBlob('w0', w, dc[1])
        workspace.FeedBlob('b0', b, dc[1])
        net = core.Net("net")
        net.Proto().CopyFrom(old_net)
        optimizeForIDEEP(net)
        self.assertTrue(len(net.Proto().op) == 2)
        self.assertTrue(net.Proto().op[1].type == "ConvFusion")
        workspace.RunNetOnce(net.Proto())
        S2 = workspace.FetchBlob('S0')
        if not np.allclose(S0, S2, atol=0.01, rtol=0.01):
            print(S2.flatten())
            print(S0.flatten())
            print(np.max(np.abs(S2 - S0)))
            self.assertTrue(False)

        workspace.SwitchWorkspace(old_ws_name)

    @given(stride=st.integers(1, 3),
           pad=st.integers(0, 3),
           kernel=st.integers(3, 5),
           size=st.integers(8, 20),
           input_channels=st.integers(1, 16),
           output_channels=st.integers(1, 16),
           batch_size=st.integers(1, 3),
           use_bias=st.booleans(),
           group=st.integers(1, 1),
           **mu.gcs)
    def test_convolution_sum_relu_fusion(self, stride, pad, kernel, size,
                             input_channels, output_channels,
                             batch_size, use_bias, group, gc, dc):
        conv_S0 = core.CreateOperator(
            "Conv",
            ["SX0", "Sw0", "Sb0"] if use_bias else ["SX0", "Sw0"],
            ["S0"],
            stride=stride,
            pad=pad,
            kernel=kernel,
            group=group,
            device_option=dc[0]
        )
        conv = core.CreateOperator(
            "Conv",
            ["X0", "w0", "b0"] if use_bias else ["X0", "w0"],
            ["Y0"],
            stride=stride,
            pad=pad,
            kernel=kernel,
            group=group,
            device_option=dc[0]
        )
        sum = core.CreateOperator(
            "Sum",
            ["S0", "Y0"],
            ["S0"],
            device_option=dc[0]
        )
        relu = core.CreateOperator(
            "Relu",
            ["S0"],
            ["S0"],
            device_option=dc[0]
        )

        # Manual fusion for Conv + Sum + ReLU
        conv_S1 = core.CreateOperator(
            "Conv",
            ["SX1", "Sw1", "Sb1"] if use_bias else ["SX1", "Sw1"],
            ["S1"],
            stride=stride,
            pad=pad,
            kernel=kernel,
            group=group,
            device_option=dc[1]
        )
        conv_fusion = core.CreateOperator(
            "ConvFusion",
            ["X1", "w1", "b1", "S1"] if use_bias else ["X1", "w1", "S1"],
            ["S1"],
            stride=stride,
            pad=pad,
            kernel=kernel,
            group=group,
            fusion_type = 3,
            device_option=dc[1]
        )
        SX = np.random.rand(
            batch_size, input_channels * group, size, size).astype(np.float32) - 0.5
        Sw = np.random.rand(
                output_channels * group, input_channels, kernel, kernel) \
            .astype(np.float32) - 0.5
        Sb = np.random.rand(output_channels * group).astype(np.float32) - 0.5
        X = np.random.rand(
            batch_size, input_channels * group, size, size).astype(np.float32) - 0.5
        w = np.random.rand(
                output_channels * group, input_channels, kernel, kernel) \
            .astype(np.float32) - 0.5
        b = np.random.rand(output_channels * group).astype(np.float32) - 0.5

        old_ws_name = workspace.CurrentWorkspace()
        workspace.SwitchWorkspace("_device_check_", True)
        workspace.FeedBlob('SX0', SX, dc[0])
        workspace.FeedBlob('Sw0', Sw, dc[0])
        workspace.FeedBlob('Sb0', Sb, dc[0])
        workspace.FeedBlob('X0', X, dc[0])
        workspace.FeedBlob('w0', w, dc[0])
        workspace.FeedBlob('b0', b, dc[0])
        workspace.RunOperatorOnce(conv_S0)
        workspace.RunOperatorOnce(conv)
        workspace.RunOperatorOnce(sum)
        workspace.RunOperatorOnce(relu)
        S0 = workspace.FetchBlob('S0')

        workspace.ResetWorkspace()
        workspace.FeedBlob('SX1', SX, dc[1])
        workspace.FeedBlob('Sw1', Sw, dc[1])
        workspace.FeedBlob('Sb1', Sb, dc[1])
        workspace.FeedBlob('X1', X, dc[1])
        workspace.FeedBlob('w1', w, dc[1])
        workspace.FeedBlob('b1', b, dc[1])
        workspace.RunOperatorOnce(conv_S1)
        workspace.RunOperatorOnce(conv_fusion)
        S1 = workspace.FetchBlob('S1')

        if not np.allclose(S0, S1, atol=0.01, rtol=0.01):
            print(S1.flatten())
            print(S0.flatten())
            print(np.max(np.abs(S1 - S0)))
            self.assertTrue(False)

        # Auto fusion for Conv + Sum + ReLU
        workspace.ResetWorkspace()
        old_net = caffe2_pb2.NetDef()
        conv_S0_old = caffe2_pb2.OperatorDef()
        conv_S0_old.CopyFrom(conv_S0)
        conv_S0_old.device_option.CopyFrom(dc[1])
        conv_old = caffe2_pb2.OperatorDef()
        conv_old.CopyFrom(conv)
        conv_old.device_option.CopyFrom(dc[1])
        sum_old = caffe2_pb2.OperatorDef()
        sum_old.CopyFrom(sum)
        sum_old.device_option.CopyFrom(dc[1])
        relu_old = caffe2_pb2.OperatorDef()
        relu_old.CopyFrom(relu)
        relu_old.device_option.CopyFrom(dc[1])
        old_net.op.extend([conv_S0_old, conv_old, sum_old, relu_old])
        workspace.FeedBlob('SX0', SX, dc[1])
        workspace.FeedBlob('Sw0', Sw, dc[1])
        workspace.FeedBlob('Sb0', Sb, dc[1])
        workspace.FeedBlob('X0', X, dc[1])
        workspace.FeedBlob('w0', w, dc[1])
        workspace.FeedBlob('b0', b, dc[1])
        net = core.Net("net")
        net.Proto().CopyFrom(old_net)
        optimizeForIDEEP(net)
        self.assertTrue(len(net.Proto().op) == 2)
        self.assertTrue(net.Proto().op[1].type == "ConvFusion")
        workspace.RunNetOnce(net.Proto())
        S2 = workspace.FetchBlob('S0')
        if not np.allclose(S0, S2, atol=0.01, rtol=0.01):
            print(S2.flatten())
            print(S0.flatten())
            print(np.max(np.abs(S2 - S0)))
            self.assertTrue(False)

        workspace.SwitchWorkspace(old_ws_name)

    @given(stride=st.integers(1, 3),
           pad=st.integers(0, 3),
           kernel=st.integers(3, 5),
           size=st.integers(8, 20),
           input_channels=st.integers(1, 16),
           output_channels=st.integers(1, 16),
           batch_size=st.integers(1, 3),
           use_bias=st.booleans(),
           group=st.integers(1, 1),
           inplace=st.sampled_from([True, False]),
           **mu.gcs)
    def test_convolution_bn_folding(
            self, stride, pad, kernel, size, input_channels,
            output_channels, batch_size, use_bias, group,
            inplace, gc, dc):
        conv = core.CreateOperator(
            "Conv",
            ["X0", "w0", "b0"] if use_bias else ["X0", "w0"],
            ["X1"],
            stride=stride,
            pad=pad,
            kernel=kernel,
            group=group,
            device_option=dc[1]
        )
        bn = core.CreateOperator(
            "SpatialBN",
            ["X1", "scale", "bias", "mean", "var"],
            ["X1" if inplace else "Y"],
            is_test=True,
            device_option=dc[1]
        )

        X = np.random.rand(
            batch_size, input_channels * group, size, size).astype(np.float32) - 0.5
        w = np.random.rand(
                output_channels * group, input_channels, kernel, kernel) \
            .astype(np.float32) - 0.5
        b = np.random.rand(output_channels * group).astype(np.float32) - 0.5
        scale = np.random.rand(output_channels).astype(np.float32) + 0.5
        bias = np.random.rand(output_channels).astype(np.float32) - 0.5
        mean = np.random.randn(output_channels).astype(np.float32)
        var = np.absolute(np.random.rand(output_channels).astype(np.float32)) + 0.5

        old_ws_name = workspace.CurrentWorkspace()
        workspace.SwitchWorkspace("_device_check_", True)
        workspace.FeedBlob('X0', X, dc[1])
        workspace.FeedBlob('w0', w, dc[1])
        workspace.FeedBlob('b0', b, dc[1])
        workspace.FeedBlob('scale', scale, dc[1])
        workspace.FeedBlob('bias', bias, dc[1])
        workspace.FeedBlob('mean', mean, dc[1])
        workspace.FeedBlob('var', var, dc[1])
        workspace.RunOperatorOnce(conv)
        workspace.RunOperatorOnce(bn)
        Y = workspace.FetchBlob('X1' if inplace else "Y")

        workspace.ResetWorkspace()
        old_net = caffe2_pb2.NetDef()
        conv_old = caffe2_pb2.OperatorDef()
        conv_old.CopyFrom(conv)
        conv_old.device_option.CopyFrom(dc[1])
        bn_old = caffe2_pb2.OperatorDef()
        bn_old.CopyFrom(bn)
        bn_old.device_option.CopyFrom(dc[1])
        old_net.op.extend([conv_old, bn_old])
        workspace.FeedBlob('X0', X, dc[1])
        workspace.FeedBlob('w0', w, dc[1])
        workspace.FeedBlob('b0', b, dc[1])
        workspace.FeedBlob('scale', scale, dc[1])
        workspace.FeedBlob('bias', bias, dc[1])
        workspace.FeedBlob('mean', mean, dc[1])
        workspace.FeedBlob('var', var, dc[1])
        net = core.Net("net")
        net.Proto().CopyFrom(old_net)
        optimizeForIDEEP(net)
        self.assertTrue(len(net.Proto().op) == 1)
        self.assertTrue(net.Proto().op[0].type == "Conv")
        workspace.RunOperatorOnce(net.Proto().op[0])
        Y1 = workspace.FetchBlob('X1' if inplace else "Y")
        if not np.allclose(Y, Y1, atol=0.01, rtol=0.01):
            print(Y.flatten())
            print(Y1.flatten())
            print(np.max(np.abs(Y - Y1)))
            self.assertTrue(False)

        workspace.SwitchWorkspace(old_ws_name)

    @given(stride=st.integers(1, 3),
           pad=st.integers(0, 3),
           kernel=st.integers(3, 5),
           size=st.integers(8, 20),
           input_channels=st.integers(1, 16),
           output_channels=st.integers(1, 16),
           batch_size=st.integers(1, 3),
           use_bias=st.booleans(),
           group=st.integers(1, 1),
           inplace=st.sampled_from([True, False]),
           **mu.gcs)
    def test_convolution_affch_folding(
            self, stride, pad, kernel, size, input_channels,
            output_channels, batch_size, use_bias, group,
            inplace, gc, dc):
        conv = core.CreateOperator(
            "Conv",
            ["X0", "w0", "b0"] if use_bias else ["X0", "w0"],
            ["X1"],
            stride=stride,
            pad=pad,
            kernel=kernel,
            group=group,
            device_option=dc[1]
        )
        affch = core.CreateOperator(
            "AffineChannel",
            ["X1", "scale", "bias"],
            ["X1" if inplace else "Y"],
            device_option=dc[1]
        )

        X = np.random.rand(
            batch_size, input_channels * group, size, size).astype(np.float32) - 0.5
        w = np.random.rand(
                output_channels * group, input_channels, kernel, kernel) \
            .astype(np.float32) - 0.5
        b = np.random.rand(output_channels * group).astype(np.float32) - 0.5
        scale = np.random.rand(output_channels).astype(np.float32) + 0.5
        bias = np.random.rand(output_channels).astype(np.float32) - 0.5

        old_ws_name = workspace.CurrentWorkspace()
        workspace.SwitchWorkspace("_device_check_", True)
        workspace.FeedBlob('X0', X, dc[1])
        workspace.FeedBlob('w0', w, dc[1])
        workspace.FeedBlob('b0', b, dc[1])
        workspace.FeedBlob('scale', scale, dc[1])
        workspace.FeedBlob('bias', bias, dc[1])
        workspace.RunOperatorOnce(conv)
        workspace.RunOperatorOnce(affch)
        Y = workspace.FetchBlob('X1' if inplace else "Y")

        workspace.ResetWorkspace()
        old_net = caffe2_pb2.NetDef()
        conv_old = caffe2_pb2.OperatorDef()
        conv_old.CopyFrom(conv)
        conv_old.device_option.CopyFrom(dc[1])
        affch_old = caffe2_pb2.OperatorDef()
        affch_old.CopyFrom(affch)
        affch_old.device_option.CopyFrom(dc[1])
        old_net.op.extend([conv_old, affch_old])
        workspace.FeedBlob('X0', X, dc[1])
        workspace.FeedBlob('w0', w, dc[1])
        workspace.FeedBlob('b0', b, dc[1])
        workspace.FeedBlob('scale', scale, dc[1])
        workspace.FeedBlob('bias', bias, dc[1])
        net = core.Net("net")
        net.Proto().CopyFrom(old_net)
        optimizeForIDEEP(net)
        self.assertTrue(len(net.Proto().op) == 1)
        self.assertTrue(net.Proto().op[0].type == "Conv")
        workspace.RunOperatorOnce(net.Proto().op[0])
        Y1 = workspace.FetchBlob('X1' if inplace else "Y")
        if not np.allclose(Y, Y1, atol=0.01, rtol=0.01):
            print(Y.flatten())
            print(Y1.flatten())
            print(np.max(np.abs(Y - Y1)))
            self.assertTrue(False)

        workspace.SwitchWorkspace(old_ws_name)

if __name__ == "__main__":
    unittest.main()
