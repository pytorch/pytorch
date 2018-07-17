from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import unittest

import onnx
import onnx.defs
from onnx.helper import make_node, make_graph, make_tensor, make_tensor_value_info, make_model
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from caffe2.python.onnx.tests.test_utils import TestCase

class OnnxifiTest(TestCase):
    @unittest.skip("Need ONNXIFI backend support")
    def test_relu_graph(self):
        batch_size = 1
        X = np.random.randn(batch_size, 1, 3, 2).astype(np.float32)
        graph_def = make_graph(
            [make_node("Relu", ["X"], ["Y"])],
            name="test",
            inputs=[make_tensor_value_info("X", onnx.TensorProto.FLOAT,
                [batch_size, 1, 3, 2])],
            outputs=[make_tensor_value_info("Y", onnx.TensorProto.FLOAT,
                [batch_size, 1, 3, 2])])
        model_def = make_model(graph_def, producer_name='relu-test')
        op = core.CreateOperator(
            "Onnxifi",
            ["X"],
            ["Y"],
            onnx_model=model_def.SerializeToString(),
            output_size_hint_0=[batch_size, 1, 3, 2])
        workspace.FeedBlob("X", X)
        workspace.RunOperatorOnce(op)
        Y = workspace.FetchBlob("Y")
        np.testing.assert_almost_equal(Y, np.maximum(X, 0))

    @unittest.skip("Need ONNXIFI backend support")
    def test_conv_graph(self):
        X = np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 5, 5) input tensor
                        [5., 6., 7., 8., 9.],
                        [10., 11., 12., 13., 14.],
                        [15., 16., 17., 18., 19.],
                        [20., 21., 22., 23., 24.]]]]).astype(np.float32)
        W = np.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
                        [1., 1., 1.],
                        [1., 1., 1.]]]]).astype(np.float32)
        Y_without_padding = np.array([[[[54., 63., 72.],  # (1, 1, 3, 3) output tensor
                                        [99., 108., 117.],
                                        [144., 153., 162.]]]]).astype(np.float32)
        graph_def = make_graph(
            [make_node(
                'Conv',
                inputs=['X', 'W'],
                outputs=['Y'],
                kernel_shape=[3, 3],
                # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
                pads=[0, 0, 0, 0],
            )],
            name="test",
            inputs=[make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 1, 5, 5]),
                make_tensor_value_info("W", onnx.TensorProto.FLOAT, [1, 1, 3, 3]),
            ],
            outputs=[make_tensor_value_info("Y", onnx.TensorProto.FLOAT,
                [1, 1, 3, 3])])
        model_def = make_model(graph_def, producer_name='conv-test')
        op = core.CreateOperator(
            "Onnxifi",
            ["X", "W"],
            ["Y"],
            onnx_model=model_def.SerializeToString(),
            initializers=["W", "W"],
            output_size_hint_0=[1, 1, 3, 3])
        workspace.FeedBlob("X", X)
        workspace.FeedBlob("W", W)
        workspace.RunOperatorOnce(op)
        Y = workspace.FetchBlob("Y")
        np.testing.assert_almost_equal(Y, Y_without_padding)


