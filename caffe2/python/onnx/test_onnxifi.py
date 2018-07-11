from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
import onnx.defs
from onnx.helper import make_node, make_graph, make_tensor, make_tensor_value_info, make_model
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from caffe2.python.onnx.tests.test_utils import TestCase

class OnnxifiTest(TestCase):
    @unittest.skipIf(not workspace.C.use_trt, "No TensortRT support")
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
