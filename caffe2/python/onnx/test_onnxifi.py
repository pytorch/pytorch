from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import numpy as np
import os
import time
import unittest

import onnx
import onnx.defs
from onnx.backend.base import namedtupledict
from onnx.helper import make_node, make_graph, make_tensor, make_tensor_value_info, make_model
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from caffe2.python.models.download import downloadFromURLToFile, getURLFromName, deleteDirectory
from caffe2.python.onnx.onnxifi import onnxifi_caffe2_net
from caffe2.python.onnx.tests.test_utils import TestCase

ONNXIFI_DATATYPE_FLOAT32 = 1


def _print_net(net):
    for i in net.external_input:
        print("Input: {}".format(i))
    for i in net.external_output:
        print("Output: {}".format(i))
    for op in net.op:
        print("Op {}".format(op.type))
        for x in op.input:
            print("  input: {}".format(x))
        for y in op.output:
            print("  output: {}".format(y))


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
            output_shape_hint_0=[ONNXIFI_DATATYPE_FLOAT32, batch_size, 1, 3, 2])
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
            ["X"],
            ["Y"],
            onnx_model=model_def.SerializeToString(),
            initializers=["W", "W"],
            output_shape_hint_0=[ONNXIFI_DATATYPE_FLOAT32, 1, 1, 3, 3])
        workspace.FeedBlob("X", X)
        workspace.FeedBlob("W", W)
        workspace.RunOperatorOnce(op)
        Y = workspace.FetchBlob("Y")
        np.testing.assert_almost_equal(Y, Y_without_padding)


class OnnxifiTransformTest(TestCase):
    def _model_dir(self, model):
        caffe2_home = os.path.expanduser(os.getenv('CAFFE2_HOME', '~/.caffe2'))
        models_dir = os.getenv('CAFFE2_MODELS', os.path.join(caffe2_home, 'models'))
        return os.path.join(models_dir, model)

    def _download(self, model):
        model_dir = self._model_dir(model)
        assert not os.path.exists(model_dir)
        os.makedirs(model_dir)
        for f in ['predict_net.pb', 'init_net.pb', 'value_info.json']:
            url = getURLFromName(model, f)
            dest = os.path.join(model_dir, f)
            try:
                try:
                    downloadFromURLToFile(url, dest,
                                          show_progress=False)
                except TypeError:
                    # show_progress not supported prior to
                    # Caffe2 78c014e752a374d905ecfb465d44fa16e02a28f1
                    # (Sep 17, 2017)
                    downloadFromURLToFile(url, dest)
            except Exception as e:
                print("Abort: {reason}".format(reason=e))
                print("Cleaning up...")
                deleteDirectory(model_dir)
                exit(1)

    # TODO: we need to modulize this function
    def _get_c2_model(self, model_name):
        model_dir = self._model_dir(model_name)
        if not os.path.exists(model_dir):
            self._download(model_name)
        c2_predict_pb = os.path.join(model_dir, 'predict_net.pb')
        c2_predict_net = caffe2_pb2.NetDef()
        with open(c2_predict_pb, 'rb') as f:
            c2_predict_net.ParseFromString(f.read())
        c2_predict_net.name = model_name

        c2_init_pb = os.path.join(model_dir, 'init_net.pb')
        c2_init_net = caffe2_pb2.NetDef()
        with open(c2_init_pb, 'rb') as f:
            c2_init_net.ParseFromString(f.read())
        c2_init_net.name = model_name + '_init'

        value_info = json.load(open(os.path.join(model_dir, 'value_info.json')))
        return c2_init_net, c2_predict_net, value_info

    def _add_head_tail(self, pred_net, new_head, new_tail):
        orig_head = pred_net.external_input[0]
        orig_tail = pred_net.external_output[0]

        # Add head
        head = caffe2_pb2.OperatorDef()
        head.type = "Copy"
        head.input.append(new_head)
        head.output.append(orig_head)
        dummy = caffe2_pb2.NetDef()
        dummy.op.extend(pred_net.op)
        del pred_net.op[:]
        pred_net.op.extend([head])
        pred_net.op.extend(dummy.op)
        pred_net.external_input[0] = new_head

        # Add tail
        tail = caffe2_pb2.OperatorDef()
        tail.type = "Copy"
        tail.input.append(orig_tail)
        tail.output.append(new_tail)
        pred_net.op.extend([tail])
        pred_net.external_output[0] = new_tail

    @unittest.skip("Need ONNXIFI backend support")
    def test_resnet50_core(self):
        N = 1
        repeat = 1
        print("Batch size: {}, repeat inference {} times".format(N, repeat))
        init_net, pred_net, _ = self._get_c2_model('resnet50')
        self._add_head_tail(pred_net, 'real_data', 'real_softmax')
        input_blob_dims = (N, 3, 224, 224)
        input_name = "real_data"

        device_option = core.DeviceOption(caffe2_pb2.CPU, 0)
        init_net.device_option.CopyFrom(device_option)
        pred_net.device_option.CopyFrom(device_option)
        for op in pred_net.op:
            op.device_option.CopyFrom(device_option)
        net_outputs = pred_net.external_output
        Y_c2 = None
        data = np.random.randn(*input_blob_dims).astype(np.float32)
        c2_time = 1
        workspace.SwitchWorkspace("onnxifi_test", True)
        with core.DeviceScope(device_option):
            workspace.FeedBlob(input_name, data)
            workspace.RunNetOnce(init_net)
            workspace.CreateNet(pred_net)
            start = time.time()
            for _ in range(repeat):
                workspace.RunNet(pred_net.name)
            end = time.time()
            c2_time = end - start
            output_values = [workspace.FetchBlob(name) for name in net_outputs]
            Y_c2 = namedtupledict('Outputs', net_outputs)(*output_values)
        workspace.ResetWorkspace()

        # Fill the workspace with the weights
        with core.DeviceScope(device_option):
            workspace.RunNetOnce(init_net)

        # Cut the graph
        start = time.time()
        pred_net_cut = onnxifi_caffe2_net(pred_net,
                                          {input_name: input_blob_dims},
                                          infer_shapes=True)
        del init_net, pred_net
        #_print_net(pred_net_cut)

        Y_trt = None
        input_name = pred_net_cut.external_input[0]
        print("C2 runtime: {}s".format(c2_time))
        with core.DeviceScope(device_option):
            workspace.FeedBlob(input_name, data)
            workspace.CreateNet(pred_net_cut)
            end = time.time()
            print("Conversion time: {:.2f}s".format(end - start))

            start = time.time()
            for _ in range(repeat):
                workspace.RunNet(pred_net_cut.name)
            end = time.time()
            trt_time = end - start
            print("Onnxifi runtime: {}s, improvement: {}%".format(trt_time, (c2_time - trt_time) / c2_time * 100))
            output_values = [workspace.FetchBlob(name) for name in net_outputs]
            Y_trt = namedtupledict('Outputs', net_outputs)(*output_values)
        np.testing.assert_allclose(Y_c2, Y_trt, rtol=1e-3)


