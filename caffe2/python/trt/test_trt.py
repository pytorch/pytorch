from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
import onnx
import onnx.defs
from onnx.helper import make_node, make_graph, make_tensor, make_tensor_value_info, make_model
from onnx.backend.base import namedtupledict
from caffe2.python.models.download import downloadFromURLToFile, getURLFromName, deleteDirectory
import caffe2.python.onnx.backend as c2
from caffe2.python.onnx.workspace import Workspace
from caffe2.python.trt.transform import convert_onnx_model_to_trt_op, transform_caffe2_net
from caffe2.python.onnx.tests.test_utils import TestCase
import numpy as np
import os.path
import json
import time
import unittest
import tarfile
import tempfile
import shutil
from six.moves.urllib.request import urlretrieve

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


_BASE_URL = 'https://s3.amazonaws.com/download.onnx/models/opset_{}'.format(onnx.defs.onnx_opset_version())

# TODO: This is copied from https://github.com/onnx/onnx/blob/master/onnx/backend/test/runner/__init__.py. Maybe we should
# expose a model retrival API from ONNX
def _download_onnx_model(model_name):
    onnx_home = os.path.expanduser(os.getenv('ONNX_HOME', os.path.join('~', '.onnx')))
    models_dir = os.getenv('ONNX_MODELS',
                           os.path.join(onnx_home, 'models'))
    model_dir = os.path.join(models_dir, model_name)
    if not os.path.exists(os.path.join(model_dir, 'model.onnx')):
        if os.path.exists(model_dir):
            bi = 0
            while True:
                dest = '{}.old.{}'.format(model_dir, bi)
                if os.path.exists(dest):
                    bi += 1
                    continue
                shutil.move(model_dir, dest)
                break
        os.makedirs(model_dir)

        # On Windows, NamedTemporaryFile can not be opened for a
        # second time
        url = '{}/{}.tar.gz'.format(_BASE_URL, model_name)
        download_file = tempfile.NamedTemporaryFile(delete=False)
        try:
            download_file.close()
            print('Start downloading model {} from {}'.format(
                model_name, url))
            urlretrieve(url, download_file.name)
            print('Done')
            with tarfile.open(download_file.name) as t:
                t.extractall(models_dir)
        except Exception as e:
            print('Failed to prepare data for model {}: {}'.format(
                model_name, e))
            raise
        finally:
            os.remove(download_file.name)
    return model_dir

class TensorRTOpTest(TestCase):
    def _test_relu_graph(self, X, batch_size, trt_max_batch_size):
        node_def = make_node("Relu", ["X"], ["Y"])
        Y_c2 = c2.run_node(node_def, {"X": X})
        graph_def = make_graph(
            [node_def],
            name="test",
            inputs=[make_tensor_value_info("X", onnx.TensorProto.FLOAT, [batch_size, 1, 3, 2])],
            outputs=[make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [batch_size, 1, 3, 2])])
        model_def = make_model(graph_def, producer_name='relu-test')
        op_outputs = [x.name for x in model_def.graph.output]
        op = convert_onnx_model_to_trt_op(model_def, max_batch_size=trt_max_batch_size)
        device_option = core.DeviceOption(caffe2_pb2.CUDA, 0)
        op.device_option.CopyFrom(device_option)
        Y_trt = None
        ws = Workspace()
        with core.DeviceScope(device_option):
            ws.FeedBlob("X", X)
            ws.RunOperatorsOnce([op])
            output_values = [ws.FetchBlob(name) for name in op_outputs]
            Y_trt = namedtupledict('Outputs', op_outputs)(*output_values)
        np.testing.assert_almost_equal(Y_c2, Y_trt)


    @unittest.skipIf('TEST_C2_TRT' not in os.environ, "No TensortRT support")
    def test_relu_graph_simple(self):
        X = np.random.randn(1, 1, 3, 2).astype(np.float32)
        self._test_relu_graph(X, 1, 50)


    @unittest.skipIf('TEST_C2_TRT' not in os.environ, "No TensortRT support")
    def test_relu_graph_big_batch(self):
        X = np.random.randn(52, 1, 3, 2).astype(np.float32)
        self._test_relu_graph(X, 52, 50)


    @unittest.skipIf('TEST_C2_TRT' not in os.environ, "No TensortRT support")
    def test_resnet50(self):
        input_blob_dims = (1, 3, 224, 224)
        model_dir = _download_onnx_model('resnet50')
        model_def = onnx.load(os.path.join(model_dir, 'model.onnx'))
        op_inputs = [x.name for x in model_def.graph.input]
        op_outputs = [x.name for x in model_def.graph.output]
        n, c, h, w = input_blob_dims
        data = np.random.randn(n, c, h, w).astype(np.float32)
        Y_c2 = c2.run_model(model_def, {op_inputs[0]: data})
        op = convert_onnx_model_to_trt_op(model_def)
        device_option = core.DeviceOption(caffe2_pb2.CUDA, 0)
        op.device_option.CopyFrom(device_option)
        Y_trt = None
        ws = Workspace()
        with core.DeviceScope(device_option):
            ws.FeedBlob(op_inputs[0], data)
            ws.RunOperatorsOnce([op])
            output_values = [ws.FetchBlob(name) for name in op_outputs]
            Y_trt = namedtupledict('Outputs', op_outputs)(*output_values)
        np.testing.assert_allclose(Y_c2, Y_trt, rtol=1e-3)


class TensorRTTransformTest(TestCase):
    def _model_dir(self, model):
        caffe2_home = os.path.expanduser(os.getenv('ONNX_HOME', '~/.caffe2'))
        models_dir = os.getenv('ONNX_MODELS', os.path.join(caffe2_home, 'models'))
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


    @unittest.skipIf('TEST_C2_TRT' not in os.environ, "No TensortRT support")
    def test_resnet50_core(self):
        N = 2
        warmup = 20
        repeat = 100
        print("Batch size: {}, repeat inference {} times, warmup {} times".format(N, repeat, warmup))
        init_net, pred_net, _  = self._get_c2_model('resnet50')
        self._add_head_tail(pred_net, 'real_data', 'real_softmax')
        input_blob_dims = (N, 3, 224, 224)
        input_name = "real_data"

        device_option = core.DeviceOption(caffe2_pb2.CUDA, 0)
        init_net.device_option.CopyFrom(device_option)
        pred_net.device_option.CopyFrom(device_option)
        for op in pred_net.op:
            op.device_option.CopyFrom(device_option)
            op.engine = 'CUDNN'
        net_outputs = pred_net.external_output
        Y_c2 = None
        data =  np.random.randn(*input_blob_dims).astype(np.float32)
        c2_time = 1
        ws = Workspace()
        with core.DeviceScope(device_option):
            ws.FeedBlob(input_name, data)
            ws.RunNetOnce(init_net)
            ws.CreateNet(pred_net)
            for _ in range(warmup):
                ws.RunNet(pred_net.name)
            start = time.time()
            for _ in range(repeat):
                ws.RunNet(pred_net.name)
            end = time.time()
            c2_time = end - start
            output_values = [ws.FetchBlob(name) for name in net_outputs]
            Y_c2 = namedtupledict('Outputs', net_outputs)(*output_values)
        ws.ResetWorkspace()

        # Cut the graph
        init_net_cut, pred_net_cut = transform_caffe2_net(init_net, pred_net, {input_name: input_blob_dims})
        del init_net, pred_net
        #print_net(pred_net_cut)

        Y_trt = None
        input_name = pred_net_cut.external_input[0]
        print("C2 runtime: {}s".format(c2_time))
        ws = Workspace()
        with core.DeviceScope(device_option):
            ws.FeedBlob(input_name, data)
            ws.RunNetOnce(init_net_cut)
            ws.CreateNet(pred_net_cut)
            for _ in range(warmup):
                ws.RunNet(pred_net_cut.name)
            start = time.time()
            for _ in range(repeat):
                ws.RunNet(pred_net_cut.name)
            end = time.time()
            trt_time = end - start
            print("TRT runtime: {}s, improvement: {}%".format(trt_time, (c2_time-trt_time)/c2_time*100))
            output_values = [ws.FetchBlob(name) for name in net_outputs]
            Y_trt = namedtupledict('Outputs', net_outputs)(*output_values)
        np.testing.assert_allclose(Y_c2, Y_trt, rtol=1e-3)


