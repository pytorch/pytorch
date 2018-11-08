from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np

from caffe2.proto import caffe2_pb2
from caffe2.python import (
    core,
    dyndep,
    workspace,
    model_helper,
    brew,
    net_drawer,
)
from caffe2.python.optimizer import build_sgd
import pydot
from caffe2.python.transformations import Transformer

dyndep.InitOpsLibrary("//caffe2/caffe2/fb/opencl:opencl")
transformer = Transformer()

engine = 'FPGA'


OP_STYLE = {
    'shape': 'box',
    'color': '#0F9D58',
    'style': 'filled',
    'fontcolor': '#FFFFFF'
}

OP_STYLE_COPY = {
    'shape': 'box',
    'color': '#9d0f24',
    'style': 'filled',
    'fontcolor': '#FFFFFF'
}

OP_STYLE_COPY_TO = {
    'shape': 'box',
    'color': '#c6c72a',
    'style': 'filled',
    'fontcolor': '#FFFFFF'
}


def NodeProducer(op, op_id):
    if op.name:
        node_name = '%s/%s (op#%d)' % (op.name, op.type, op_id)
    else:
        node_name = '%s (op#%d)' % (op.type, op_id)
    if op.type.lower() in ['copyfromopencl']:
        return pydot.Node(node_name, **OP_STYLE_COPY)
    elif op.type.lower() in ['copytoopencl']:
        return pydot.Node(node_name, **OP_STYLE_COPY_TO)
    else:
        return pydot.Node(node_name, **OP_STYLE)


class Model(unittest.TestCase):
    def test_simple_model(self):
        workspace.ResetWorkspace()
        device = core.DeviceOption(caffe2_pb2.OPENCL, 0)
        device.extra_info.append(engine)

        model = model_helper.ModelHelper(name="train_net")
        bs = 64
        ref_w = [2.0, 1.0]
        ref_b = [0.5]

        X = model.GaussianFill([], "X", shape=[bs, 2], mean=0.0, std=1.0, run_once=0)
        Y_gt = brew.fc(model, X, "Y", 2, 1, name='CPU')
        model.param_init_net.GivenTensorFill([], "Y_w", shape=[1, 2], values=ref_w)
        model.param_init_net.GivenTensorFill([], "Y_b", shape=[1], values=ref_b)

        noise = model.GaussianFill([], "noise", shape=[bs, 1], mean=0.0,
          std=1e-6, run_once=0)
        Y_noise = model.Add([noise, Y_gt], "Y_noise")
        Y_noise = model.StopGradient(Y_noise, "Y_noise")

        Y_pred = brew.fc(model, X, "Y_pred", 2, 1)
        Y_pred_relu = brew.relu(model, Y_pred, "Y_pred_relu")

        dist = model.SquaredL2Distance([Y_noise, Y_pred_relu], "dist")
        loss = model.AveragedLoss([dist], ["loss"])
        model.AddGradientOperators([loss])
        build_sgd(model, 0.1, policy="fixed")

        workspace.RunNetOnce(model.param_init_net)

        # print(model.net.Proto())

        op_types = ['FC', 'FCGradient', 'Relu', 'ReluGradient']
        for op in model.net.Proto().op:
            if op.type in op_types and op.name != 'CPU':
                op.engine = engine
                op.device_option.CopyFrom(device)

        transformer.ConvertToOpenCL(model.net)

        for op in model.net.Proto().op:
            if op.type == "CopyFromOpenCL" or op.type == "CopyToOpenCL":
                op.engine = engine
                op.device_option.CopyFrom(device)

        workspace.CreateNet(model.net)
        # print(model.net.Proto())

        graph = net_drawer.GetPydotGraph(model.net.Proto().op, "train",
             rankdir="LR", node_producer=NodeProducer)
        with open('image.png', 'wb') as f:
            f.write(graph.create_png())
        print(workspace.FetchBlob("Y_w"))
        print(workspace.FetchBlob("Y_pred_w"))

        for _ in range(30):
            workspace.RunNet(model.net, 10)

            _Y_pred_w = workspace.FetchBlob("Y_pred_w")
            _Y_pred_b = workspace.FetchBlob("Y_pred_b")
            loss = workspace.FetchBlob("loss")
            print("w", _Y_pred_w)
            print("b", _Y_pred_b)

            print("loss", loss)

        np.allclose(ref_w, _Y_pred_w, rtol=0.1)
        np.allclose(ref_b, _Y_pred_b, rtol=0.1)
        # assert(loss < 1e-4)


        print("done")
