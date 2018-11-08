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
import pydot
import time
from caffe2.python.optimizer import build_sgd
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


class MyMLPTest(unittest.TestCase):
    def test_multiply(self):
        workspace.ResetWorkspace()
        device = core.DeviceOption(caffe2_pb2.OPENCL, 0)
        device.extra_info.append(engine)

        model = model_helper.ModelHelper(name="train_net")

        batch_size = 4096
        xdim = 2048
        yout = 512
        ref_w = np.random.rand(yout, xdim)
        ref_b = np.random.rand(yout)

        X = model.GaussianFill([], "X", shape=[batch_size, xdim], mean=0.0,
          std=1.0, run_once=0)
        Y_gt = brew.fc(model, X, "Y", xdim, yout, name='CPU')
        Y_gt = model.StopGradient(Y_gt, "Y_gt")

        model.param_init_net.GivenTensorFill([], "Y_w", shape=[yout, xdim],
          values=ref_w)
        model.param_init_net.GivenTensorFill([], "Y_b", shape=[yout], values=ref_b)

        Y_pred = brew.fc(model, X, "Y_pred", xdim, yout)
        Y_pred_relu = brew.relu(model, Y_pred, "Y_pred_relu")

        dist = model.SquaredL2Distance([Y_gt, Y_pred_relu], "dist")
        loss = model.AveragedLoss([dist], ["loss"])
        model.AddGradientOperators([loss])
        build_sgd(model, 0.001, policy="fixed")

        workspace.RunNetOnce(model.param_init_net)

        # print(model.net.Proto())
        op_types = ['FC', 'FCGradient', 'Relu', 'ReluGradient']
        for op in model.net.Proto().op:
            if op.type in op_types:
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

        N = 100
        N_int = 10
        start = time.time()
        for _ in range(N):
            workspace.RunNet(model.net, N_int)
            workspace.FetchBlob("Y_pred_w")
            workspace.FetchBlob("Y_pred_b")
            loss = workspace.FetchBlob("loss")
            print("loss", loss)
            print("dt", time.time() - start)


def main():
    unittest.main()


if __name__ == "__main__":
    main()
