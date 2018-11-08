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
)
from caffe2.python.fb.opencl.test.utils import bfloat_conversion
from hypothesis import given
from hypothesis.strategies import integers

dyndep.InitOpsLibrary("//caffe2/caffe2/fb/opencl:opencl")

engine = 'FPGA'


class ReLU(unittest.TestCase):
    @given(integers(min_value=1, max_value=2048), integers(min_value=1, max_value=2048))
    def test_relu(self, n, k):
        workspace.ResetWorkspace()
        device = core.DeviceOption(caffe2_pb2.OPENCL, 0)
        device.extra_info.append(engine)

        net = caffe2_pb2.NetDef()
        net.op.extend([
            core.CreateOperator('CopyToOpenCL', ['X'], ['d_X'],
                device_option=device, engine=engine)
        ])
        net.external_input.append('X')

        net.op.extend([
            core.CreateOperator('Relu', ['d_X'],
               ['d_Y'], 'relu1', device_option=device, engine=engine)
        ])

        net.op.extend([
            core.CreateOperator('CopyFromOpenCL', ['d_Y'], ['Y'],
                device_option=device, engine=engine)
        ])
        net.external_output.append('Y')

        # Print the protobuf if needed
        # print(net)

        def relu_bfp16(X):
            f = np.vectorize(lambda x: bfloat_conversion(x, 16))
            X_bfp16 = f(X)
            out = (np.sign(X_bfp16) + 1) / 2 * X_bfp16
            #print(out)
            return out

        print("running ", n, k)

        X = np.random.rand(n, k) - 0.5

        workspace.FeedBlob("X", X.astype(np.float32))

        workspace.RunNetOnce(net)
        Y_pred = workspace.FetchBlob('Y')
        y_ref = relu_bfp16(X.astype(np.float32))

        issame = np.allclose(Y_pred, y_ref)
        if not issame:
            print("Input", X)
            print(Y_pred.shape, Y_pred)
            print(y_ref.shape, y_ref)
            diff = Y_pred - y_ref
            print("diff", diff)
            rows, cols = np.nonzero(Y_pred - y_ref)
            print(np.transpose([rows, cols]))
            print(diff[rows, cols])

        assert(issame)

    @given(integers(min_value=1, max_value=2048), integers(min_value=1, max_value=2048))
    def test_relugrad(self, n, k):
        workspace.ResetWorkspace()
        device = core.DeviceOption(caffe2_pb2.OPENCL, 0)
        device.extra_info.append(engine)

        net = caffe2_pb2.NetDef()
        net.op.extend([
            core.CreateOperator('CopyToOpenCL', ['X'], ['d_X'],
                device_option=device, engine=engine)
        ])
        net.external_input.append('X')
        net.op.extend([
            core.CreateOperator('CopyToOpenCL', ['dX'], ['d_dX'],
                device_option=device, engine=engine)
        ])
        net.external_input.append('dX')

        net.op.extend([
            core.CreateOperator('ReluGradient', ['d_X', 'd_dX'],
               ['d_Y'], 'relugrad1', device_option=device, engine=engine)
        ])

        net.op.extend([
            core.CreateOperator('CopyFromOpenCL', ['d_Y'], ['Y'],
                device_option=device, engine=engine)
        ])
        net.external_output.append('Y')

        # Print the protobuf if needed
        # print(net)

        def relugrad_bfp16(X, dX):
            f = np.vectorize(lambda x: bfloat_conversion(x, 16))
            dX_bfp16 = f(dX)
            out = (np.sign(X - 1e-5) + 1) / 2 * dX_bfp16
            return out

        print("running ", n, k)

        X = np.random.rand(n, k) - 0.5
        dX = np.random.rand(n, k) - 0.5

        workspace.FeedBlob("X", X.astype(np.float32))
        workspace.FeedBlob("dX", dX.astype(np.float32))

        workspace.RunNetOnce(net)
        Y_pred = workspace.FetchBlob('Y')
        y_ref = relugrad_bfp16(X.astype(np.float32), dX.astype(np.float32))

        issame = np.allclose(Y_pred, y_ref)

        if not issame:
            print("Input X", X)
            print("Input dX", dX)
            print(Y_pred.shape, Y_pred)
            print("np reference", y_ref.shape, y_ref)
            diff = Y_pred - y_ref
            print("diff", diff)
            rows, cols = np.nonzero(Y_pred - y_ref)
            print(np.transpose([rows, cols]))
            print(diff[rows, cols])

        assert(issame)
