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
from caffe2.python.optimizer import build_sgd

dyndep.InitOpsLibrary("//caffe2/caffe2/fb/opencl:opencl")

engine = 'FPGA'


class Gradient(unittest.TestCase):
    def test_grad(self):
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
            core.CreateOperator('CopyToOpenCL', ['W'], ['d_W'],
                device_option=device, engine=engine)
        ])
        net.external_input.append('W')

        net.op.extend([
            core.CreateOperator('CopyToOpenCL', ['dY'], ['d_dY'],
                device_option=device, engine=engine)
        ])
        net.external_input.append('dY')

        net.op.extend([
            core.CreateOperator('FCGradient', ['d_X', 'd_W', 'd_dY'],
               ['d_dW', 'd_db', 'd_dX'], 'fcgrad1', device_option=device, engine=engine)
        ])

        net.op.extend([
            core.CreateOperator('CopyFromOpenCL', ['d_dW'], ['dW'],
                device_option=device, engine=engine)
        ])
        net.external_output.append('dW')

        net.op.extend([
            core.CreateOperator('CopyFromOpenCL', ['d_db'], ['db'],
                device_option=device, engine=engine)
        ])
        net.external_output.append('db')

        net.op.extend([
            core.CreateOperator('CopyFromOpenCL', ['d_dX'], ['dX'],
                device_option=device, engine=engine)
        ])
        net.external_output.append('dX')
        # print(net)

        np.random.seed(0)

        n = 1024
        print("running ", n)
        n = n
        k = n
        m = n

        W = np.random.rand(n, k)
        X = np.identity(m)
        dY = np.random.rand(n, k)

        workspace.FeedBlob("W", W.astype(np.float32))
        workspace.FeedBlob("X", X.astype(np.float32))
        workspace.FeedBlob("dY", dY.astype(np.float32))

        def fcgrad_op_bfp16(W, X, dY):
            f = np.vectorize(lambda x: bfloat_conversion(x, 16))
            W = f(W)
            X = f(X)
            dY = f(dY)

            dW = np.matmul(dY.transpose(), X)
            db = np.matmul(dY.transpose(), np.ones(X.shape[0]))
            dX = np.matmul(dY, W)

            dW = f(dW)
            db = f(db)
            dX = f(dX)
            return dW, db, dX

        workspace.RunNetOnce(net)
        dW_pred = workspace.FetchBlob('dW')
        db_pred = workspace.FetchBlob('db')
        dX_pred = workspace.FetchBlob('dX')

        dW_ref, db_ref, dX_ref = fcgrad_op_bfp16(W, X, dY)

        isclose = np.allclose(dW_pred, dW_ref, rtol=8e-3) and \
            np.allclose(db_pred, db_ref, rtol=8e-3) and \
            np.allclose(dX_pred, dX_ref, rtol=8e-3)

        if not isclose:
            print("W", W)
            print("X", X)
            print("dY", dY)
            print("dWr", dW_ref)
            print("dbr", db_ref)
            print("dXr", dX_ref)
            print("dW", dW_pred)
            print("db", db_pred)
            print("dX", dX_pred)

        assert(isclose)


if __name__ == "__main__":
    unittest.main()
