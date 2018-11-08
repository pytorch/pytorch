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

dyndep.InitOpsLibrary("//caffe2/caffe2/fb/opencl:opencl")

engine = 'FPGA'


class MyMLPTest(unittest.TestCase):
    def test_multiply(self):
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
            core.CreateOperator('CopyToOpenCL', ['B'], ['d_B'],
                device_option=device, engine=engine)
        ])
        net.external_input.append('B')

        net.op.extend([
            core.CreateOperator('FC', ['d_X', 'd_W', 'd_B'],
               ['d_Y'], 'fc_1', device_option=device, engine=engine)
        ])

        net.op.extend([
            core.CreateOperator('CopyFromOpenCL', ['d_Y'], ['Y'],
                device_option=device, engine=engine)
        ])
        net.external_output.append('Y')

        # Print the protobuf if needed
        # print(net)
        np.random.seed(0)

        def fc_op_bfp16(X, W, B):
            f = np.vectorize(lambda x: bfloat_conversion(x, 16))
            W_bfp16 = f(W)
            X_bfp16 = f(X)
            B_bfp16 = f(B)

            Y = np.matmul(X_bfp16, W_bfp16.transpose())
            bias = np.ones((X.shape[0], 1)) * B_bfp16
            Y += bias
            Y_bfp = f(Y)
            return Y_bfp

        for i in range(2, 1024, 50):
            n = 2 * i  # 1500
            k = 3 * i
            m = 5 * i
            print("testing", n, k, m)

            W = np.random.rand(n, k)
            X = np.random.rand(m, k)
            B = np.random.rand(n)

            workspace.FeedBlob("X", X.astype(np.float32))
            workspace.FeedBlob("W", W.astype(np.float32))
            workspace.FeedBlob("B", B.astype(np.float32))

            workspace.RunNetOnce(net)
            Y_pred = workspace.FetchBlob('Y')
            y_ref = fc_op_bfp16(X.astype(np.float32), W.astype(np.float32),
                B.astype(np.float32))

            rtol = 1e-2 * n  # TODO: fix tolerance
            issame = np.allclose(Y_pred, y_ref, atol=0.0, rtol=rtol)
            if not issame:
                print("from device", Y_pred.shape, Y_pred)
                print("reference", y_ref.shape, y_ref)
                print("W", W)
                print("X", X)
                print("B", B)
                diff = Y_pred - y_ref
                print("diff", diff)
                rows, cols = np.nonzero(Y_pred - y_ref)
                reldiff = (Y_pred - y_ref) / y_ref
                print(np.transpose([rows, cols]))
                print(diff[rows, cols])
                print(reldiff[rows, cols])

            assert(issame)

    # CPU test for comparison
    def test_cpu(self):
        workspace.ResetWorkspace()

        net = caffe2_pb2.NetDef()
        net.external_input.append('X')
        net.external_input.append('W')
        net.external_input.append('B')

        net.op.extend([
            core.CreateOperator('FC', ['X', 'W', 'B'],
               ['Y'], 'fc')
        ])

        net.external_output.append('Y')

        def fc_op(X, W):
            Y = np.matmul(X, W.transpose())
            return Y

        # n = 1024
        # k = 1024
        # m = 1024

        W = np.random.rand(1, 2)
        X = np.random.rand(4, 2)
        B = np.random.rand(1) * 0

        # W = np.random.rand(n, k)
        # X = np.identity(m)
        # B = np.zeros(m)

        workspace.FeedBlob("X", X.astype(np.float32))
        workspace.FeedBlob("W", W.astype(np.float32))
        workspace.FeedBlob("B", B.astype(np.float32))

        workspace.RunNetOnce(net)
        Y_pred = workspace.FetchBlob('Y')
        Y = fc_op(X, W)

        assert(np.allclose(Y_pred, Y))


if __name__ == "__main__":
    unittest.main()
