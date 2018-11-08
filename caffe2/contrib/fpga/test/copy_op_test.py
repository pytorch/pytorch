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


class MyMLPTest(unittest.TestCase):
    @given(integers(min_value=1, max_value=2048), integers(min_value=1, max_value=2048))
    def test_copy_back_and_forth(self, n, k):
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
            core.CreateOperator('CopyFromOpenCL', ['d_X'], ['X_hat'],
                device_option=device, engine=engine)
        ])
        net.external_output.append('X_hat')

        # Could be useful for debugging, so keeping here, helpful
        # for debugging matrix transpositions and when the tile sizes are
        # very big
        # X = np.ones((n, k)) * 0.31415
        # X[0, 1] = 1.23
        # X[:, 6:X.shape[1]] = 0
        # X[4:X.shape[0], :] = 0

        X = np.random.rand(n, k)

        workspace.FeedBlob("X", X.astype(np.float32))
        workspace.RunNetOnce(net)
        X_hat = workspace.FetchBlob('X_hat')
        assert np.allclose(X, X_hat, rtol=7.9e-3), (X, X_hat)

        f = np.vectorize(lambda x: bfloat_conversion(x, 16))
        X = f(X)
        workspace.FeedBlob("X", X.astype(np.float32))
        workspace.RunNetOnce(net)
        X_hat = workspace.FetchBlob('X_hat')
        assert np.allclose(X, X_hat), (X, X_hat)


if __name__ == "__main__":
    unittest.main()
