from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import unittest

import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import cnn, core, workspace, test_util

@unittest.skipIf(not workspace.C.has_mkldnn, "Skipping as we do not have mkldnn.")
class TestMKLBasic(test_util.TestCase):
    def testReLUConsistencyWithCPU(self):
        X = np.random.randn(128, 4096).astype(np.float32)
        mkl_do = core.DeviceOption(caffe2_pb2.MKLDNN)
        # Makes sure that feed works.
        workspace.FeedBlob("X", X)
        workspace.FeedBlob("X_mkl", X, device_option=mkl_do)
        model = cnn.CNNModelHelper()
        # Makes sure that we can run relu.
        model.Relu("X", "Y")
        model.Relu("X_mkl", "Y_mkl", device_option=mkl_do)
        workspace.CreateNet(model.net)
        workspace.RunNet(model.net)
        # makes sure that the results are good.
        np.testing.assert_allclose(
            workspace.FetchBlob("Y"),
            workspace.FetchBlob("Y_mkl"),
            atol=1e-10,
            rtol=1e-10)
        runtime = workspace.BenchmarkNet(model.net.Proto().name, 1, 10, True)
        # The returned runtime is the time of
        # [whole_net, cpu_op, mkl_op]
        # so we will assume that the MKL one runs faster than the CPU one.
        self.assertTrue(runtime[1] >= runtime[2])
        print("CPU runtime {}, MKL runtime {}.".format(runtime[1], runtime[2]))


if __name__ == '__main__':
    unittest.main()
