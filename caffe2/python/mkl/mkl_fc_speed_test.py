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
    def testFCSpeed(self):
        # We randomly select a shape to test the speed. Intentionally we
        # test a batch size of 1 since this may be the most frequent use
        # case for MKL during deployment time.
        X = np.random.rand(1, 256, 6, 6).astype(np.float32) - 0.5
        #X = np.random.rand(32, 256*6*6).astype(np.float32) - 0.5
        W = np.random.rand(4096, 9216).astype(np.float32) - 0.5
        b = np.random.rand(4096).astype(np.float32) - 0.5
        mkl_do = core.DeviceOption(caffe2_pb2.MKLDNN)
        # Makes sure that feed works.
        workspace.FeedBlob("X", X)
        workspace.FeedBlob("W", W)
        workspace.FeedBlob("b", b)
        workspace.FeedBlob("X_mkl", X, device_option=mkl_do)
        workspace.FeedBlob("W_mkl", W, device_option=mkl_do)
        workspace.FeedBlob("b_mkl", b, device_option=mkl_do)
        net = core.Net("test")
        # Makes sure that we can run relu.
        net.FC(["X", "W", "b"], "Y")
        net.FC(["X_mkl", "W_mkl", "b_mkl"], "Y_mkl", device_option=mkl_do)

        workspace.CreateNet(net)
        workspace.RunNet(net)
        # makes sure that the results are good.
        np.testing.assert_allclose(
            workspace.FetchBlob("Y"),
            workspace.FetchBlob("Y_mkl"),
            atol=1e-2,
            rtol=1e-2)
        runtime = workspace.BenchmarkNet(net.Proto().name, 1, 100, True)

        print("FC CPU runtime {}, MKL runtime {}.".format(runtime[1], runtime[2]))

    def testConvReluMaxPoolFcSpeed(self):
        # We randomly select a shape to test the speed. Intentionally we
        # test a batch size of 1 since this may be the most frequent use
        # case for MKL during deployment time.
        X = np.random.rand(1, 256, 13, 13).astype(np.float32) - 0.5
        W = np.random.rand(256, 256, 3, 3).astype(np.float32) - 0.5
        b = np.random.rand(256).astype(np.float32) - 0.5

        w_fc = np.random.rand(4096, 9216).astype(np.float32) - 0.5
        b_fc = np.random.rand(4096).astype(np.float32) - 0.5
        mkl_do = core.DeviceOption(caffe2_pb2.MKLDNN)
        # Makes sure that feed works.
        workspace.FeedBlob("X", X)
        workspace.FeedBlob("W", W)
        workspace.FeedBlob("b", b)
        workspace.FeedBlob("w_fc", w_fc)
        workspace.FeedBlob("b_fc", b_fc)
        workspace.FeedBlob("X_mkl", X, device_option=mkl_do)
        workspace.FeedBlob("W_mkl", W, device_option=mkl_do)
        workspace.FeedBlob("b_mkl", b, device_option=mkl_do)
        workspace.FeedBlob("w_fc_mkl", w_fc, device_option=mkl_do)
        workspace.FeedBlob("b_fc_mkl", b_fc, device_option=mkl_do)

        net = core.Net("test")

        net.Conv(["X", "W", "b"], "C", pad=1, stride=1, kernel=3)
        net.Relu("C", "R")
        net.MaxPool("R", "P", stride=2, kernel=3)
        net.FC(["P","w_fc", "b_fc"], "Y")

        net.Conv(["X_mkl", "W_mkl", "b_mkl"], "C_mkl",
                 pad=1, stride=1, kernel=3, device_option=mkl_do)
        net.Relu("C_mkl", "R_mkl", device_option=mkl_do)
        net.MaxPool("R_mkl", "P_mkl",
                 stride=2, kernel=3, device_option=mkl_do)
        net.FC(["P_mkl","w_fc_mkl", "b_fc_mkl"], "Y_mkl", device_option=mkl_do)

        workspace.CreateNet(net)
        workspace.RunNet(net)
        # makes sure that the results are good.
        np.testing.assert_allclose(
            workspace.FetchBlob("Y"),
            workspace.FetchBlob("Y_mkl"),
            atol=1e-2,
            rtol=1e-2)
        runtime = workspace.BenchmarkNet(net.Proto().name, 1, 100, True)


if __name__ == '__main__':
    unittest.main()
