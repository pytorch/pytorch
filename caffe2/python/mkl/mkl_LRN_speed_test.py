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
    def testLRNSpeed(self):
        # We randomly select a shape to test the speed. Intentionally we
        # test a batch size of 1 since this may be the most frequent use
        # case for MKL during deployment time.
        X = np.random.rand(1, 2, 224, 224).astype(np.float32)
        mkl_do = core.DeviceOption(caffe2_pb2.MKLDNN)
        # Makes sure that feed works.
        workspace.FeedBlob("X", X)
        workspace.FeedBlob("X_mkl", X, device_option=mkl_do)
        net = core.Net("test")
        # Makes sure that we can run relu.
        net.LRN("X", ["Y", "Y_Scale"], size=5, alpha=0.001, beta=0.75, bias=2.0, order="NCHW")
        net.LRN("X_mkl", ["Y_mkl", "Y_Scale_mkl"], size=5, alpha=0.001, beta=0.75, bias=2.0, order="NCHW", device_option=mkl_do)
        workspace.CreateNet(net)
        workspace.RunNet(net)

        # makes sure that the results are good.
        np.testing.assert_allclose(
            workspace.FetchBlob("Y"),
            workspace.FetchBlob("Y_mkl"),
            atol=1e-2,
            rtol=1e-2)
        runtime = workspace.BenchmarkNet(net.Proto().name, 1, 100, True)

        print("LRN CPU runtime {}, MKL runtime {}.".format(runtime[1], runtime[2]))

    def testConvReluLRNSpeed(self):
        # We randomly select a shape to test the speed. Intentionally we
        # test a batch size of 1 since this may be the most frequent use
        # case for MKL during deployment time.
        X = np.random.rand(1, 3, 224, 224).astype(np.float32) - 0.5
        W = np.random.rand(64, 3, 11, 11).astype(np.float32) - 0.5
        b = np.random.rand(64).astype(np.float32) - 0.5

        mkl_do = core.DeviceOption(caffe2_pb2.MKLDNN)
        # Makes sure that feed works.
        workspace.FeedBlob("X", X)
        workspace.FeedBlob("W", W)
        workspace.FeedBlob("b", b)
        workspace.FeedBlob("X_mkl", X, device_option=mkl_do)
        workspace.FeedBlob("W_mkl", W, device_option=mkl_do)
        workspace.FeedBlob("b_mkl", b, device_option=mkl_do)

        net = core.Net("test")

        net.Conv(["X", "W", "b"], "C", pad=1, stride=1, kernel=11)
        net.Conv(["X_mkl", "W_mkl", "b_mkl"], "C_mkl",
                 pad=1, stride=1, kernel=11, device_option=mkl_do)
        net.Relu("C", "R")
        net.Relu("C_mkl", "R_mkl", device_option=mkl_do)
        net.LRN("R", ["Y", "Y_Scale"], size=5, alpha=0.001, beta=0.75, bias=2.0, order="NCHW")
        net.LRN("R_mkl", ["Y_mkl", "Y_Scale_mkl"],size=5, alpha=0.001, beta=0.75, bias=2.0, order="NCHW", device_option=mkl_do)

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
