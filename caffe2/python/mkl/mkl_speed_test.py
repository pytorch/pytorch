



import unittest

import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace, test_util


@unittest.skipIf(not workspace.C.has_mkldnn, "Skipping as we do not have mkldnn.")
class TestMKLBasic(test_util.TestCase):
    def testReLUSpeed(self):
        X = np.random.randn(128, 4096).astype(np.float32)
        mkl_do = core.DeviceOption(caffe2_pb2.MKLDNN)
        # Makes sure that feed works.
        workspace.FeedBlob("X", X)
        workspace.FeedBlob("X_mkl", X, device_option=mkl_do)
        net = core.Net("test")
        # Makes sure that we can run relu.
        net.Relu("X", "Y")
        net.Relu("X_mkl", "Y_mkl", device_option=mkl_do)
        workspace.CreateNet(net)
        workspace.RunNet(net)
        # makes sure that the results are good.
        np.testing.assert_allclose(
            workspace.FetchBlob("Y"),
            workspace.FetchBlob("Y_mkl"),
            atol=1e-10,
            rtol=1e-10)
        runtime = workspace.BenchmarkNet(net.Proto().name, 1, 100, True)

        # The returned runtime is the time of
        # [whole_net, cpu_op, mkl_op]
        # so we will assume that the MKL one runs faster than the CPU one.

        # Note(Yangqing): in fact, it seems that in optimized mode, this is
        # not always guaranteed - MKL runs slower than the Eigen vectorized
        # version, so I am turning this assertion off.
        #self.assertTrue(runtime[1] >= runtime[2])

        print("Relu CPU runtime {}, MKL runtime {}.".format(runtime[1], runtime[2]))


    def testConvSpeed(self):
        # We randomly select a shape to test the speed. Intentionally we
        # test a batch size of 1 since this may be the most frequent use
        # case for MKL during deployment time.
        X = np.random.rand(1, 256, 27, 27).astype(np.float32) - 0.5
        W = np.random.rand(192, 256, 3, 3).astype(np.float32) - 0.5
        b = np.random.rand(192).astype(np.float32) - 0.5
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
        net.Conv(["X", "W", "b"], "Y", pad=1, stride=1, kernel=3)
        net.Conv(["X_mkl", "W_mkl", "b_mkl"], "Y_mkl",
                 pad=1, stride=1, kernel=3, device_option=mkl_do)
        workspace.CreateNet(net)
        workspace.RunNet(net)
        # makes sure that the results are good.
        np.testing.assert_allclose(
            workspace.FetchBlob("Y"),
            workspace.FetchBlob("Y_mkl"),
            atol=1e-2,
            rtol=1e-2)
        runtime = workspace.BenchmarkNet(net.Proto().name, 1, 100, True)

        print("Conv CPU runtime {}, MKL runtime {}.".format(runtime[1], runtime[2]))


if __name__ == '__main__':
    unittest.main()
