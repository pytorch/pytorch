



import unittest

import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import cnn, core, workspace, test_util


@unittest.skipIf(not workspace.C.has_mkldnn, "Skipping as we do not have mkldnn.")
class TestMKLBasic(test_util.TestCase):
    def testSpatialBNTestingSpeed(self):

        input_channel = 10
        X = np.random.rand(1, input_channel, 100, 100).astype(np.float32) - 0.5
        scale = np.random.rand(input_channel).astype(np.float32) + 0.5
        bias = np.random.rand(input_channel).astype(np.float32) - 0.5
        mean = np.random.randn(input_channel).astype(np.float32)
        var = np.random.rand(input_channel).astype(np.float32) + 0.5

        mkl_do = core.DeviceOption(caffe2_pb2.MKLDNN)
        # Makes sure that feed works.
        workspace.FeedBlob("X", X)
        workspace.FeedBlob("scale", scale)
        workspace.FeedBlob("bias", bias)
        workspace.FeedBlob("mean", mean)
        workspace.FeedBlob("var", var)
        workspace.FeedBlob("X_mkl", X, device_option=mkl_do)
        workspace.FeedBlob("scale_mkl", scale, device_option=mkl_do)
        workspace.FeedBlob("bias_mkl", bias, device_option=mkl_do)
        workspace.FeedBlob("mean_mkl", mean, device_option=mkl_do)
        workspace.FeedBlob("var_mkl", var, device_option=mkl_do)
        net = core.Net("test")
        # Makes sure that we can run relu.
        net.SpatialBN(["X", "scale", "bias","mean","var"], "Y", order="NCHW",
            is_test=True,
            epsilon=1e-5)
        net.SpatialBN(["X_mkl", "scale_mkl", "bias_mkl","mean_mkl","var_mkl"], "Y_mkl", order="NCHW",
            is_test=True,
            epsilon=1e-5, device_option=mkl_do)

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

    def testSpatialBNTrainingSpeed(self):
        input_channel = 10
        X = np.random.rand(1, input_channel, 100, 100).astype(np.float32) - 0.5
        scale = np.random.rand(input_channel).astype(np.float32) + 0.5
        bias = np.random.rand(input_channel).astype(np.float32) - 0.5
        mean = np.random.randn(input_channel).astype(np.float32)
        var = np.random.rand(input_channel).astype(np.float32) + 0.5

        #mean = np.zeros(input_channel)
        #var = np.zeros(input_channel)

        mkl_do = core.DeviceOption(caffe2_pb2.MKLDNN)
        # Makes sure that feed works.
        workspace.FeedBlob("X", X)
        workspace.FeedBlob("scale", scale)
        workspace.FeedBlob("bias", bias)
        workspace.FeedBlob("mean", mean)
        workspace.FeedBlob("var", var)
        workspace.FeedBlob("X_mkl", X, device_option=mkl_do)
        workspace.FeedBlob("scale_mkl", scale, device_option=mkl_do)
        workspace.FeedBlob("bias_mkl", bias, device_option=mkl_do)
        workspace.FeedBlob("mean_mkl", mean, device_option=mkl_do)
        workspace.FeedBlob("var_mkl", var, device_option=mkl_do)
        net = core.Net("test")
        # Makes sure that we can run relu.
        net.SpatialBN(["X", "scale", "bias","mean", "var"],
            ["Y", "mean", "var", "saved_mean", "saved_var"],
            order="NCHW",
            is_test=False,
            epsilon=1e-5)
        net.SpatialBN(["X_mkl", "scale_mkl", "bias_mkl","mean_mkl","var_mkl"],
            ["Y_mkl", "mean_mkl", "var_mkl", "saved_mean_mkl", "saved_var_mkl"],
            order="NCHW",
            is_test=False,
            epsilon=1e-5,
            device_option=mkl_do)

        workspace.CreateNet(net)
        workspace.RunNet(net)

        # makes sure that the results are good.
        np.testing.assert_allclose(
            workspace.FetchBlob("Y"),
            workspace.FetchBlob("Y_mkl"),
            atol=1e-2,
            rtol=1e-2)
        np.testing.assert_allclose(
            workspace.FetchBlob("mean"),
            workspace.FetchBlob("mean_mkl"),
            atol=1e-2,
            rtol=1e-2)
        np.testing.assert_allclose(
            workspace.FetchBlob("var"),
            workspace.FetchBlob("var_mkl"),
            atol=1e-2,
            rtol=1e-2)

        runtime = workspace.BenchmarkNet(net.Proto().name, 1, 100, True)

        print("FC CPU runtime {}, MKL runtime {}.".format(runtime[1], runtime[2]))



if __name__ == '__main__':
    unittest.main()
