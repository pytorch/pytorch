import numpy as np
import unittest

from caffe2.proto import caffe2_pb2
from caffe2.python import (
    workspace,
    device_checker,
    test_util,
    model_helper,
    brew,
)


class TestMiniAlexNet(test_util.TestCase):

    def _MiniAlexNetNoDropout(self, order):
        # First, AlexNet using the cnn wrapper.
        model = model_helper.ModelHelper(name="alexnet")
        conv1 = brew.conv(
            model,
            "data",
            "conv1",
            3,
            16,
            11,
            ("XavierFill", {}),
            ("ConstantFill", {}),
            stride=4,
            pad=0
        )
        relu1 = brew.relu(model, conv1, "relu1")
        norm1 = brew.lrn(model, relu1, "norm1", size=5, alpha=0.0001, beta=0.75)
        pool1 = brew.max_pool(model, norm1, "pool1", kernel=3, stride=2)
        conv2 = brew.group_conv(
            model,
            pool1,
            "conv2",
            16,
            32,
            5,
            ("XavierFill", {}),
            ("ConstantFill", {"value": 0.1}),
            group=2,
            stride=1,
            pad=2
        )
        relu2 = brew.relu(model, conv2, "relu2")
        norm2 = brew.lrn(model, relu2, "norm2", size=5, alpha=0.0001, beta=0.75)
        pool2 = brew.max_pool(model, norm2, "pool2", kernel=3, stride=2)
        conv3 = brew.conv(
            model,
            pool2,
            "conv3",
            32,
            64,
            3,
            ("XavierFill", {'std': 0.01}),
            ("ConstantFill", {}),
            pad=1
        )
        relu3 = brew.relu(model, conv3, "relu3")
        conv4 = brew.group_conv(
            model,
            relu3,
            "conv4",
            64,
            64,
            3,
            ("XavierFill", {}),
            ("ConstantFill", {"value": 0.1}),
            group=2,
            pad=1
        )
        relu4 = brew.relu(model, conv4, "relu4")
        conv5 = brew.group_conv(
            model,
            relu4,
            "conv5",
            64,
            32,
            3,
            ("XavierFill", {}),
            ("ConstantFill", {"value": 0.1}),
            group=2,
            pad=1
        )
        relu5 = brew.relu(model, conv5, "relu5")
        pool5 = brew.max_pool(model, relu5, "pool5", kernel=3, stride=2)
        fc6 = brew.fc(
            model, pool5, "fc6", 1152, 1024, ("XavierFill", {}),
            ("ConstantFill", {"value": 0.1})
        )
        relu6 = brew.relu(model, fc6, "relu6")
        fc7 = brew.fc(
            model, relu6, "fc7", 1024, 1024, ("XavierFill", {}),
            ("ConstantFill", {"value": 0.1})
        )
        relu7 = brew.relu(model, fc7, "relu7")
        fc8 = brew.fc(
            model, relu7, "fc8", 1024, 5, ("XavierFill", {}),
            ("ConstantFill", {"value": 0.0})
        )
        pred = brew.softmax(model, fc8, "pred")
        xent = model.LabelCrossEntropy([pred, "label"], "xent")
        loss = model.AveragedLoss([xent], ["loss"])
        model.AddGradientOperators([loss])
        return model

    def _testMiniAlexNet(self, order):
        # First, we get all the random initialization of parameters.
        model = self._MiniAlexNetNoDropout(order)
        workspace.ResetWorkspace()
        workspace.RunNetOnce(model.param_init_net)
        inputs = dict(
            [(str(name), workspace.FetchBlob(str(name))) for name in
             model.params]
        )
        if order == "NCHW":
            inputs["data"] = np.random.rand(4, 3, 227, 227).astype(np.float32)
        else:
            inputs["data"] = np.random.rand(4, 227, 227, 3).astype(np.float32)
        inputs["label"] = np.array([1, 2, 3, 4]).astype(np.int32)

        cpu_device = caffe2_pb2.DeviceOption()
        cpu_device.device_type = caffe2_pb2.CPU
        gpu_device = caffe2_pb2.DeviceOption()
        gpu_device.device_type = workspace.GpuDeviceType

        checker = device_checker.DeviceChecker(0.05, [cpu_device, gpu_device])
        ret = checker.CheckNet(
            model.net.Proto(),
            inputs,
            # The indices sometimes may be sensitive to small numerical
            # differences in the input, so we ignore checking them.
            ignore=['_pool1_idx', '_pool2_idx', '_pool5_idx']
        )
        self.assertEqual(ret, True)

    @unittest.skipIf(not workspace.has_gpu_support
                    and not workspace.has_hip_support,
                     "No GPU support. Skipping test.")
    def testMiniAlexNetNCHW(self):
        self._testMiniAlexNet("NCHW")

    # No Group convolution support for NHWC right now
    #@unittest.skipIf(not workspace.has_gpu_support,
    #                 "No GPU support. Skipping test.")
    #def testMiniAlexNetNHWC(self):
    #    self._testMiniAlexNet("NHWC")


if __name__ == '__main__':
    unittest.main()
