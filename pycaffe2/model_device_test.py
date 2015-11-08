import numpy as np
import unittest
import sys

from caffe2.proto import caffe2_pb2, caffe2_legacy_pb2
from pycaffe2 import core, cnn, workspace, device_checker

class TestMNISTLeNet(unittest.TestCase):
  def setUp(self):
    pass

  def _MNISTNetworks(self):
    init_net = core.Net("init")
    filter1 = init_net.XavierFill([], "filter1", shape=[20, 1, 5, 5])
    bias1 = init_net.ConstantFill([], "bias1", shape=[20,], value=0.0)
    filter2 = init_net.XavierFill([], "filter2", shape=[50, 20, 5, 5])
    bias2 = init_net.ConstantFill([], "bias2", shape=[50,], value=0.0)
    W3 = init_net.XavierFill([], "W3", shape=[500, 800])
    B3 = init_net.ConstantFill([], "B3", shape=[500], value=0.0)
    W4 = init_net.XavierFill([], "W4", shape=[10, 500])
    B4 = init_net.ConstantFill([], "B4", shape=[10], value=0.0)
    data, label = init_net.TensorProtosDBInput(
        [], ["data", "label"], batch_size=64,
        db="gen/data/mnist/mnist-train-nchw-minidb", db_type="minidb")
    LR = init_net.ConstantFill([], "LR", shape=[1], value=-0.1)
    ONE = init_net.ConstantFill([], "ONE", shape=[1], value=1.0)

    train_net = core.Net("train")
    conv1 = train_net.Conv([data, filter1, bias1], "conv1", kernel=5, pad=0, stride=1, order="NCHW")
    pool1, maxid1 = conv1.MaxPool([], ["pool1", "maxid1"], kernel=2, stride=2, order="NCHW")
    conv2 = pool1.Conv([filter2, bias2], "conv2", kernel=5, pad=0, stride=1, order="NCHW")
    pool2, maxid2 = conv2.MaxPool([], ["pool2", "maxid2"], kernel=2, stride=2, order="NCHW")
    flatten2 = pool2.Flatten([], "pool2_flatten")
    softmax = (flatten2.FC([W3, B3], "fc3")
                       .Relu([], "fc3_relu")
                       .FC([W4, B4], "pred")
                       .Softmax([], "softmax"))
    # Cross entropy, and accuracy
    xent = softmax.LabelCrossEntropy([label], "xent")
    # The loss function.
    loss = xent.AveragedLoss([], ["loss"])
    # Get gradient, skipping the input and flatten layers.
    train_net.AddGradientOperators()
    accuracy = softmax.Accuracy([label], "accuracy")
    # parameter update.
    for param in [filter1, bias1, filter2, bias2, W3, B3, W4, B4]:
      train_net.WeightedSum([param, ONE, param.Grad(), LR], param)
    return init_net, train_net

  def testMNISTNetworks(self):
    # First, we get all the random initialization of parameters.
    init_net, train_net = self._MNISTNetworks()
    workspace.ResetWorkspace()
    workspace.RunNetOnce(init_net)
    inputs = dict([(str(name), workspace.FetchBlob(str(name)))
                   for name in workspace.Blobs()])
    cpu_device = caffe2_pb2.DeviceOption()
    cpu_device.device_type = caffe2_pb2.CPU
    gpu_device = caffe2_pb2.DeviceOption()
    gpu_device.device_type = caffe2_pb2.CUDA

    checker = device_checker.DeviceChecker(
        1e-2, [cpu_device, gpu_device])
    ret = checker.CheckNet(
        train_net.Proto(), inputs,
        ignore=['maxid1', 'maxid2'])
    self.assertEqual(ret, True)


class TestMiniAlexNet(unittest.TestCase):
  def setUp(self):
    pass

  def _MiniAlexNetNoDropout(self, order):
    # First, AlexNet using the cnn wrapper.
    model = cnn.CNNModelHelper(order, name="alexnet")
    conv1 = model.Conv("data", "conv1", 3, 16, 11,
                   ("XavierFill", {}),
                   ("ConstantFill", {}), stride=4, pad=0)
    relu1 = model.Relu(conv1, "relu1")
    norm1 = model.LRN(relu1, "norm1", size=5, alpha=0.0001, beta=0.75)
    pool1 = model.MaxPool(norm1, "pool1", kernel=3, stride=2)
    conv2 = model.GroupConv(pool1, "conv2", 16, 32, 5,
                            ("XavierFill", {}),
                            ("ConstantFill", {"value": 0.1}),
                            group=2, stride=1, pad=2)
    relu2 = model.Relu(conv2, "relu2")
    norm2 = model.LRN(relu2, "norm2", size=5, alpha=0.0001, beta=0.75)
    pool2 = model.MaxPool(norm2, "pool2", kernel=3, stride=2)
    conv3 = model.Conv(pool2, "conv3", 32, 64, 3,
                       ("XavierFill", {'std': 0.01}),
                       ("ConstantFill", {}), pad=1)
    relu3 = model.Relu(conv3, "relu3")
    conv4 = model.GroupConv(relu3, "conv4", 64, 64, 3,
                            ("XavierFill", {}),
                            ("ConstantFill", {"value": 0.1}),
                            group=2, pad=1)
    relu4 = model.Relu(conv4, "relu4")
    conv5 = model.GroupConv(relu4, "conv5", 64, 32, 3,
                            ("XavierFill", {}),
                            ("ConstantFill", {"value": 0.1}),
                            group=2, pad=1)
    relu5 = model.Relu(conv5, "relu5")
    pool5 = model.MaxPool(relu5, "pool5", kernel=3, stride=2)
    fc6 = model.FC(pool5, "fc6", 1152, 1024,
                  ("XavierFill", {}),
                  ("ConstantFill", {"value": 0.1}))
    relu6 = model.Relu(fc6, "relu6")
    fc7 = model.FC(relu6, "fc7", 1024, 1024,
                  ("XavierFill", {}),
                  ("ConstantFill", {"value": 0.1}))
    relu7 = model.Relu(fc7, "relu7")
    fc8 = model.FC(relu7, "fc8", 1024, 5,
                  ("XavierFill", {}),
                  ("ConstantFill", {"value": 0.0}))
    pred = model.Softmax(fc8, "pred")
    xent = model.LabelCrossEntropy([pred, "label"], "xent")
    loss = model.AveragedLoss([xent], ["loss"])
    model.AddGradientOperators()
    return model

  def _testMiniAlexNet(self, order):
    # First, we get all the random initialization of parameters.
    model = self._MiniAlexNetNoDropout(order);
    workspace.ResetWorkspace()
    workspace.RunNetOnce(model.param_init_net)
    inputs = dict(
        [(str(name), workspace.FetchBlob(str(name))) for name in model.params])
    if order == "NCHW":
      inputs["data"] = np.random.rand(4, 3, 227, 227).astype(np.float32)
    else:
      inputs["data"] = np.random.rand(4, 227, 227, 3).astype(np.float32)
    inputs["label"] = np.array([1, 2, 3, 4]).astype(np.int32)

    cpu_device = caffe2_pb2.DeviceOption()
    cpu_device.device_type = caffe2_pb2.CPU
    gpu_device = caffe2_pb2.DeviceOption()
    gpu_device.device_type = caffe2_pb2.CUDA

    checker = device_checker.DeviceChecker(
        1e-2, [cpu_device, gpu_device])
    ret = checker.CheckNet(
        model.net.Proto(), inputs,
        # The indices sometimes may be sensitive to small numerical differences
        # in the input, so we ignore checking them.
        ignore=['_pool1_idx', '_pool2_idx', '_pool5_idx'])
    self.assertEqual(ret, True)

  def testMiniAlexNet(self):
    self._testMiniAlexNet("NCHW")
    self._testMiniAlexNet("NHWC")


if __name__ == '__main__':
  if not workspace.has_gpu_support:
    print 'No GPU support. Skipping gpu test.'
  elif workspace.NumberOfGPUs() == 0:
    print 'No GPU device. Skipping gpu test.'
  else:
    unittest.main()
