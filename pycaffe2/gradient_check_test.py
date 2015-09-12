import numpy as np
from pycaffe2 import core, device_checker, gradient_checker, workspace
from caffe2.proto import caffe2_pb2, caffe2_legacy_pb2

import sys
import unittest

if workspace.has_gpu_support and workspace.NumberOfGPUs() > 0:
  gpu_device_option = caffe2_pb2.DeviceOption()
  gpu_device_option.device_type = caffe2_pb2.CUDA
  cpu_device_option = caffe2_pb2.DeviceOption()
  device_checker = device_checker.DeviceChecker(
      0.01, [gpu_device_option, cpu_device_option])
  gradient_checkers = [
      gradient_checker.GradientChecker(
          0.005, 0.05, gpu_device_option, "gpu_checker_ws"),
      gradient_checker.GradientChecker(
          0.01, 0.05, cpu_device_option, "cpu_checker_ws"),
  ]
else:
  cpu_device_option = caffe2_pb2.DeviceOption()
  device_checker = device_checker.DeviceChecker(
      0.01, [cpu_device_option])
  gradient_checkers = [
      gradient_checker.GradientChecker(
          0.01, 0.05, cpu_device_option, "cpu_checker_ws")
  ]



class TestConvLegacyPooling(unittest.TestCase):
  def setUp(self):
    self.test_configs = [
        # stride, kernel, legacy_pad, size, order
        (1, 1, 1, 7, "NHWC"),
        (1, 1, 2, 7, "NHWC"),
        (1, 3, 1, 7, "NHWC"),
        (1, 3, 2, 7, "NHWC"),
        (1, 5, 1, 14, "NHWC"),
        (1, 5, 2, 14, "NHWC"),
        (2, 7, 1, 24, "NHWC"),
        (2, 7, 2, 24, "NHWC"),
        (1, 1, 1, 7, "NCHW"),
        (1, 1, 2, 7, "NCHW"),
        (1, 3, 1, 7, "NCHW"),
        (1, 3, 2, 7, "NCHW"),
        (1, 5, 1, 14, "NCHW"),
        (1, 5, 2, 14, "NCHW"),
        (2, 7, 1, 24, "NCHW"),
        (2, 7, 2, 24, "NCHW"),
    ]

  def testConvolutionLegacyPadding(self):
    for stride, kernel, legacy_pad, size, order in self.test_configs:
      print 'conv', stride, kernel, legacy_pad, size, order
      op = core.CreateOperator("Conv")(
          ["X", "w", "b"], ["Y"], stride=stride, kernel=kernel,
          legacy_pad=legacy_pad, order=order)
      if order == "NHWC":
          X = np.random.rand(2, size, size, 3).astype(np.float32) - 0.5
          w = np.random.rand(4, kernel, kernel, 3).astype(np.float32) - 0.5
      else:
          X = np.random.rand(2, 3, size, size).astype(np.float32) - 0.5
          w = np.random.rand(4, 3, kernel, kernel).astype(np.float32) - 0.5
      b = np.random.rand(4).astype(np.float32) - 0.5
      res = device_checker.CheckSimple(op, [X, w, b], [0])
      self.assertTrue(res)
      for checker in gradient_checkers:
        for i in range(3):
          res, grad, grad_estimated = checker.CheckSimple(
              op, [X, w, b], i, [0])
          self.assertTrue(res)

class TestMaxPoolingLegacyPadding(unittest.TestCase):
  def setUp(self):
    self.test_configs = [
        (2, 3, 2, 12, "NHWC"),
        (2, 3, 2, 16, "NHWC"),
        (1, 3, 2, 8, "NHWC"),
        (1, 3, 2, 14, "NHWC"),
        (2, 3, 2, 14, "NHWC"),
        (1, 3, 2, 7, "NHWC"),
        (2, 3, 2, 12, "NCHW"),
        (2, 3, 2, 16, "NCHW"),
        (1, 3, 2, 8, "NCHW"),
        (1, 3, 2, 14, "NCHW"),
        (2, 3, 2, 14, "NCHW"),
        (1, 3, 2, 7, "NCHW"),
    ]

  def testMaxPoolingLegacyPadding(self):
    for stride, kernel, legacy_pad, size, order in self.test_configs:
      print 'MaxPool', stride, kernel, legacy_pad, size, order
      op = core.CreateOperator("MaxPool")(
          ["X"], ["Y", "Y_maxid"], stride=stride, kernel=kernel,
          legacy_pad=legacy_pad, order=order)
      # In order to avoid the problem of race conditions, we will do a randperm
      # so that the values will be apart at least 0.01
      if order == "NHWC":
          X = np.random.permutation(1 * size * size * 3).reshape(1, size, size, 3).astype(np.float32) * 0.01
      else:
          X = np.random.permutation(1 * size * size * 3).reshape(1, 3, size, size).astype(np.float32) * 0.01
      res = device_checker.CheckSimple(op, [X], [0])
      self.assertTrue(res)
      for checker in gradient_checkers:
        res, grad, grad_estimated = checker.CheckSimple(op, [X], 0, [0])
        self.assertTrue(res)

class TestAveragePoolingLegacyPadding(unittest.TestCase):
  def setUp(self):
    self.test_configs = [
        (1, 7, 1, 7, "NHWC"),
        (1, 7, 2, 7, "NHWC"),
        (1, 7, 1, 7, "NCHW"),
        (1, 7, 2, 7, "NCHW"),
    ]

  def testAveragePoolingLegacyPadding(self):
    for stride, kernel, legacy_pad, size, order in self.test_configs:
      print 'AveragePool', stride, kernel, legacy_pad, size, order
      op = core.CreateOperator("AveragePool")(
          ["X"], ["Y"], stride=stride, kernel=kernel,
          legacy_pad=legacy_pad, order=order)
      if order == "NHWC":
        X = np.random.rand(2, size, size, 3).astype(np.float32)
      else:
        X = np.random.rand(2, 3, size, size).astype(np.float32)
      res = device_checker.CheckSimple(op, [X], [0])
      self.assertTrue(res)
      for checker in gradient_checkers:
        res, grad, grad_estimated = checker.CheckSimple(op, [X], 0, [0])
        self.assertTrue(res)

class TestLRN(unittest.TestCase):
  def setUp(self):
    self.test_configs = [
        (6, 10),
        (3, 13),
    ]

  def testLRN(self):
    for input_size, depth in self.test_configs:
      op = core.CreateOperator("LRN")(
          ["X"], ["Y", "Y_scale"], size=11, alpha=0.001, beta=0.5, bias=2.0, order="NHWC")
      X = np.random.rand(2, input_size, input_size, depth).astype(np.float32)
      res = device_checker.CheckSimple(op, [X], [0])
      self.assertTrue(res)
      for checker in gradient_checkers:
        res, grad, grad_estimated = checker.CheckSimple(op, [X], 0, [0])
        self.assertTrue(res)

class TestDepthConcat(unittest.TestCase):
  def setUp(self):
    self.test_configs = [
        # input_size, depth1, depth2, depth3, depth4
        (3, 2, 3, 4, 5),
        (4, 5, 4, 3, 2),
    ]

  def testDepthConcatNHWC(self):
    for input_size, d1, d2, d3, d4 in self.test_configs:
      op = core.CreateOperator("DepthConcat")(
          ["X1", "X2", "X3", "X4"], ["Y", "Y_dims"], order="NHWC")
      Xs = [np.random.rand(2, input_size, input_size, d1).astype(np.float32),
            np.random.rand(2, input_size, input_size, d2).astype(np.float32),
            np.random.rand(2, input_size, input_size, d3).astype(np.float32),
            np.random.rand(2, input_size, input_size, d4).astype(np.float32)]
      for i in range(4):
        res = device_checker.CheckSimple(op, Xs, [0])
        self.assertTrue(res)
        for checker in gradient_checkers:
          res, grad, grad_estimated = checker.CheckSimple(op, Xs, i, [0])
          self.assertTrue(res)

  def testDepthConcatNCHW(self):
    for input_size, d1, d2, d3, d4 in self.test_configs:
      op = core.CreateOperator("DepthConcat")(
          ["X1", "X2", "X3", "X4"], ["Y", "Y_dims"], order="NCHW")
      Xs = [np.random.rand(2, d1, input_size, input_size).astype(np.float32),
            np.random.rand(2, d2, input_size, input_size).astype(np.float32),
            np.random.rand(2, d3, input_size, input_size).astype(np.float32),
            np.random.rand(2, d4, input_size, input_size).astype(np.float32)]
      for i in range(4):
        res = device_checker.CheckSimple(op, Xs, [0])
        self.assertTrue(res)
        for checker in gradient_checkers:
          res, grad, grad_estimated = checker.CheckSimple(op, Xs, i, [0])
          self.assertTrue(res)

class TestRelu(unittest.TestCase):
  def setUp(self):
    self.test_configs = [
        # input size
        (1, 1),
        (2, 1),
        (1, 3, 3, 1),
        (2, 3, 3, 1),
        (1, 5, 5, 3),
        (2, 5, 5, 3),
    ]

  def testRelu(self):
    for input_size in self.test_configs:
      op = core.CreateOperator("Relu")(["X"], ["Y"])
      X = np.random.rand(*input_size).astype(np.float32)
      # go away from the origin point to avoid kink problems
      X += 0.01 * np.sign(X)
      X[X==0] = 0.01
      res = device_checker.CheckSimple(op, [X], [0])
      self.assertTrue(res)
      for checker in gradient_checkers:
        res, grad, grad_estimated = checker.CheckSimple(op, [X], 0, [0])
        self.assertTrue(res)


if __name__ == '__main__':
  unittest.main()