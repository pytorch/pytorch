import numpy as np
from caffe2.python import \
    core, device_checker, gradient_checker, test_util, workspace
from caffe2.proto import caffe2_pb2

import collections
import unittest


if workspace.has_gpu_support and workspace.NumCudaDevices() > 0:
    gpu_device_option = caffe2_pb2.DeviceOption()
    gpu_device_option.device_type = caffe2_pb2.CUDA
    cpu_device_option = caffe2_pb2.DeviceOption()
    gpu_device_checker = device_checker.DeviceChecker(
        0.01, [gpu_device_option]
    )
    device_checker = device_checker.DeviceChecker(
        0.01, [gpu_device_option, cpu_device_option]
    )
    gpu_gradient_checkers = [
        gradient_checker.GradientChecker(
            0.005, 0.05, gpu_device_option, "gpu_checker_ws"
        ),
    ]
    gradient_checkers = [
        gradient_checker.GradientChecker(
            0.005, 0.05, gpu_device_option, "gpu_checker_ws"
        ),
        gradient_checker.GradientChecker(
            0.01, 0.05, cpu_device_option, "cpu_checker_ws"
        ),
    ]
else:
    cpu_device_option = caffe2_pb2.DeviceOption()
    gpu_device_option = None
    gpu_device_checker = device_checker.DeviceChecker(
        0.01, []
    )
    device_checker = device_checker.DeviceChecker(0.01, [cpu_device_option])

    gradient_checkers = [
        gradient_checker.GradientChecker(
            0.01, 0.05, cpu_device_option, "cpu_checker_ws"
        )
    ]
    gpu_gradient_checkers = []


class TestConv(test_util.TestCase):

    def setUp(self):
        self.test_configs = [
            (1, 1, 0, 7, "NHWC", ""),
            (1, 1, 1, 7, "NHWC", "CUDNN"),
            (1, 3, 0, 7, "NHWC", ""),
            (1, 3, 1, 7, "NHWC", "CUDNN"),
            (1, 3, 2, 7, "NHWC", ""),
            (2, 3, 0, 7, "NHWC", "CUDNN"),
            (2, 3, 1, 7, "NHWC", ""),
            (2, 3, 2, 7, "NHWC", "CUDNN"),
            (1, 5, 0, 10, "NHWC", ""),
            (1, 5, 1, 10, "NHWC", "CUDNN"),
            (1, 5, 2, 10, "NHWC", ""),
            (1, 1, 0, 7, "NCHW", "CUDNN"),
            (1, 1, 1, 7, "NCHW", ""),
            (1, 3, 0, 7, "NCHW", "CUDNN"),
            (1, 3, 1, 7, "NCHW", ""),
            (1, 3, 2, 7, "NCHW", "CUDNN"),
            (2, 3, 0, 7, "NCHW", ""),
            (2, 3, 1, 7, "NCHW", "CUDNN"),
            (2, 3, 2, 7, "NCHW", ""),
            (1, 5, 0, 10, "NCHW", "CUDNN"),
            (1, 5, 1, 10, "NCHW", ""),
            (1, 5, 2, 10, "NCHW", "CUDNN"),
        ]

    def testConvolutionnPadding(self):
        for stride, kernel, pad, size, order, engine in self.test_configs:
            print('conv {} {} {} {} {} {}'.format(
                stride, kernel, pad, size, order, engine)
            )
            op = core.CreateOperator("Conv",
                                     ["X", "w", "b"],
                                     ["Y"],
                                     stride=stride,
                                     kernel=kernel,
                                     pad=pad,
                                     order=order,
                                     engine=engine,
                                     )
            if order == "NHWC":
                X = np.random.rand(2, size, size, 3).astype(np.float32) - 0.5
                w = np.random.rand(4, kernel, kernel,
                                   3).astype(np.float32) - 0.5
            else:
                X = np.random.rand(2, 3, size, size).astype(np.float32) - 0.5
                w = np.random.rand(4, 3, kernel,
                                   kernel).astype(np.float32) - 0.5
            b = np.random.rand(4).astype(np.float32) - 0.5
            res = device_checker.CheckSimple(op, [X, w, b], [0])
            self.assertTrue(res)
            for checker in gradient_checkers:
                for i in range(3):
                    res, grad, grad_estimated = checker.CheckSimple(
                        op, [X, w, b], i, [0]
                    )
                    self.assertTrue(res)

    def testConvolutionLayoutCorrespondence(self):
        for stride, kernel, pad, size, _, engine in self.test_configs:
            print('conv {} {} {} {} {}'.format(
                stride, kernel, pad, size, engine)
            )
            for device_option in device_checker._device_options:
                X = np.random.rand(2, size, size, 3).astype(np.float32) - 0.5
                w = np.random.rand(4, kernel, kernel,
                                   3).astype(np.float32) - 0.5
                b = np.random.rand(4).astype(np.float32) - 0.5
                outputs = {}
                for order in ["NCHW", "NHWC"]:
                    op = core.CreateOperator("Conv",
                                             ["X", "w", "b"],
                                             ["Y"],
                                             stride=stride,
                                             kernel=kernel,
                                             pad=pad,
                                             order=order,
                                             engine=engine,
                                             device_option=device_option,
                                             )
                    if order == "NCHW":
                        X_f = X.transpose((0, 3, 1, 2))
                        w_f = w.transpose((0, 3, 1, 2))
                    else:
                        X_f = X
                        w_f = w
                    workspace.FeedBlob("X", X_f, device_option=device_option)
                    workspace.FeedBlob("w", w_f, device_option=device_option)
                    workspace.FeedBlob("b", b, device_option=device_option)
                    workspace.RunOperatorOnce(op)
                    outputs[order] = workspace.FetchBlob("Y")
                np.testing.assert_allclose(
                    outputs["NCHW"],
                    outputs["NHWC"].transpose((0, 3, 1, 2)),
                    atol=1e-4,
                    rtol=1e-4)


class TestConvLegacyPooling(test_util.TestCase):

    def setUp(self):
        self.test_configs = [
            # stride, kernel, legacy_pad, size, order
            (1, 1, 1, 7, "NHWC"),
            (1, 1, 2, 7, "NHWC"),
            (1, 3, 1, 7, "NHWC"),
            (1, 3, 2, 7, "NHWC"),
            (1, 5, 1, 10, "NHWC"),
            (1, 5, 2, 10, "NHWC"),
            (2, 7, 1, 10, "NHWC"),
            (2, 7, 2, 10, "NHWC"),
            (1, 1, 1, 7, "NCHW"),
            (1, 1, 2, 7, "NCHW"),
            (1, 3, 1, 7, "NCHW"),
            (1, 3, 2, 7, "NCHW"),
            (1, 5, 1, 10, "NCHW"),
            (1, 5, 2, 10, "NCHW"),
            (2, 7, 1, 10, "NCHW"),
            (2, 7, 2, 10, "NCHW"),
        ]

    def testConvolutionLegacyPadding(self):
        for stride, kernel, legacy_pad, size, order in self.test_configs:
            print('conv legacypad {} {} {} {} {}'.format(
                stride, kernel, legacy_pad, size, order)
            )
            op = core.CreateOperator("Conv",
                                     ["X", "w", "b"],
                                     ["Y"],
                                     stride=stride,
                                     kernel=kernel,
                                     legacy_pad=legacy_pad,
                                     order=order
                                     )
            if order == "NHWC":
                X = np.random.rand(2, size, size, 3).astype(np.float32) - 0.5
                w = np.random.rand(4, kernel, kernel,
                                   3).astype(np.float32) - 0.5
            else:
                X = np.random.rand(2, 3, size, size).astype(np.float32) - 0.5
                w = np.random.rand(4, 3, kernel,
                                   kernel).astype(np.float32) - 0.5
            b = np.random.rand(4).astype(np.float32) - 0.5
            res = device_checker.CheckSimple(op, [X, w, b], [0])
            self.assertTrue(res)
            for checker in gradient_checkers:
                for i in range(3):
                    res, grad, grad_estimated = checker.CheckSimple(
                        op, [X, w, b], i, [0]
                    )
                    self.assertTrue(res)


class TestMaxPoolingLegacyPadding(test_util.TestCase):

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
            print('MaxPool {} {} {} {} {}'.format(stride, kernel, legacy_pad,
                                                  size, order))
            op = core.CreateOperator("MaxPool",
                                     ["X"],
                                     ["Y"],
                                     stride=stride,
                                     kernel=kernel,
                                     legacy_pad=legacy_pad,
                                     order=order
                                     )
            # In order to avoid the problem of race conditions, we will do a
            # randperm so that the values will be apart at least 0.01
            if order == "NHWC":
                X = np.random.permutation(1 * size * size * 3).reshape(
                    1, size, size, 3).astype(np.float32) * 0.01
            else:
                X = np.random.permutation(1 * size * size * 3).reshape(
                    1, 3, size, size).astype(np.float32) * 0.01
            res = device_checker.CheckSimple(op, [X], [0])
            self.assertTrue(res)
            for checker in gradient_checkers:
                res, grad, grad_estimated = checker.CheckSimple(op, [X], 0, [0])
                self.assertTrue(res)


class TestAveragePoolingLegacyPadding(test_util.TestCase):

    def setUp(self):
        self.test_configs = [
            (1, 7, 1, 7, "NHWC"),
            (1, 7, 2, 7, "NHWC"),
            (1, 7, 1, 7, "NCHW"),
            (1, 7, 2, 7, "NCHW"),
        ]

    def testAveragePoolingLegacyPadding(self):
        for stride, kernel, legacy_pad, size, order in self.test_configs:
            print('AveragePool {} {} {} {} {}'.format(
                stride, kernel, legacy_pad, size, order))
            op = core.CreateOperator("AveragePool",
                                     ["X"],
                                     ["Y"],
                                     stride=stride,
                                     kernel=kernel,
                                     legacy_pad=legacy_pad,
                                     order=order
                                     )
            if order == "NHWC":
                X = np.random.rand(2, size, size, 3).astype(np.float32)
            else:
                X = np.random.rand(2, 3, size, size).astype(np.float32)
            res = device_checker.CheckSimple(op, [X], [0])
            self.assertTrue(res)
            for checker in gradient_checkers:
                res, grad, grad_estimated = checker.CheckSimple(op, [X], 0, [0])
                self.assertTrue(res)


class TestLRN(test_util.TestCase):

    def setUp(self):
        self.test_configs = [(6, 10), (3, 13), ]

    def testLRN(self):
        for input_size, depth in self.test_configs:
            op = core.CreateOperator("LRN",
                                     ["X"],
                                     ["Y", "Y_scale"],
                                     size=11,
                                     alpha=0.001,
                                     beta=0.5,
                                     bias=2.0,
                                     order="NHWC"
                                     )
            X = np.random.rand(2, input_size, input_size,
                               depth).astype(np.float32)
            res = device_checker.CheckSimple(op, [X], [0])
            self.assertTrue(res)
            for checker in gradient_checkers:
                res, grad, grad_estimated = checker.CheckSimple(op, [X], 0, [0])
                self.assertTrue(res)


class TestFlatten(test_util.TestCase):

    def testFlatten(self):
        op = core.CreateOperator("Flatten", ["X"], ["Y"])
        X = np.random.rand(2, 3, 4, 5).astype(np.float32)
        res = device_checker.CheckSimple(op, [X], [0])
        self.assertTrue(res)
        for checker in gradient_checkers:
            res, grad, grad_estimated = checker.CheckSimple(op, [X], 0, [0])
            self.assertTrue(res)


class TestConcat(test_util.TestCase):

    def setUp(self):
        self.test_configs = [
            # input_size, depth1, depth2, depth3, depth4
            (3, 2, 3, 4, 5),
            (4, 5, 4, 3, 2),
        ]

    def testConcatNHWC(self):
        for input_size, d1, d2, d3, d4 in self.test_configs:
            op = core.CreateOperator("Concat",
                                     ["X1", "X2", "X3", "X4"],
                                     ["Y", "Y_dims"],
                                     order="NHWC"
                                     )
            Xs = [
                np.random.rand(2, input_size, input_size,
                               d1).astype(np.float32),
                np.random.rand(2, input_size, input_size,
                               d2).astype(np.float32),
                np.random.rand(2, input_size, input_size,
                               d3).astype(np.float32),
                np.random.rand(2, input_size, input_size, d4).astype(np.float32)
            ]
            for i in range(4):
                res = device_checker.CheckSimple(op, Xs, [0])
                self.assertTrue(res)
                for checker in gradient_checkers:
                    res, grad, grad_estimated = checker.CheckSimple(op, Xs, i,
                                                                    [0])
                    self.assertTrue(res)

    def testConcatNCHW(self):
        for input_size, d1, d2, d3, d4 in self.test_configs:
            op = core.CreateOperator("Concat",
                                     ["X1", "X2", "X3", "X4"],
                                     ["Y", "Y_dims"],
                                     order="NCHW"
                                     )
            Xs = [
                np.random.rand(2, d1, input_size,
                               input_size).astype(np.float32),
                np.random.rand(2, d2, input_size,
                               input_size).astype(np.float32),
                np.random.rand(2, d3, input_size,
                               input_size).astype(np.float32),
                np.random.rand(2, d4, input_size, input_size).astype(np.float32)
            ]
            for i in range(4):
                res = device_checker.CheckSimple(op, Xs, [0])
                self.assertTrue(res)
                for checker in gradient_checkers:
                    res, grad, grad_estimated = checker.CheckSimple(op, Xs, i,
                                                                    [0])
                    self.assertTrue(res)


class TestRelu(test_util.TestCase):

    def setUp(self):
        self.test_configs = [
            # input size
            (0, 1),
            (1, 1),
            (2, 1),
            (1, 3, 3, 1),
            (2, 3, 3, 1),
            (1, 5, 5, 3),
            (2, 5, 5, 3),
        ]

    def testRelu(self):
        for input_size in self.test_configs:
            op = core.CreateOperator("Relu", ["X"], ["Y"])
            X = np.random.rand(*input_size).astype(np.float32)
            # go away from the origin point to avoid kink problems
            X += 0.01 * np.sign(X)
            X[X == 0] = 0.01
            res = device_checker.CheckSimple(op, [X], [0])
            self.assertTrue(res)
            for checker in gradient_checkers:
                res, grad, grad_estimated = checker.CheckSimple(op, [X], 0, [0])
                self.assertTrue(res)


class TestTanh(test_util.TestCase):

    def setUp(self):
        self.test_configs = [
            (0, 1),
            (1, 1),
            (2, 1),
            (1, 2, 3, 4),
        ]

    def testTanh(self):
        for input_size in self.test_configs:
            op = core.CreateOperator("Tanh", ["X"], ["Y"])
            X = np.random.rand(*input_size).astype(np.float32) - 0.5
            res = device_checker.CheckSimple(op, [X], [0])
            self.assertTrue(res)
            for checker in gradient_checkers:
                res, grad, grad_estimated = checker.CheckSimple(op, [X], 0, [0])
                self.assertTrue(res)


class TestExp(test_util.TestCase):

    def setUp(self):
        self.test_configs = [(0, 1), (1, 1), (2, 1), (1, 2, 3, 4), ]

    def testExp(self):
        for input_size in self.test_configs:
            op = core.CreateOperator("Exp", ["X"], ["Y"])
            X = np.random.rand(*input_size).astype(np.float32) - 0.5
            res = device_checker.CheckSimple(op, [X], [0])
            self.assertTrue(res)
            for checker in gradient_checkers:
                res, grad, grad_estimated = checker.CheckSimple(op, [X], 0, [0])
                self.assertTrue(res)


class TestSigmoid(test_util.TestCase):

    def setUp(self):
        self.test_configs = [(0, 1), (1, 1), (2, 1), (1, 2, 3, 4), ]

    def testSigmoid(self):
        for input_size in self.test_configs:
            op = core.CreateOperator("Sigmoid", ["X"], ["Y"])
            X = np.random.rand(*input_size).astype(np.float32) - 0.5
            res = device_checker.CheckSimple(op, [X], [0])
            self.assertTrue(res)
            for checker in gradient_checkers:
                res, grad, grad_estimated = checker.CheckSimple(op, [X], 0, [0])
                self.assertTrue(res)


class TestSum(test_util.TestCase):

    def setUp(self):
        self.test_configs = [
            ((0, 1), False),
            ((1, 2, 3, 4), True),
            ((1, 2, 3, 4), False)]

    def testSum(self):
        for (input_size, in_place) in self.test_configs:
            op = core.CreateOperator("Sum", ["X1", "X2"],
                                     ["Y" if not in_place else "X1"])
            X1 = np.random.rand(*input_size).astype(np.float32) - 0.5
            X2 = np.random.rand(*input_size).astype(np.float32) - 0.5
            res = device_checker.CheckSimple(op, [X1, X2], [0])
            self.assertTrue(res)
            for checker in gradient_checkers:
                res, grad, grad_estimated = checker.CheckSimple(
                    op, [X1, X2], 0, [0])
                self.assertTrue(res)


class TestMakeTwoClass(test_util.TestCase):

    def setUp(self):
        self.test_configs = [
            # input size
            (0, 1),
            (1,),
            (7,),
            (1, 3),
            (2, 5),
        ]

    def testMakeTwoClass(self):
        for input_size in self.test_configs:
            op = core.CreateOperator("MakeTwoClass", ["X"], ["Y"])
            X = np.random.rand(*input_size).astype(np.float32)
            # step a little to avoid gradient problems
            X[X < 0.01] += 0.01
            X[X > 0.99] -= 0.01
            res = device_checker.CheckSimple(op, [X], [0])
            self.assertTrue(res)
            for checker in gradient_checkers:
                res, grad, grad_estimated = checker.CheckSimple(op, [X], 0, [0])
                self.assertTrue(res)


if __name__ == '__main__':
    workspace.GlobalInit(["python"])
    unittest.main()
