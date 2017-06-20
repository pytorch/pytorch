# TODO(jiayq): as more and more tests are moving to hypothesis test, we
# can gradually remove this test script. DO NOT ADD MORE TESTS TO THIS
# FILE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
from caffe2.python import (
    brew,
    core,
    device_checker,
    gradient_checker,
    model_helper,
    test_util,
    workspace,
)
from caffe2.python.gradient_checker import NetGradientChecker
from caffe2.proto import caffe2_pb2

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
            # (0, 1),
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
            # (0, 1),
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
        self.test_configs = [
            # (0, 1),
            (1, 1),
            (2, 1),
            (1, 2, 3, 4),
        ]

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
        self.test_configs = [
            # (0, 1),
            (1, 1),
            (2, 1),
            (1, 2, 3, 4),
        ]

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
            # ((0, 1), False),
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
            # (0, 1),
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


class TestNetGradientChecker(test_util.TestCase):
    def test_net_gradient_checker(self):
        model = model_helper.ModelHelper(name="test")
        const = model.net.AddExternalInputs("const1", "const2")
        fc = brew.fc(model, dim_in=3, dim_out=4, blob_in="X", blob_out="Y", axis=0)
        dist = [model.net.SquaredL2Distance([fc, c]) for c in const]
        losses = [model.net.AveragedLoss(d) for d in dist]  # using two losses here

        workspace.RunNetOnce(model.param_init_net)
        NetGradientChecker.Check(
            model.net,
            outputs_with_grad=losses,
            input_values={"X": np.array([1, 2, 3], dtype="float32"),
                          const[0]: np.array([1, 1, 1, 1], dtype="float32"),
                          const[1]: np.array([2, 2, 2, 2], dtype="float32")},
            input_to_check="X",
        )

    def test_net_comparison(self):
        # (a + b) * (c + d) == a * c + a * d + b * c + b * d
        net1 = core.Net("net1")
        a, b, c, d = net1.AddExternalInputs("a", "b", "c", "d")
        a_b = net1.Sum([a, b], "a+b")
        c_d = net1.Sum([c, d], "c+d")
        x = net1.Mul([a_b, c_d], "x")

        net2 = core.Net("net2")
        ac = net2.Mul([a, c], "ac")
        ad = net2.Mul([a, d], "ad")
        bc = net2.Mul([b, c], "bc")
        bd = net2.Mul([b, d], "bd")
        y = net2.Sum([ac, ad, bc, bd], "y")

        input_values = {blob: np.array([i], dtype=np.float32)
                        for i, blob in enumerate([a, b, c, d])}

        NetGradientChecker.CompareNets(
            [net1, net2], [[x], [y]], [0],
            inputs_with_grads=[a, b, c, d],
            input_values=input_values,
        )

if __name__ == '__main__':
    workspace.GlobalInit(["python"])
    unittest.main()
