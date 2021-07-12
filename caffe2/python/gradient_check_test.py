# TODO(jiayq): as more and more tests are moving to hypothesis test, we
# can gradually remove this test script. DO NOT ADD MORE TESTS TO THIS
# FILE.





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
from caffe2.python.net_builder import ops, NetBuilder
from caffe2.proto import caffe2_pb2

import unittest
from typing import Optional


if workspace.has_gpu_support and workspace.NumGpuDevices() > 0:
    _gpu_dev_option = caffe2_pb2.DeviceOption()
    _gpu_dev_option.device_type = workspace.GpuDeviceType
    cpu_device_option = caffe2_pb2.DeviceOption()
    gpu_device_checker = device_checker.DeviceChecker(
        0.01, [_gpu_dev_option]
    )
    device_checker = device_checker.DeviceChecker(
        0.01, [_gpu_dev_option, cpu_device_option]
    )
    gpu_gradient_checkers = [
        gradient_checker.GradientChecker(
            0.005, 0.05, _gpu_dev_option, "gpu_checker_ws"
        ),
    ]
    gradient_checkers = [
        gradient_checker.GradientChecker(
            0.005, 0.05, _gpu_dev_option, "gpu_checker_ws"
        ),
        gradient_checker.GradientChecker(
            0.01, 0.05, cpu_device_option, "cpu_checker_ws"
        ),
    ]
    gpu_device_option: Optional[caffe2_pb2.DeviceOption] = _gpu_dev_option
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


class TestAbs(test_util.TestCase):

    def setUp(self):
        self.test_configs = [
            (1, 1),
            (2, 3),
            (2, 3, 4),
            (2, 3, 4, 5),
        ]

    def testAbs(self):
        for input_size in self.test_configs:
            op = core.CreateOperator("Abs", ["X"], ["Y"])
            X = np.random.rand(*input_size).astype(np.float32)
            # go away from the origin point to avoid kink problems
            X += 0.01 * np.sign(X)
            X[X == 0] = 0.01
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

class TestCos(test_util.TestCase):

    def setUp(self):
        self.test_configs = [
            (1, 1),
            (2, 3),
            (2, 3, 4),
            (2, 3, 4, 5),
        ]

    def testCos(self):
        for input_size in self.test_configs:
            op = core.CreateOperator("Cos", ["X"], ["Y"])
            X = np.random.rand(*input_size).astype(np.float32) - 0.5
            res = device_checker.CheckSimple(op, [X], [0])
            self.assertTrue(res)
            for checker in gradient_checkers:
                res, grad, grad_estimated = checker.CheckSimple(op, [X], 0, [0])
                self.assertTrue(res)

class TestSin(test_util.TestCase):

    def setUp(self):
        self.test_configs = [
            (1, 1),
            (2, 3),
            (2, 3, 4),
            (2, 3, 4, 5),
        ]

    def testSin(self):
        for input_size in self.test_configs:
            op = core.CreateOperator("Sin", ["X"], ["Y"])
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
                res, grad, grad_estimated = checker.CheckSimple(
                    op, [X1, X2], 1, [0])
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


class TestIf(test_util.TestCase):
    def testIf(self):
        W_a_values = [2.0, 1.5]
        B_a_values = [0.5]
        W_b_values = [7.0, 3.5]
        B_b_values = [1.5]

        with NetBuilder(_use_control_ops=True) as init_nb:
            W_a = ops.UniformFill([], "W_a", shape=[1, 2], min=-1., max=1.)
            B_a = ops.ConstantFill([], "B_a", shape=[1], value=0.0)
            W_b = ops.UniformFill([], "W_b", shape=[1, 2], min=-1., max=1.)
            B_b = ops.ConstantFill([], "B_b", shape=[1], value=0.0)

            W_gt_a = ops.GivenTensorFill(
                [], "W_gt_a", shape=[1, 2], values=W_a_values)
            B_gt_a = ops.GivenTensorFill([], "B_gt_a", shape=[1], values=B_a_values)
            W_gt_b = ops.GivenTensorFill(
                [], "W_gt_b", shape=[1, 2], values=W_b_values)
            B_gt_b = ops.GivenTensorFill([], "B_gt_b", shape=[1], values=B_b_values)

        params = [W_gt_a, B_gt_a, W_a, B_a, W_gt_b, B_gt_b, W_b, B_b]

        with NetBuilder(_use_control_ops=True, initial_scope=params) as train_nb:
            Y_pred = ops.ConstantFill([], "Y_pred", shape=[1], value=0.0)
            Y_noise = ops.ConstantFill([], "Y_noise", shape=[1], value=0.0)

            switch = ops.UniformFill(
                [], "switch", shape=[1], min=-1., max=1., run_once=0)
            zero = ops.ConstantFill([], "zero", shape=[1], value=0.0)
            X = ops.GaussianFill(
                [], "X", shape=[4096, 2], mean=0.0, std=1.0, run_once=0)
            noise = ops.GaussianFill(
                [], "noise", shape=[4096, 1], mean=0.0, std=1.0, run_once=0)

            with ops.IfNet(ops.LT([switch, zero])):
                Y_gt = ops.FC([X, W_gt_a, B_gt_a], "Y_gt")
                ops.Add([Y_gt, noise], Y_noise)
                ops.FC([X, W_a, B_a], Y_pred)
            with ops.Else():
                Y_gt = ops.FC([X, W_gt_b, B_gt_b], "Y_gt")
                ops.Add([Y_gt, noise], Y_noise)
                ops.FC([X, W_b, B_b], Y_pred)

            dist = ops.SquaredL2Distance([Y_noise, Y_pred], "dist")
            loss = dist.AveragedLoss([], ["loss"])

        assert len(init_nb.get()) == 1, "Expected a single init net produced"
        assert len(train_nb.get()) == 1, "Expected a single train net produced"

        train_net = train_nb.get()[0]
        gradient_map = train_net.AddGradientOperators([loss])

        init_net = init_nb.get()[0]
        ITER = init_net.ConstantFill(
            [], "ITER", shape=[1], value=0, dtype=core.DataType.INT64)
        train_net.Iter(ITER, ITER)
        LR = train_net.LearningRate(ITER, "LR", base_lr=-0.1,
                                        policy="step", stepsize=20, gamma=0.9)
        ONE = init_net.ConstantFill([], "ONE", shape=[1], value=1.)
        train_net.WeightedSum([W_a, ONE, gradient_map[W_a], LR], W_a)
        train_net.WeightedSum([B_a, ONE, gradient_map[B_a], LR], B_a)
        train_net.WeightedSum([W_b, ONE, gradient_map[W_b], LR], W_b)
        train_net.WeightedSum([B_b, ONE, gradient_map[B_b], LR], B_b)

        workspace.RunNetOnce(init_net)
        workspace.CreateNet(train_net)
        # print("Before training, W_a is: {}".format(workspace.FetchBlob("W_a")))
        # print("Before training, B_a is: {}".format(workspace.FetchBlob("B_a")))
        # print("Before training, W_b is: {}".format(workspace.FetchBlob("W_b")))
        # print("Before training, B_b is: {}".format(workspace.FetchBlob("B_b")))

        for _epoch in range(1000):
            workspace.RunNet(train_net.Proto().name)

        # print("After training, W_a is: {}".format(workspace.FetchBlob("W_a")))
        # print("After training, B_a is: {}".format(workspace.FetchBlob("B_a")))
        # print("After training, W_b is: {}".format(workspace.FetchBlob("W_b")))
        # print("After training, B_b is: {}".format(workspace.FetchBlob("B_b")))
        # print("Ground truth W_a is: {}".format(workspace.FetchBlob("W_gt_a")))
        # print("Ground truth B_a is: {}".format(workspace.FetchBlob("B_gt_a")))
        # print("Ground truth W_b is: {}".format(workspace.FetchBlob("W_gt_b")))
        # print("Ground truth B_b is: {}".format(workspace.FetchBlob("B_gt_b")))

        values_map = {
            "W_a": W_a_values,
            "B_a": B_a_values,
            "W_b": W_b_values,
            "B_b": B_b_values,
        }

        train_eps = 0.01

        for blob_name, values in values_map.items():
            trained_values = workspace.FetchBlob(blob_name)
            if trained_values.ndim == 2:
                self.assertEqual(trained_values.shape[0], 1)
                trained_values = trained_values[0][:]
            else:
                self.assertEqual(trained_values.ndim, 1)

            self.assertEqual(trained_values.size, len(values))
            for idx in range(len(trained_values)):
                self.assertTrue(abs(trained_values[idx] - values[idx]) < train_eps)


class TestWhile(test_util.TestCase):
    @unittest.skip("Skip flaky test.")
    def testWhile(self):
        with NetBuilder(_use_control_ops=True) as nb:
            ops.Copy(ops.Const(0), "i")
            ops.Copy(ops.Const(1), "one")
            ops.Copy(ops.Const(2), "two")
            ops.Copy(ops.Const(2.0), "x")
            ops.Copy(ops.Const(3.0), "y")
            ops.Copy(ops.Const(2.0), "z")
            # raises x to the power of 4 and y to the power of 2
            # and z to the power of 3
            with ops.WhileNet():
                with ops.Condition():
                    ops.Add(["i", "one"], "i")
                    ops.LE(["i", "two"])
                ops.Pow("x", "x", exponent=2.0)
                with ops.IfNet(ops.LT(["i", "two"])):
                    ops.Pow("y", "y", exponent=2.0)
                with ops.Else():
                    ops.Pow("z", "z", exponent=3.0)

            ops.Add(["x", "y"], "x_plus_y")
            ops.Add(["x_plus_y", "z"], "s")

        assert len(nb.get()) == 1, "Expected a single net produced"
        net = nb.get()[0]

        net.AddGradientOperators(["s"])
        workspace.RunNetOnce(net)
        # (x^4)' = 4x^3
        self.assertAlmostEqual(workspace.FetchBlob("x_grad"), 32)
        self.assertAlmostEqual(workspace.FetchBlob("x"), 16)
        # (y^2)' = 2y
        self.assertAlmostEqual(workspace.FetchBlob("y_grad"), 6)
        self.assertAlmostEqual(workspace.FetchBlob("y"), 9)
        # (z^3)' = 3z^2
        self.assertAlmostEqual(workspace.FetchBlob("z_grad"), 12)
        self.assertAlmostEqual(workspace.FetchBlob("z"), 8)


if __name__ == '__main__':
    workspace.GlobalInit(["python"])
    unittest.main()
