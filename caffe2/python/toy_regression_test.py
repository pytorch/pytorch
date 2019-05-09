import numpy as np
import unittest

from caffe2.python import core, workspace, test_util


class TestToyRegression(test_util.TestCase):
    def testToyRegression(self):
        """Tests a toy regression end to end.

        The test code carries a simple toy regression in the form
            y = 2.0 x1 + 1.5 x2 + 0.5
        by randomly generating gaussian inputs and calculating the ground
        truth outputs in the net as well. It uses a standard SGD to then
        train the parameters.
        """
        workspace.ResetWorkspace()
        init_net = core.Net("init")
        W = init_net.UniformFill([], "W", shape=[1, 2], min=-1., max=1.)
        B = init_net.ConstantFill([], "B", shape=[1], value=0.0)
        W_gt = init_net.GivenTensorFill(
            [], "W_gt", shape=[1, 2], values=[2.0, 1.5])
        B_gt = init_net.GivenTensorFill([], "B_gt", shape=[1], values=[0.5])
        LR = init_net.ConstantFill([], "LR", shape=[1], value=-0.1)
        ONE = init_net.ConstantFill([], "ONE", shape=[1], value=1.)
        ITER = init_net.ConstantFill([], "ITER", shape=[1], value=0,
                                     dtype=core.DataType.INT64)

        train_net = core.Net("train")
        X = train_net.GaussianFill([], "X", shape=[64, 2], mean=0.0, std=1.0)
        Y_gt = X.FC([W_gt, B_gt], "Y_gt")
        Y_pred = X.FC([W, B], "Y_pred")
        dist = train_net.SquaredL2Distance([Y_gt, Y_pred], "dist")
        loss = dist.AveragedLoss([], ["loss"])
        # Get gradients for all the computations above. Note that in fact we
        # don't need to get the gradient the Y_gt computation, but we'll just
        # leave it there. In many cases, I am expecting one to load X and Y
        # from the disk, so there is really no operator that will calculate the
        # Y_gt input.
        input_to_grad = train_net.AddGradientOperators([loss], skip=2)
        # updates
        train_net.Iter(ITER, ITER)
        train_net.LearningRate(ITER, "LR", base_lr=-0.1,
                               policy="step", stepsize=20, gamma=0.9)
        train_net.WeightedSum([W, ONE, input_to_grad[str(W)], LR], W)
        train_net.WeightedSum([B, ONE, input_to_grad[str(B)], LR], B)
        for blob in [loss, W, B]:
            train_net.Print(blob, [])

        # the CPU part.
        plan = core.Plan("toy_regression")
        plan.AddStep(core.ExecutionStep("init", init_net))
        plan.AddStep(core.ExecutionStep("train", train_net, 200))

        workspace.RunPlan(plan)
        W_result = workspace.FetchBlob("W")
        B_result = workspace.FetchBlob("B")
        np.testing.assert_array_almost_equal(W_result, [[2.0, 1.5]], decimal=2)
        np.testing.assert_array_almost_equal(B_result, [0.5], decimal=2)
        workspace.ResetWorkspace()


if __name__ == '__main__':
    unittest.main()
