from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from hypothesis import given, settings, unlimited
import hypothesis.strategies as st
import numpy as np
import unittest
from caffe2.python import brew, core, workspace
import caffe2.python.hypothesis_test_util as hu
from caffe2.python.model_helper import ModelHelper
import caffe2.python.ideep_test_util as mu


@unittest.skipIf(not workspace.C.use_ideep, "No IDEEP support.")
class TestSpatialBN(hu.HypothesisTestCase):
    @given(size=st.integers(7, 10),
           input_channels=st.integers(7, 10),
           batch_size=st.integers(1, 3),
           seed=st.integers(0, 65535),
           order=st.sampled_from(["NCHW"]),
           epsilon=st.floats(min_value=1e-5, max_value=1e-2),
           inplace=st.sampled_from([True, False]),
           **mu.gcs)
    @settings(deadline=None, timeout=unlimited)
    def test_spatialbn_test_mode(
            self, size, input_channels, batch_size, seed, order, epsilon,
            inplace, gc, dc):
        op = core.CreateOperator(
            "SpatialBN",
            ["X", "scale", "bias", "mean", "var"],
            ["X" if inplace else "Y"],
            order=order,
            is_test=True,
            epsilon=epsilon
        )

        def reference_spatialbn_test(X, scale, bias, mean, var):
            if order == "NCHW":
                scale = scale[np.newaxis, :, np.newaxis, np.newaxis]
                bias = bias[np.newaxis, :, np.newaxis, np.newaxis]
                mean = mean[np.newaxis, :, np.newaxis, np.newaxis]
                var = var[np.newaxis, :, np.newaxis, np.newaxis]
            return ((X - mean) / np.sqrt(var + epsilon) * scale + bias,)

        np.random.seed(1701)
        scale = np.random.rand(input_channels).astype(np.float32) + 0.5
        bias = np.random.rand(input_channels).astype(np.float32) - 0.5
        mean = np.random.randn(input_channels).astype(np.float32)
        var = np.random.rand(input_channels).astype(np.float32) + 0.5
        X = np.random.rand(
            batch_size, input_channels, size, size).astype(np.float32) - 0.5

        if order == "NHWC":
            X = X.swapaxes(1, 2).swapaxes(2, 3)

        self.assertDeviceChecks(dc, op, [X, scale, bias, mean, var], [0])

    @given(size=st.integers(7, 10),
           input_channels=st.integers(7, 10),
           batch_size=st.integers(1, 3),
           seed=st.integers(0, 65535),
           order=st.sampled_from(["NCHW"]),
           epsilon=st.floats(1e-5, 1e-2),
           inplace=st.sampled_from([True, False]),
           **mu.gcs)
    @settings(deadline=None)
    def test_spatialbn_train_mode(
            self, size, input_channels, batch_size, seed, order, epsilon,
            inplace, gc, dc):
        op0 = core.CreateOperator(
             "SpatialBN",
            ["X0", "scale0", "bias0", "running_mean0", "running_var0"],
            ["X0" if inplace else "Y0",
            "running_mean0", "running_var0", "saved_mean0", "saved_var0"],
            order=order,
            is_test=False,
            epsilon=epsilon,
            device_option=dc[0]
        )
        op1 = core.CreateOperator(
            "SpatialBN",
            ["X1", "scale1", "bias1", "running_mean1", "running_var1"],
            ["X1" if inplace else "Y1",
             "running_mean1", "running_var1", "saved_mean1", "saved_var1"],
            order=order,
            is_test=False,
            epsilon=epsilon,
            device_option=dc[1]
        )
        np.random.seed(1701)
        scale = np.random.rand(input_channels).astype(np.float32) + 0.5
        bias = np.random.rand(input_channels).astype(np.float32) - 0.5
        mean = np.random.randn(input_channels).astype(np.float32)
        var = np.random.rand(input_channels).astype(np.float32) + 0.5
        X = np.random.rand(
            batch_size, input_channels, size, size).astype(np.float32) - 0.5

        if order == "NHWC":
            X = X.swapaxes(1, 2).swapaxes(2, 3)

        old_ws_name = workspace.CurrentWorkspace()
        workspace.SwitchWorkspace("_device_check_", True)
        workspace.FeedBlob('X0', X, dc[0])
        workspace.FeedBlob('scale0', scale, dc[0])
        workspace.FeedBlob('bias0', bias, dc[0])
        workspace.FeedBlob('running_mean0', mean, dc[0])
        workspace.FeedBlob('running_var0', var, dc[0])
        workspace.RunOperatorOnce(op0)
        Y0 = workspace.FetchBlob('X0' if inplace else 'Y0')
        running_mean0 = workspace.FetchBlob('running_mean0')
        running_var0 = workspace.FetchBlob('running_var0')
        saved_mean0 = workspace.FetchBlob('saved_mean0')
        saved_var0 = workspace.FetchBlob('saved_var0')

        workspace.ResetWorkspace()
        workspace.FeedBlob('X1', X, dc[1])
        workspace.FeedBlob('scale1', scale, dc[1])
        workspace.FeedBlob('bias1', bias, dc[1])
        workspace.FeedBlob('running_mean1', mean, dc[1])
        workspace.FeedBlob('running_var1', var, dc[1])
        workspace.RunOperatorOnce(op1)
        Y1 = workspace.FetchBlob('X1' if inplace else 'Y1')
        running_mean1 = workspace.FetchBlob('running_mean1')
        running_var1 = workspace.FetchBlob('running_var1')
        saved_mean1 = workspace.FetchBlob('saved_mean1')
        saved_var1 = workspace.FetchBlob('saved_var1')

        if not np.allclose(Y0, Y1, atol=0.01, rtol=0.01):
            print("Failure in checking device option 1 and output. The outputs are:")
            print(Y1.flatten())
            print(Y0.flatten())
            print(np.max(np.abs(Y1 - Y0)))
            self.assertTrue(False)

        if not np.allclose(running_mean0, running_mean1, atol=0.01, rtol=0.01):
            print("Failure in checking device option 1 and output running mean. The outputs are:")
            print(running_mean1.flatten())
            print(running_mean0.flatten())
            print(np.max(np.abs(running_mean1 - running_mean0)))
            self.assertTrue(False)

        if not np.allclose(running_var0, running_var1, atol=0.01, rtol=0.01):
            print("Failure in checking device option 1 and outpu running var. The outputs are:")
            print(running_var1.flatten())
            print(running_var0.flatten())
            print(np.max(np.abs(running_var1 - running_var0)))
            self.assertTrue(False)

        if not np.allclose(saved_mean0, saved_mean1, atol=0.01, rtol=0.01):
            print("Failure in checking device option 1 and outpu mean. The outputs are:")
            print(saved_mean1.flatten())
            print(saved_mean0.flatten())
            print(np.max(np.abs(saved_mean1 - saved_mean0)))
            self.assertTrue(False)

        saved_var0 = np.square(1 / saved_var0)
        if not np.allclose(saved_var0, saved_var1, atol=0.01, rtol=0.01):
            print("Failure in checking device option 1 and outpu var. The outputs are:")
            print(saved_var1.flatten())
            print(saved_var0.flatten())
            print(np.max(np.abs(saved_var1 - saved_var0)))
            self.assertTrue(False)

        workspace.SwitchWorkspace(old_ws_name)

if __name__ == "__main__":
    unittest.main()
