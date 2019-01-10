from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from hypothesis import given, settings
import hypothesis.strategies as st
import numpy as np
import unittest
from caffe2.python import brew, core, workspace
import caffe2.python.hypothesis_test_util as hu
from caffe2.python.model_helper import ModelHelper
import caffe2.python.ideep_test_util as mu


@unittest.skipIf(not workspace.C.use_mkldnn, "No MKLDNN support.")
class TestSpatialBN(hu.HypothesisTestCase):
    @given(size=st.integers(7, 10),
           input_channels=st.integers(7, 10),
           batch_size=st.integers(1, 3),
           seed=st.integers(0, 65535),
           order=st.sampled_from(["NCHW"]),
           epsilon=st.floats(min_value=1e-5, max_value=1e-2),
           inplace=st.sampled_from([True, False]),
           **mu.gcs)
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
    def test_spatialbn_train_mode(
            self, size, input_channels, batch_size, seed, order, epsilon,
            inplace, gc, dc):
        print("dc0: {}, dc1: {}".format(dc[0], dc[1]))
        op = core.CreateOperator(
            "SpatialBN",
            ["X", "scale", "bias", "running_mean", "running_var"],
            ["X" if inplace else "Y",
            "running_mean", "running_var", "saved_mean", "saved_var"],
            order=order,
            is_test=False,
            epsilon=epsilon,
        )
        np.random.seed(1701)
        scale = np.random.rand(input_channels).astype(np.float32) + 0.5
        bias = np.random.rand(input_channels).astype(np.float32) - 0.5
        running_mean = np.random.randn(input_channels).astype(np.float32)
        running_var = np.random.rand(input_channels).astype(np.float32) + 0.5
        X = np.random.rand(
            batch_size, input_channels, size, size).astype(np.float32) - 0.5

        if order == "NHWC":
            X = X.swapaxes(1, 2).swapaxes(2, 3)

        # TODO: It looks like IDEEP spatial_bn op outputs save_var (output[4])
        # as the reciprocal of CPU op's output. Need to check back and add
        # output[4] for comparison
        self.assertDeviceChecks(dc, op, [X, scale, bias, running_mean, running_var],
            [0, 1, 2, 3])

    @given(size=st.integers(7, 10),
           input_channels=st.integers(1, 10),
           batch_size=st.integers(1, 3),
           seed=st.integers(0, 65535),
           order=st.sampled_from(["NCHW"]),
           epsilon=st.floats(min_value=1e-5, max_value=1e-2),
           **mu.gcs)
    def test_spatialbn_train_mode_gradient_check(
            self, size, input_channels, batch_size, seed, order, epsilon,
            gc, dc):
        op = core.CreateOperator(
            "SpatialBN",
            ["X", "scale", "bias", "mean", "var"],
            ["Y", "mean", "var", "saved_mean", "saved_var"],
            order=order,
            is_test=False,
            epsilon=epsilon,
        )
        np.random.seed(seed)
        scale = np.random.rand(input_channels).astype(np.float32) + 0.5
        bias = np.random.rand(input_channels).astype(np.float32) - 0.5
        mean = np.random.randn(input_channels).astype(np.float32)
        var = np.random.rand(input_channels).astype(np.float32) + 0.5
        X = np.random.rand(
            batch_size, input_channels, size, size).astype(np.float32) - 0.5
        if order == "NHWC":
            X = X.swapaxes(1, 2).swapaxes(2, 3)

        for input_to_check in [0, 1, 2]:  # dX, dScale, dBias
            self.assertGradientChecks(gc, op, [X, scale, bias, mean, var],
                                      input_to_check, [0])



if __name__ == "__main__":
    unittest.main()
