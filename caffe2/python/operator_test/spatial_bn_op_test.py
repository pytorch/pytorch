from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.proto import caffe2_pb2
from caffe2.python import brew, core, workspace
import caffe2.python.hip_test_util as hiputl
import caffe2.python.hypothesis_test_util as hu
from caffe2.python.model_helper import ModelHelper
import caffe2.python.serialized_test.serialized_test_util as serial

from hypothesis import given, assume
import hypothesis.strategies as st
import numpy as np
import unittest


class TestSpatialBN(serial.SerializedTestCase):

    @serial.given(size=st.integers(7, 10),
           input_channels=st.integers(1, 10),
           batch_size=st.integers(0, 3),
           seed=st.integers(0, 65535),
           order=st.sampled_from(["NCHW", "NHWC"]),
           epsilon=st.floats(min_value=1e-5, max_value=1e-2),
           inplace=st.booleans(),
           engine=st.sampled_from(["", "CUDNN"]),
           **hu.gcs)
    def test_spatialbn_test_mode_3d(
            self, size, input_channels, batch_size, seed, order, epsilon,
            inplace, engine, gc, dc):
        # Currently MIOPEN SpatialBN only supports 2D
        if hiputl.run_in_hip(gc, dc):
            assume(engine != "CUDNN")
        op = core.CreateOperator(
            "SpatialBN",
            ["X", "scale", "bias", "mean", "var"],
            ["X" if inplace else "Y"],
            order=order,
            is_test=True,
            epsilon=epsilon,
            engine=engine,
        )

        def reference_spatialbn_test(X, scale, bias, mean, var):
            if order == "NCHW":
                scale = scale[np.newaxis, :,
                              np.newaxis, np.newaxis, np.newaxis]
                bias = bias[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
                mean = mean[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
                var = var[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]

            return ((X - mean) / np.sqrt(var + epsilon) * scale + bias,)

        np.random.seed(1701)
        scale = np.random.rand(input_channels).astype(np.float32) + 0.5
        bias = np.random.rand(input_channels).astype(np.float32) - 0.5
        mean = np.random.randn(input_channels).astype(np.float32)
        var = np.random.rand(input_channels).astype(np.float32) + 0.5
        X = np.random.rand(batch_size, input_channels, size, size, size)\
            .astype(np.float32) - 0.5

        if order == "NHWC":
            X = X.transpose(0, 2, 3, 4, 1)
        self.assertReferenceChecks(gc, op, [X, scale, bias, mean, var],
                                   reference_spatialbn_test)
        self.assertDeviceChecks(dc, op, [X, scale, bias, mean, var], [0])

    @unittest.skipIf((not workspace.has_gpu_support) and (
        not workspace.has_hip_support), "No gpu support")
    @given(size=st.integers(7, 10),
           input_channels=st.integers(1, 10),
           batch_size=st.integers(0, 3),
           seed=st.integers(0, 65535),
           order=st.sampled_from(["NCHW", "NHWC"]),
           epsilon=st.floats(min_value=1e-5, max_value=1e-2),
           inplace=st.booleans(),
           engine=st.sampled_from(["", "CUDNN"]),
           **hu.gcs)
    def test_spatialbn_test_mode_1d(
            self, size, input_channels, batch_size, seed, order, epsilon,
            inplace, engine, gc, dc):
        # Currently MIOPEN SpatialBN only supports 2D
        if hiputl.run_in_hip(gc, dc):
            assume(engine != "CUDNN")
        op = core.CreateOperator(
            "SpatialBN",
            ["X", "scale", "bias", "mean", "var"],
            ["X" if inplace else "Y"],
            order=order,
            is_test=True,
            epsilon=epsilon,
            engine=engine,
        )

        def reference_spatialbn_test(X, scale, bias, mean, var):
            if order == "NCHW":
                scale = scale[np.newaxis, :, np.newaxis]
                bias = bias[np.newaxis, :, np.newaxis]
                mean = mean[np.newaxis, :, np.newaxis]
                var = var[np.newaxis, :, np.newaxis]
            return ((X - mean) / np.sqrt(var + epsilon) * scale + bias,)

        np.random.seed(1701)
        scale = np.random.rand(input_channels).astype(np.float32) + 0.5
        bias = np.random.rand(input_channels).astype(np.float32) - 0.5
        mean = np.random.randn(input_channels).astype(np.float32)
        var = np.random.rand(input_channels).astype(np.float32) + 0.5
        X = np.random.rand(
            batch_size, input_channels, size).astype(np.float32) - 0.5

        if order == "NHWC":
            X = X.swapaxes(1, 2)
        self.assertReferenceChecks(gc, op, [X, scale, bias, mean, var],
                                   reference_spatialbn_test)
        self.assertDeviceChecks(dc, op, [X, scale, bias, mean, var], [0])

    @given(size=st.integers(7, 10),
           input_channels=st.integers(1, 10),
           batch_size=st.integers(0, 3),
           seed=st.integers(0, 65535),
           order=st.sampled_from(["NCHW", "NHWC"]),
           epsilon=st.floats(min_value=1e-5, max_value=1e-2),
           engine=st.sampled_from(["", "CUDNN"]),
           inplace=st.booleans(),
           **hu.gcs)
    def test_spatialbn_test_mode(
            self, size, input_channels, batch_size, seed, order, epsilon,
            inplace, engine, gc, dc):
        # Currently HIP SpatialBN only supports NCHW
        if hiputl.run_in_hip(gc, dc):
            assume(order == "NCHW")

        op = core.CreateOperator(
            "SpatialBN",
            ["X", "scale", "bias", "mean", "var"],
            ["X" if inplace else "Y"],
            order=order,
            is_test=True,
            epsilon=epsilon,
            engine=engine
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

        self.assertReferenceChecks(gc, op, [X, scale, bias, mean, var],
                                   reference_spatialbn_test)
        self.assertDeviceChecks(dc, op, [X, scale, bias, mean, var], [0])

    @given(size=st.integers(7, 10),
           input_channels=st.integers(1, 10),
           batch_size=st.integers(0, 3),
           seed=st.integers(0, 65535),
           order=st.sampled_from(["NCHW", "NHWC"]),
           epsilon=st.floats(1e-5, 1e-2),
           momentum=st.floats(0.5, 0.9),
           engine=st.sampled_from(["", "CUDNN"]),
           inplace=st.sampled_from([True, False]),
           **hu.gcs)
    def test_spatialbn_train_mode(
            self, size, input_channels, batch_size, seed, order, epsilon,
            momentum, inplace, engine, gc, dc):
        # Currently HIP SpatialBN only supports NCHW
        if hiputl.run_in_hip(gc, dc):
            assume(order == "NCHW")

        op = core.CreateOperator(
            "SpatialBN",
            ["X", "scale", "bias", "running_mean", "running_var"],
            ["X" if inplace else "Y",
             "running_mean", "running_var", "saved_mean", "saved_var"],
            order=order,
            is_test=False,
            epsilon=epsilon,
            momentum=momentum,
            engine=engine,
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

        self.assertDeviceChecks(dc, op, [X, scale, bias, mean, var],
                                [0, 1, 2, 3, 4])

    @given(size=st.integers(7, 10),
           input_channels=st.integers(1, 10),
           batch_size=st.integers(0, 3),
           seed=st.integers(0, 65535),
           order=st.sampled_from(["NCHW", "NHWC"]),
           epsilon=st.floats(min_value=1e-5, max_value=1e-2),
           momentum=st.floats(0.5, 0.9),
           engine=st.sampled_from(["", "CUDNN"]),
           **hu.gcs)
    def test_spatialbn_train_mode_gradient_check(
            self, size, input_channels, batch_size, seed, order, epsilon,
            momentum, engine, gc, dc):
        # Currently HIP SpatialBN only supports NCHW
        if hiputl.run_in_hip(gc, dc):
            assume(order == "NCHW")

        op = core.CreateOperator(
            "SpatialBN",
            ["X", "scale", "bias", "mean", "var"],
            ["Y", "mean", "var", "saved_mean", "saved_var"],
            order=order,
            is_test=False,
            epsilon=epsilon,
            momentum=momentum,
            engine=engine
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

    @given(size=st.integers(7, 10),
           input_channels=st.integers(1, 10),
           batch_size=st.integers(0, 3),
           seed=st.integers(0, 65535),
           order=st.sampled_from(["NCHW", "NHWC"]),
           epsilon=st.floats(min_value=1e-5, max_value=1e-2),
           momentum=st.floats(min_value=0.5, max_value=0.9),
           engine=st.sampled_from(["", "CUDNN"]),
           **hu.gcs)
    def test_spatialbn_train_mode_gradient_check_1d(
            self, size, input_channels, batch_size, seed, order, epsilon,
            momentum, engine, gc, dc):
        # Currently MIOPEN SpatialBN only supports 2D
        if hiputl.run_in_hip(gc, dc):
            assume(engine != "CUDNN")
        op = core.CreateOperator(
            "SpatialBN",
            ["X", "scale", "bias", "mean", "var"],
            ["Y", "mean", "var", "saved_mean", "saved_var"],
            order=order,
            is_test=False,
            epsilon=epsilon,
            momentum=momentum,
            engine=engine,
        )
        np.random.seed(seed)
        scale = np.random.rand(input_channels).astype(np.float32) + 0.5
        bias = np.random.rand(input_channels).astype(np.float32) - 0.5
        mean = np.random.randn(input_channels).astype(np.float32)
        var = np.random.rand(input_channels).astype(np.float32) + 0.5
        X = np.random.rand(
            batch_size, input_channels, size).astype(np.float32) - 0.5
        if order == "NHWC":
            X = X.swapaxes(1, 2)

        for input_to_check in [0, 1, 2]:  # dX, dScale, dBias
            self.assertGradientChecks(gc, op, [X, scale, bias, mean, var],
                                      input_to_check, [0], stepsize=0.01)

    @given(N=st.integers(0, 5),
           C=st.integers(1, 10),
           H=st.integers(1, 5),
           W=st.integers(1, 5),
           epsilon=st.floats(1e-5, 1e-2),
           momentum=st.floats(0.5, 0.9),
           order=st.sampled_from(["NCHW", "NHWC"]),
           num_batches=st.integers(2, 5),
           in_place=st.booleans(),
           engine=st.sampled_from(["", "CUDNN"]),
           **hu.gcs)
    def test_spatial_bn_multi_batch(
            self, N, C, H, W, epsilon, momentum, order, num_batches, in_place,
            engine, gc, dc):
        if in_place:
            outputs = ["Y", "mean", "var", "batch_mean", "batch_var"]
        else:
            outputs = ["Y", "mean", "var", "saved_mean", "saved_var"]
        op = core.CreateOperator(
            "SpatialBN",
            ["X", "scale", "bias", "mean", "var", "batch_mean", "batch_var"],
            outputs,
            order=order,
            is_test=False,
            epsilon=epsilon,
            momentum=momentum,
            num_batches=num_batches,
            engine=engine,
        )
        if order == "NCHW":
            X = np.random.randn(N, C, H, W).astype(np.float32)
        else:
            X = np.random.randn(N, H, W, C).astype(np.float32)
        scale = np.random.randn(C).astype(np.float32)
        bias = np.random.randn(C).astype(np.float32)
        mean = np.random.randn(C).astype(np.float32)
        var = np.random.rand(C).astype(np.float32)
        batch_mean = np.random.rand(C).astype(np.float32) - 0.5
        batch_var = np.random.rand(C).astype(np.float32) + 1.0
        inputs = [X, scale, bias, mean, var, batch_mean, batch_var]

        def spatial_bn_multi_batch_ref(
                X, scale, bias, mean, var, batch_mean, batch_var):
            if N == 0:
                batch_mean = np.zeros(C).astype(np.float32)
                batch_var = np.zeros(C).astype(np.float32)
            else:
                size = num_batches * N * H * W
                batch_mean /= size
                batch_var = batch_var / size - np.square(batch_mean)
                mean = momentum * mean + (1.0 - momentum) * batch_mean
                var = momentum * var + (1.0 - momentum) * batch_var
                batch_var = 1.0 / np.sqrt(batch_var + epsilon)
            if order == "NCHW":
                scale = np.reshape(scale, (C, 1, 1))
                bias = np.reshape(bias, (C, 1, 1))
                batch_mean = np.reshape(batch_mean, (C, 1, 1))
                batch_var = np.reshape(batch_var, (C, 1, 1))
            Y = (X - batch_mean) * batch_var * scale + bias
            if order == "NCHW":
                batch_mean = np.reshape(batch_mean, (C))
                batch_var = np.reshape(batch_var, (C))
            return (Y, mean, var, batch_mean, batch_var)

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=inputs,
            reference=spatial_bn_multi_batch_ref,
        )
        self.assertDeviceChecks(dc, op, inputs, [0, 1, 2, 3, 4])

    @given(N=st.integers(0, 5),
           C=st.integers(1, 10),
           H=st.integers(1, 5),
           W=st.integers(1, 5),
           epsilon=st.floats(1e-5, 1e-2),
           order=st.sampled_from(["NCHW", "NHWC"]),
           num_batches=st.integers(2, 5),
           in_place=st.booleans(),
           engine=st.sampled_from(["", "CUDNN"]),
           **hu.gcs)
    def test_spatial_bn_multi_batch_grad(
            self, N, C, H, W, epsilon, order, num_batches, in_place, engine,
            gc, dc):
        if in_place:
            outputs = ["dX", "dscale_sum", "dbias_sum"]
        else:
            outputs = ["dX", "dscale", "dbias"]
        op = core.CreateOperator(
            "SpatialBNGradient",
            ["X", "scale", "dY", "mean", "rstd", "dscale_sum", "dbias_sum"],
            outputs,
            order=order,
            epsilon=epsilon,
            num_batches=num_batches,
            engine=engine,
        )
        if order == "NCHW":
            dY = np.random.randn(N, C, H, W).astype(np.float32)
            X = np.random.randn(N, C, H, W).astype(np.float32)
        else:
            dY = np.random.randn(N, H, W, C).astype(np.float32)
            X = np.random.randn(N, H, W, C).astype(np.float32)
        scale = np.random.randn(C).astype(np.float32)
        mean = np.random.randn(C).astype(np.float32)
        rstd = np.random.rand(C).astype(np.float32)
        dscale_sum = np.random.randn(C).astype(np.float32)
        dbias_sum = np.random.randn(C).astype(np.float32)
        inputs = [X, scale, dY, mean, rstd, dscale_sum, dbias_sum]

        def spatial_bn_multi_batch_grad_ref(
                X, scale, dY, mean, rstd, dscale_sum, dbias_sum):
            if N == 0:
                dscale = np.zeros(C).astype(np.float32)
                dbias = np.zeros(C).astype(np.float32)
                alpha = np.zeros(C).astype(np.float32)
                beta = np.zeros(C).astype(np.float32)
                gamma = np.zeros(C).astype(np.float32)
            else:
                dscale = dscale_sum / num_batches
                dbias = dbias_sum / num_batches
                alpha = scale * rstd
                beta = -alpha * dscale * rstd / (N * H * W)
                gamma = alpha * (mean * dscale * rstd - dbias) / (N * H * W)
            if order == "NCHW":
                alpha = np.reshape(alpha, (C, 1, 1))
                beta = np.reshape(beta, (C, 1, 1))
                gamma = np.reshape(gamma, (C, 1, 1))
            dX = alpha * dY + beta * X + gamma
            return (dX, dscale, dbias)

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=inputs,
            reference=spatial_bn_multi_batch_grad_ref,
        )
        self.assertDeviceChecks(dc, op, inputs, [0, 1, 2])

    @given(size=st.integers(7, 10),
           input_channels=st.integers(1, 10),
           batch_size=st.integers(0, 3),
           seed=st.integers(0, 65535),
           epsilon=st.floats(1e-5, 1e-2),
           engine=st.sampled_from(["", "CUDNN"]),
           **hu.gcs)
    def test_spatialbn_brew_wrapper(
            self, size, input_channels, batch_size, seed, epsilon,
            engine, gc, dc):
        np.random.seed(seed)
        X = np.random.rand(
            batch_size, input_channels, size, size).astype(np.float32)

        workspace.FeedBlob('X', X)

        model = ModelHelper(name='test_spatialbn_brew_wrapper')

        brew.spatial_bn(
            model,
            'X',
            'Y',
            input_channels,
            epsilon=epsilon,
            is_test=False,
        )

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)


if __name__ == "__main__":
    unittest.main()
