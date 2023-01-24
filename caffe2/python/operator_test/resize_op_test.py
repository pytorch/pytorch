



import numpy as np
import hypothesis.strategies as st
import unittest
import caffe2.python.hypothesis_test_util as hu
from caffe2.python import core
from caffe2.proto import caffe2_pb2
from hypothesis import assume, given, settings


class TestResize(hu.HypothesisTestCase):
    @given(height_scale=st.floats(0.25, 4.0) | st.just(2.0),
           width_scale=st.floats(0.25, 4.0) | st.just(2.0),
           height=st.integers(4, 32),
           width=st.integers(4, 32),
           num_channels=st.integers(1, 4),
           batch_size=st.integers(1, 4),
           seed=st.integers(0, 65535),
           order=st.sampled_from(["NCHW", "NHWC"]),
           **hu.gcs)
    @settings(max_examples=10, deadline=None)
    def test_nearest(self, height_scale, width_scale, height, width,
                     num_channels, batch_size, seed, order,
                     gc, dc):

        assume(order == "NCHW" or gc.device_type == caffe2_pb2.CPU)
        # NHWC currently only supported for CPU. Ignore other devices.
        if order == "NHWC":
            dc = [d for d in dc if d.device_type == caffe2_pb2.CPU]

        np.random.seed(seed)
        op = core.CreateOperator(
            "ResizeNearest",
            ["X"],
            ["Y"],
            width_scale=width_scale,
            height_scale=height_scale,
            order=order,
        )

        X = np.random.rand(
            batch_size, num_channels, height, width).astype(np.float32)
        if order == "NHWC":
            X = X.transpose([0, 2, 3, 1])

        def ref(X):
            output_height = np.int32(height * height_scale)
            output_width = np.int32(width * width_scale)

            output_h_idxs, output_w_idxs = np.meshgrid(np.arange(output_height),
                                                       np.arange(output_width),
                                                       indexing='ij')

            input_h_idxs = np.minimum(
                output_h_idxs / height_scale, height - 1).astype(np.int32)
            input_w_idxs = np.minimum(
                output_w_idxs / width_scale, width - 1).astype(np.int32)

            if order == "NCHW":
                Y = X[:, :, input_h_idxs, input_w_idxs]
            else:
                Y = X[:, input_h_idxs, input_w_idxs, :]

            return Y,

        self.assertReferenceChecks(gc, op, [X], ref)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0], stepsize=0.1, threshold=1e-2)

    @given(height_scale=st.floats(0.25, 4.0) | st.just(2.0),
           width_scale=st.floats(0.25, 4.0) | st.just(2.0),
           height=st.integers(4, 32),
           width=st.integers(4, 32),
           num_channels=st.integers(1, 4),
           batch_size=st.integers(1, 4),
           seed=st.integers(0, 65535),
           order=st.sampled_from(["NCHW", "NHWC"]),
           **hu.gcs)
    def test_nearest_grad(self, height_scale, width_scale, height, width,
                          num_channels, batch_size, seed, order, gc, dc):

        assume(order == "NCHW" or gc.device_type == caffe2_pb2.CPU)
        # NHWC currently only supported for CPU. Ignore other devices.
        if order == "NHWC":
            dc = [d for d in dc if d.device_type == caffe2_pb2.CPU]

        np.random.seed(seed)

        output_height = np.int32(height * height_scale)
        output_width = np.int32(width * width_scale)
        X = np.random.rand(batch_size,
                           num_channels,
                           height,
                           width).astype(np.float32)

        dY = np.random.rand(batch_size,
                            num_channels,
                            output_height,
                            output_width).astype(np.float32)
        if order == "NHWC":
            X = X.transpose([0, 2, 3, 1])
            dY = dY.transpose([0, 2, 3, 1])

        op = core.CreateOperator(
            "ResizeNearestGradient",
            ["dY", "X"],
            ["dX"],
            width_scale=width_scale,
            height_scale=height_scale,
            order=order,
        )

        def ref(dY, X):
            dX = np.zeros_like(X)

            for i in range(output_height):
                for j in range(output_width):
                    input_i = np.minimum(i / height_scale, height - 1).astype(np.int32)
                    input_j = np.minimum(j / width_scale, width - 1).astype(np.int32)
                    if order == "NCHW":
                        dX[:, :, input_i, input_j] += dY[:, :, i, j]
                    else:
                        dX[:, input_i, input_j, :] += dY[:, i, j, :]
            return dX,

        self.assertDeviceChecks(dc, op, [dY, X], [0])
        self.assertReferenceChecks(gc, op, [dY, X], ref)

    @given(height_scale=st.floats(0.25, 4.0) | st.just(2.0),
           width_scale=st.floats(0.25, 4.0) | st.just(2.0),
           height=st.integers(4, 8),
           width=st.integers(4, 8),
           num_channels=st.integers(1, 4),
           batch_size=st.integers(1, 4),
           seed=st.integers(0, 65535),
           order=st.sampled_from(["NCHW", "NHWC"]),
           **hu.gcs)
    @settings(deadline=10000)
    def test_nearest_onnx(self, height_scale, width_scale, height, width,
                          num_channels, batch_size, seed, order,
                          gc, dc):

        assume(order == "NCHW" or gc.device_type == caffe2_pb2.CPU)
        # NHWC currently only supported for CPU. Ignore other devices.
        if order == "NHWC":
            dc = [d for d in dc if d.device_type == caffe2_pb2.CPU]

        np.random.seed(seed)
        op = core.CreateOperator(
            "ResizeNearest",
            ["X", "scales"],
            ["Y"],
            order=order,
        )

        X = np.random.rand(
            batch_size, num_channels, height, width).astype(np.float32)
        if order == "NHWC":
            X = X.transpose([0, 2, 3, 1])

        scales = np.array([height_scale, width_scale]).astype(np.float32)

        def ref(X, scales):
            output_height = np.int32(height * scales[0])
            output_width = np.int32(width * scales[1])

            output_h_idxs, output_w_idxs = np.meshgrid(np.arange(output_height),
                                                       np.arange(output_width),
                                                       indexing='ij')

            input_h_idxs = np.minimum(
                output_h_idxs / scales[0], height - 1).astype(np.int32)
            input_w_idxs = np.minimum(
                output_w_idxs / scales[1], width - 1).astype(np.int32)

            if order == "NCHW":
                Y = X[:, :, input_h_idxs, input_w_idxs]
            else:
                Y = X[:, input_h_idxs, input_w_idxs, :]

            return Y,

        self.assertReferenceChecks(gc, op, [X, scales], ref)
        self.assertDeviceChecks(dc, op, [X, scales], [0])
        self.assertGradientChecks(gc, op, [X, scales], 0, [0], stepsize=0.1,
                                  threshold=1e-2)

    @given(height_scale=st.floats(0.25, 4.0) | st.just(2.0),
           width_scale=st.floats(0.25, 4.0) | st.just(2.0),
           height=st.integers(4, 8),
           width=st.integers(4, 8),
           num_channels=st.integers(1, 4),
           batch_size=st.integers(1, 4),
           seed=st.integers(0, 65535),
           order=st.sampled_from(["NCHW", "NHWC"]),
           **hu.gcs)
    def test_nearest_onnx_grad(self, height_scale, width_scale, height, width,
                               num_channels, batch_size, seed, order, gc, dc):

        assume(order == "NCHW" or gc.device_type == caffe2_pb2.CPU)
        # NHWC currently only supported for CPU. Ignore other devices.
        if order == "NHWC":
            dc = [d for d in dc if d.device_type == caffe2_pb2.CPU]

        np.random.seed(seed)

        output_height = np.int32(height * height_scale)
        output_width = np.int32(width * width_scale)
        X = np.random.rand(batch_size,
                           num_channels,
                           height,
                           width).astype(np.float32)
        dY = np.random.rand(batch_size,
                            num_channels,
                            output_height,
                            output_width).astype(np.float32)
        if order == "NHWC":
            X = X.transpose([0, 2, 3, 1])
            dY = dY.transpose([0, 2, 3, 1])

        scales = np.array([height_scale, width_scale]).astype(np.float32)

        op = core.CreateOperator(
            "ResizeNearestGradient",
            ["dY", "X", "scales"],
            ["dX"],
            order=order,
        )

        def ref(dY, X, scales):
            dX = np.zeros_like(X)

            for i in range(output_height):
                for j in range(output_width):
                    input_i = np.minimum(i / scales[0], height - 1).astype(np.int32)
                    input_j = np.minimum(j / scales[1], width - 1).astype(np.int32)

                    if order == "NCHW":
                        dX[:, :, input_i, input_j] += dY[:, :, i, j]
                    else:
                        dX[:, input_i, input_j, :] += dY[:, i, j, :]

            return dX,

        self.assertDeviceChecks(dc, op, [dY, X, scales], [0])
        self.assertReferenceChecks(gc, op, [dY, X, scales], ref)



if __name__ == "__main__":
    unittest.main()
