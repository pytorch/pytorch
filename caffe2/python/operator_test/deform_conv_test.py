

import os
import unittest

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import core, utils, workspace
from hypothesis import assume, given


def _cudnn_supports(dilation=False, nhwc=False):
    """Return True if cuDNN supports this configuration."""
    v = workspace.GetCuDNNVersion()
    if dilation and v < 6000:
        # Dilation not supported until v6
        return False
    if dilation and nhwc:
        # Dilation and NHWC not supported together
        return False
    return True


def _conv_1d_output_size(size, kernel, pad, dilation, stride):
    return max(1, int((size + pad * 2 - (dilation * (kernel - 1) + 1)) / stride) + 1)


def _conv_2d_output_size(size, kernel, pad_h, pad_w, dilation, stride_h, stride_w):
    return [
        _conv_1d_output_size(size, kernel, pad_h, dilation, stride_h),
        _conv_1d_output_size(size, kernel, pad_w, dilation, stride_w),
    ]


def _conv_2d_offsets_dims(
    batch_size,
    size,
    kernel,
    pad_h,
    pad_w,
    dilation,
    stride_h,
    stride_w,
    deformable_group,
):
    dims = [batch_size, 2 * kernel * kernel * deformable_group]
    dims.extend(
        _conv_2d_output_size(size, kernel, pad_h, pad_w, dilation, stride_h, stride_w)
    )
    return dims


def _conv_2d_random_offsets(batch_size, kernel, dims, num_deformable_group):
    o = []
    for y0 in range(0, kernel):
        for x0 in range(0, kernel):
            # stay away from integer offsets which correspond to "ridges" on the
            # interpolated surface resulting in less precise estimates
            x = np.random.randint(0, kernel) + np.random.uniform(0.05, 0.95)
            y = np.random.randint(0, kernel) + np.random.uniform(0.05, 0.95)
            o.append(y - y0)
            o.append(x - x0)
    o = o * num_deformable_group
    e = []
    for v in o:
        e.append([[v] * dims[1]] * dims[0])
    return np.array([e] * batch_size).astype(np.float32)


def _conv_2d_shuffle_offsets(
    batch_size, kernel, dims, num_deformable_group, input_channels, output_channels
):
    o = []
    w0 = [[0 for x in range(kernel)] for y in range(kernel)]
    for y0 in range(0, kernel):
        for x0 in range(0, kernel):
            x = np.random.randint(0, kernel)
            y = np.random.randint(0, kernel)
            o.append(y - y0)
            o.append(x - x0)
            w0[y][x] += 1
    o = o * num_deformable_group
    e = []
    for v in o:
        e.append([[v] * int(dims[1])] * int(dims[0]))
    w0 = [[w0] * input_channels] * output_channels
    return (
        np.array([e] * batch_size).astype(np.float32),
        utils.NCHW2NHWC(np.array(w0).astype(np.float32)),
    )


class TestConvolution(hu.HypothesisTestCase):
    @unittest.skipIf(not workspace.has_gpu_support, "No gpu support")
    @given(
        stride=st.integers(1, 3),
        pad=st.integers(0, 3),
        kernel=st.integers(1, 5),
        dilation=st.integers(1, 3),
        size=st.integers(7, 10),
        input_channels=st.integers(1, 8),
        output_channels=st.integers(1, 8),
        batch_size=st.integers(1, 3),
        order=st.sampled_from(["NCHW"]),
        engine=st.sampled_from(["", "CUDNN", "MKLDNN"]),
        use_bias=st.booleans(),
        deformable_group=st.integers(1, 3),
        **hu.gcs_gpu_only
    )
    def test_null_offset_convolution(
        self,
        stride,
        pad,
        kernel,
        dilation,
        size,
        input_channels,
        output_channels,
        batch_size,
        order,
        engine,
        use_bias,
        deformable_group,
        gc,
        dc,
    ):
        dkernel = dilation * (kernel - 1) + 1

        if gc.device_type == caffe2_pb2.CUDA and engine == "CUDNN":
            assume(_cudnn_supports(dilation=(dilation > 1), nhwc=(order == "NHWC")))

        assume(engine != "MKLDNN" or use_bias is True)

        op = core.CreateOperator(
            "DeformConv",
            ["X", "o", "w", "b"] if use_bias else ["X", "o", "w"],
            ["Y"],
            stride=stride,
            kernel=kernel,
            dilation=dilation,
            pad=pad,
            order=order,
            engine=engine,
            deformable_group=deformable_group,
        )
        offset_dims = _conv_2d_offsets_dims(
            batch_size,
            size,
            kernel,
            pad,
            pad,
            dilation,
            stride,
            stride,
            deformable_group,
        )
        X = (
            np.random.rand(batch_size, size, size, input_channels).astype(np.float32)
            - 0.5
        )
        o = np.zeros(tuple(offset_dims), np.float32)
        w = (
            np.random.rand(output_channels, kernel, kernel, input_channels).astype(
                np.float32
            )
            - 0.5
        )
        b = np.random.rand(output_channels).astype(np.float32) - 0.5
        if order == "NCHW":
            X = utils.NHWC2NCHW(X)
            w = utils.NHWC2NCHW(w)

        inputs = [X, o, w, b] if use_bias else [X, o, w]

        # Error handling path.
        if size + pad + pad < dkernel or size + pad + pad < dkernel:
            with self.assertRaises(RuntimeError):
                self.assertDeviceChecks(dc, op, inputs, [0])
            return
        if input_channels % deformable_group != 0:
            with self.assertRaises(RuntimeError):
                self.assertDeviceChecks(dc, op, inputs, [0])
            return
        if output_channels % deformable_group != 0:
            with self.assertRaises(RuntimeError):
                self.assertDeviceChecks(dc, op, inputs, [0])
            return

        def reference_conv_op(*args):
            reference_op = core.CreateOperator(
                "Conv",
                ["X", "w", "b"] if use_bias else ["X", "w"],
                ["Y0"],
                stride=stride,
                kernel=kernel,
                dilation=dilation,
                pad=pad,
                order=order,
                engine=engine,
                device_option=gc,
            )
            workspace.RunOperatorOnce(reference_op)
            reference_blob = workspace.FetchBlob("Y0")
            return (reference_blob,)

        self.assertReferenceChecks(gc, op, inputs, reference_conv_op)

    @unittest.skipIf(not workspace.has_gpu_support, "No gpu support")
    @given(
        stride=st.integers(1, 3),
        pad=st.integers(0, 0),
        kernel=st.integers(1, 5),
        dilation=st.integers(1, 3),
        size=st.integers(7, 10),
        input_channels=st.integers(1, 8),
        output_channels=st.integers(1, 8),
        batch_size=st.integers(1, 3),
        order=st.sampled_from(["NCHW"]),
        engine=st.sampled_from(["", "CUDNN", "MKLDNN"]),
        use_bias=st.booleans(),
        deformable_group=st.integers(1, 4),
        **hu.gcs_gpu_only
    )
    def test_flat_input_convolution(
        self,
        stride,
        pad,
        kernel,
        dilation,
        size,
        input_channels,
        output_channels,
        batch_size,
        order,
        engine,
        use_bias,
        deformable_group,
        gc,
        dc,
    ):
        dkernel = dilation * (kernel - 1) + 1

        if gc.device_type == caffe2_pb2.CUDA and engine == "CUDNN":
            assume(_cudnn_supports(dilation=(dilation > 1), nhwc=(order == "NHWC")))

        assume(engine != "MKLDNN" or use_bias is True)

        op = core.CreateOperator(
            "DeformConv",
            ["X", "o", "w", "b"] if use_bias else ["X", "o", "w"],
            ["Y"],
            stride=stride,
            kernel=kernel,
            dilation=dilation,
            pad=pad,
            order=order,
            engine=engine,
            deformable_group=deformable_group,
        )
        X = np.ones((batch_size, size, size, input_channels), np.float32) - 0.5
        output_size = _conv_2d_output_size(
            size, kernel, pad, pad, dilation, stride, stride
        )
        o = _conv_2d_random_offsets(batch_size, kernel, output_size, deformable_group)
        w = np.ones((output_channels, kernel, kernel, input_channels), np.float32) - 0.5
        b = np.random.rand(output_channels).astype(np.float32) - 0.5
        if order == "NCHW":
            X = utils.NHWC2NCHW(X)
            w = utils.NHWC2NCHW(w)

        inputs = [X, o, w, b] if use_bias else [X, o, w]

        # Error handling path.
        if size + pad + pad < dkernel or size + pad + pad < dkernel:
            with self.assertRaises(RuntimeError):
                self.assertDeviceChecks(dc, op, inputs, [0])
            return
        if input_channels % deformable_group != 0:
            with self.assertRaises(RuntimeError):
                self.assertDeviceChecks(dc, op, inputs, [0])
            return
        if output_channels % deformable_group != 0:
            with self.assertRaises(RuntimeError):
                self.assertDeviceChecks(dc, op, inputs, [0])
            return

        def reference_conv_op(*args):
            reference_op = core.CreateOperator(
                "Conv",
                ["X", "w", "b"] if use_bias else ["X", "w"],
                ["Y0"],
                stride=stride,
                kernel=kernel,
                dilation=dilation,
                pad=pad,
                order=order,
                engine=engine,
                device_option=gc,
            )
            workspace.RunOperatorOnce(reference_op)
            reference_blob = workspace.FetchBlob("Y0")
            return (reference_blob,)

        self.assertReferenceChecks(gc, op, inputs, reference_conv_op)

    @unittest.skipIf(not workspace.has_gpu_support, "No gpu support")
    @given(
        stride=st.integers(1, 1),
        pad=st.integers(0, 0),
        kernel=st.integers(1, 5),
        dilation=st.integers(1, 1),
        size=st.integers(7, 10),
        input_channels=st.integers(1, 8),
        output_channels=st.integers(1, 8),
        batch_size=st.integers(1, 3),
        order=st.sampled_from(["NCHW"]),
        engine=st.sampled_from(["", "CUDNN", "MKLDNN"]),
        use_bias=st.booleans(),
        deformable_group=st.integers(1, 4),
        **hu.gcs_gpu_only
    )
    def test_shuffle_input_convolution(
        self,
        stride,
        pad,
        kernel,
        dilation,
        size,
        input_channels,
        output_channels,
        batch_size,
        order,
        engine,
        use_bias,
        deformable_group,
        gc,
        dc,
    ):
        dkernel = dilation * (kernel - 1) + 1

        if gc.device_type == caffe2_pb2.CUDA and engine == "CUDNN":
            assume(_cudnn_supports(dilation=(dilation > 1), nhwc=(order == "NHWC")))

        assume(engine != "MKLDNN" or use_bias is True)

        op = core.CreateOperator(
            "DeformConv",
            ["X", "o", "w", "b"] if use_bias else ["X", "o", "w"],
            ["Y"],
            stride=stride,
            kernel=kernel,
            dilation=dilation,
            pad=pad,
            order=order,
            engine=engine,
            deformable_group=deformable_group,
        )
        X = (
            np.random.rand(batch_size, size, size, input_channels).astype(np.float32)
            - 0.5
        )
        output_size = _conv_2d_output_size(
            size, kernel, pad, pad, dilation, stride, stride
        )
        o, w0 = _conv_2d_shuffle_offsets(
            batch_size,
            kernel,
            output_size,
            deformable_group,
            input_channels,
            output_channels,
        )
        w = np.ones((output_channels, kernel, kernel, input_channels), np.float32)
        b = np.random.rand(output_channels).astype(np.float32) - 0.5

        if order == "NCHW":
            X = utils.NHWC2NCHW(X)
            w = utils.NHWC2NCHW(w)
            w0 = utils.NHWC2NCHW(w0)

        inputs = [X, o, w, b] if use_bias else [X, o, w]

        # Error handling path.
        if size + pad + pad < dkernel or size + pad + pad < dkernel:
            with self.assertRaises(RuntimeError):
                self.assertDeviceChecks(dc, op, inputs, [0])
            return
        if input_channels % deformable_group != 0:
            with self.assertRaises(RuntimeError):
                self.assertDeviceChecks(dc, op, inputs, [0])
            return
        if output_channels % deformable_group != 0:
            with self.assertRaises(RuntimeError):
                self.assertDeviceChecks(dc, op, inputs, [0])
            return

        def reference_conv_op(*args):
            with core.DeviceScope(gc):
                workspace.FeedBlob("w0", w0)
            reference_op = core.CreateOperator(
                "Conv",
                ["X", "w0", "b"] if use_bias else ["X", "w0"],
                ["Y0"],
                stride=stride,
                kernel=kernel,
                dilation=dilation,
                pad=pad,
                order=order,
                engine=engine,
                device_option=gc,
            )
            workspace.RunOperatorOnce(reference_op)
            reference_blob = workspace.FetchBlob("Y0")
            return (reference_blob,)

        self.assertReferenceChecks(gc, op, inputs, reference_conv_op)

    # CUDNN does NOT support different padding values and we skip it
    @unittest.skipIf(not workspace.has_gpu_support, "No gpu support")
    @given(
        stride_h=st.integers(1, 3),
        stride_w=st.integers(1, 3),
        pad_h=st.integers(0, 3),
        pad_w=st.integers(0, 3),
        kernel=st.integers(2, 5),
        size=st.integers(1, 8),
        input_channels=st.integers(1, 3),
        output_channels=st.integers(1, 3),
        batch_size=st.integers(1, 3),
        order=st.sampled_from(["NCHW"]),
        shared_buffer=st.booleans(),
        use_bias=st.booleans(),
        deformable_group=st.integers(1, 3),
        **hu.gcs_gpu_only
    )
    def test_conv_separate_stride_pad_gradients(
        self,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        kernel,
        size,
        input_channels,
        output_channels,
        batch_size,
        order,
        shared_buffer,
        use_bias,
        deformable_group,
        gc,
        dc,
    ):
        op = core.CreateOperator(
            "DeformConv",
            ["X", "o", "w", "b"] if use_bias else ["X", "o", "w"],
            ["Y"],
            stride_h=stride_h,
            stride_w=stride_w,
            pad_t=pad_h,
            pad_l=pad_w,
            pad_b=pad_h,
            pad_r=pad_w,
            kernel=kernel,
            order=order,
            shared_buffer=int(shared_buffer),
            deformable_group=deformable_group,
        )
        X = (
            np.random.rand(batch_size, size, size, input_channels).astype(np.float32)
            - 0.5
        )
        output_size = _conv_2d_output_size(
            size, kernel, pad_h, pad_w, 1, stride_h, stride_w
        )
        o = _conv_2d_random_offsets(batch_size, kernel, output_size, deformable_group)
        w = (
            np.random.rand(output_channels, kernel, kernel, input_channels).astype(
                np.float32
            )
            - 0.5
        )
        b = np.random.rand(output_channels).astype(np.float32) - 0.5
        if order == "NCHW":
            X = utils.NHWC2NCHW(X)
            w = utils.NHWC2NCHW(w)

        inputs = [X, o, w, b] if use_bias else [X, o, w]

        # Error handling path.
        if size + pad_h * 2 < kernel or size + pad_w * 2 < kernel:
            with self.assertRaises(RuntimeError):
                self.assertDeviceChecks(dc, op, inputs, [0])
            return
        if input_channels % deformable_group != 0:
            with self.assertRaises(RuntimeError):
                self.assertDeviceChecks(dc, op, inputs, [0])
            return
        if output_channels % deformable_group != 0:
            with self.assertRaises(RuntimeError):
                self.assertDeviceChecks(dc, op, inputs, [0])
            return

        self.assertDeviceChecks(dc, op, inputs, [0])
        for i in range(len(inputs)):
            self.assertGradientChecks(gc, op, inputs, i, [0])

    @unittest.skipIf(not workspace.has_gpu_support, "No gpu support")
    @given(
        stride=st.integers(1, 3),
        pad=st.integers(0, 3),
        kernel=st.integers(1, 5),
        dilation=st.integers(1, 3),
        size=st.integers(7, 10),
        input_channels=st.integers(1, 8),
        output_channels=st.integers(1, 8),
        batch_size=st.integers(1, 3),
        order=st.sampled_from(["NCHW"]),
        engine=st.sampled_from(["", "CUDNN", "MKLDNN"]),
        use_bias=st.booleans(),
        deformable_group=st.integers(1, 3),
        **hu.gcs_gpu_only
    )
    def test_conv_gradients(
        self,
        stride,
        pad,
        kernel,
        dilation,
        size,
        input_channels,
        output_channels,
        batch_size,
        order,
        engine,
        use_bias,
        deformable_group,
        gc,
        dc,
    ):
        dkernel = dilation * (kernel - 1) + 1

        if gc.device_type == caffe2_pb2.CUDA and engine == "CUDNN":
            assume(_cudnn_supports(dilation=(dilation > 1), nhwc=(order == "NHWC")))

        assume(engine != "MKLDNN" or use_bias is True)

        op = core.CreateOperator(
            "DeformConv",
            ["X", "o", "w", "b"] if use_bias else ["X", "o", "w"],
            ["Y"],
            stride=stride,
            kernel=kernel,
            dilation=dilation,
            pad=pad,
            order=order,
            engine=engine,
            deformable_group=deformable_group,
        )
        X = (
            np.random.rand(batch_size, size, size, input_channels).astype(np.float32)
            - 0.5
        )
        output_size = _conv_2d_output_size(
            size, kernel, pad, pad, dilation, stride, stride
        )
        o = _conv_2d_random_offsets(batch_size, kernel, output_size, deformable_group)
        w = (
            np.random.rand(output_channels, kernel, kernel, input_channels).astype(
                np.float32
            )
            - 0.5
        )
        b = np.random.rand(output_channels).astype(np.float32) - 0.5
        if order == "NCHW":
            X = utils.NHWC2NCHW(X)
            w = utils.NHWC2NCHW(w)

        inputs = [X, o, w, b] if use_bias else [X, o, w]
        # Error handling path.
        if size + pad + pad < dkernel or size + pad + pad < dkernel:
            with self.assertRaises(RuntimeError):
                self.assertDeviceChecks(dc, op, inputs, [0])
            return
        if input_channels % deformable_group != 0:
            with self.assertRaises(RuntimeError):
                self.assertDeviceChecks(dc, op, inputs, [0])
            return
        if output_channels % deformable_group != 0:
            with self.assertRaises(RuntimeError):
                self.assertDeviceChecks(dc, op, inputs, [0])
            return

        self.assertDeviceChecks(dc, op, inputs, [0])
        for i in range(len(inputs)):
            self.assertGradientChecks(gc, op, inputs, i, [0])


if __name__ == "__main__":
    import unittest

    unittest.main()
