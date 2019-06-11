from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from hypothesis import assume, given, settings
import hypothesis.strategies as st

from caffe2.proto import caffe2_pb2
from caffe2.python import core, utils
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.hip_test_util as hiputl


class TestConvolutionTranspose(hu.HypothesisTestCase):
    @given(stride=st.integers(1, 3),
           pad=st.integers(0, 3),
           kernel=st.integers(1, 5),
           adj=st.integers(0, 2),
           size=st.integers(7, 10),
           input_channels=st.integers(1, 8),
           output_channels=st.integers(1, 8),
           batch_size=st.integers(1, 3),
           engine=st.sampled_from(["", "CUDNN", "BLOCK"]),
           shared_buffer=st.booleans(),
           use_bias=st.booleans(),
           **hu.gcs)
    def test_convolution_transpose_layout_legacy_args(
            self, stride, pad, kernel, adj,
            size, input_channels,
            output_channels, batch_size,
            engine, shared_buffer, use_bias, gc, dc):
        assume(adj < stride)
        X = np.random.rand(
            batch_size, size, size, input_channels).astype(np.float32) - 0.5
        w = np.random.rand(
            input_channels, kernel, kernel, output_channels)\
            .astype(np.float32) - 0.5
        b = np.random.rand(output_channels).astype(np.float32) - 0.5
        outputs = {}
        for order in ["NCHW", "NHWC"]:
            # MIOPEN doesn't work with NHWC, fallback to use normal hip
            if hiputl.run_in_hip(gc, dc) and order == "NHWC":
                tmp_engine = ""
            else:
                tmp_engine = engine
            op = core.CreateOperator(
                "ConvTranspose",
                ["X", "w", "b"] if use_bias else ["X", "w"],
                ["Y"],
                stride=stride,
                kernel=kernel,
                pad=pad,
                adj=adj,
                order=order,
                engine=tmp_engine,
                shared_buffer=int(shared_buffer),
                device_option=gc,
            )
            if order == "NCHW":
                X_f = utils.NHWC2NCHW(X)
                w_f = utils.NHWC2NCHW(w)
            else:
                X_f = X
                w_f = w
            self.assertDeviceChecks(
                dc,
                op,
                [X_f, w_f, b] if use_bias else [X_f, w_f],
                [0])
            self.ws.create_blob("X").feed(X_f, device_option=gc)
            self.ws.create_blob("w").feed(w_f, device_option=gc)
            self.ws.create_blob("b").feed(b, device_option=gc)
            self.ws.run(op)
            outputs[order] = self.ws.blobs["Y"].fetch()
        output_size = (size - 1) * stride + kernel + adj - 2 * pad
        self.assertEqual(
            outputs["NCHW"].shape,
            (batch_size, output_channels, output_size, output_size))
        np.testing.assert_allclose(
            outputs["NCHW"],
            utils.NHWC2NCHW(outputs["NHWC"]),
            atol=1e-4,
            rtol=1e-4)

    @given(stride=st.integers(1, 3),
           pad=st.integers(0, 3),
           kernel=st.integers(1, 5),
           adj=st.integers(0, 2),
           size=st.integers(7, 10),
           input_channels=st.integers(1, 8),
           output_channels=st.integers(1, 8),
           batch_size=st.integers(1, 3),
           engine=st.sampled_from(["", "CUDNN", "BLOCK"]),
           shared_buffer=st.booleans(),
           use_bias=st.booleans(),
           **hu.gcs)
    def test_convolution_transpose_layout(
            self, stride, pad, kernel, adj,
            size, input_channels,
            output_channels, batch_size,
            engine, shared_buffer, use_bias, gc, dc):
        assume(adj < stride)
        X = np.random.rand(
            batch_size, size, size, input_channels).astype(np.float32) - 0.5
        w = np.random.rand(
            input_channels, kernel, kernel, output_channels)\
            .astype(np.float32) - 0.5
        b = np.random.rand(output_channels).astype(np.float32) - 0.5
        outputs = {}
        for order in ["NCHW", "NHWC"]:
            if hiputl.run_in_hip(gc, dc) and order == "NHWC":
                # MIOPEN doesn't work with NHWC, fallback to use normal hip
                tmp_engine = ""
            else:
                tmp_engine = engine
            op = core.CreateOperator(
                "ConvTranspose",
                ["X", "w", "b"] if use_bias else ["X", "w"],
                ["Y"],
                strides=[stride] * 2,
                kernels=[kernel] * 2,
                pads=[pad] * 4,
                adjs=[adj] * 2,
                order=order,
                engine=tmp_engine,
                shared_buffer=int(shared_buffer),
                device_option=gc,
            )
            if order == "NCHW":
                X_f = utils.NHWC2NCHW(X)
                w_f = utils.NHWC2NCHW(w)
            else:
                X_f = X
                w_f = w
            self.assertDeviceChecks(
                dc,
                op,
                [X_f, w_f, b] if use_bias else [X_f, w_f],
                [0])
            self.ws.create_blob("X").feed(X_f, device_option=gc)
            self.ws.create_blob("w").feed(w_f, device_option=gc)
            self.ws.create_blob("b").feed(b, device_option=gc)
            self.ws.run(op)
            outputs[order] = self.ws.blobs["Y"].fetch()
        output_size = (size - 1) * stride + kernel + adj - 2 * pad
        self.assertEqual(
            outputs["NCHW"].shape,
            (batch_size, output_channels, output_size, output_size))
        np.testing.assert_allclose(
            outputs["NCHW"],
            utils.NHWC2NCHW(outputs["NHWC"]),
            atol=1e-4,
            rtol=1e-4)

    # CUDNN does not support separate stride and pad so we skip it.
    @given(stride_h=st.integers(1, 3),
           stride_w=st.integers(1, 3),
           pad_t=st.integers(0, 3),
           pad_l=st.integers(0, 3),
           pad_b=st.integers(0, 3),
           pad_r=st.integers(0, 3),
           kernel=st.integers(1, 5),
           adj_h=st.integers(0, 2),
           adj_w=st.integers(0, 2),
           size=st.integers(7, 10),
           input_channels=st.integers(1, 8),
           output_channels=st.integers(1, 8),
           batch_size=st.integers(1, 3),
           engine=st.sampled_from(["", "BLOCK"]),
           use_bias=st.booleans(),
           **hu.gcs)
    def test_convolution_transpose_separate_stride_pad_adj_layout(
            self, stride_h, stride_w, pad_t, pad_l, pad_b, pad_r, kernel,
            adj_h, adj_w, size, input_channels, output_channels, batch_size,
            engine, use_bias, gc, dc):
        assume(adj_h < stride_h)
        assume(adj_w < stride_w)
        X = np.random.rand(
            batch_size, size, size, input_channels).astype(np.float32) - 0.5
        w = np.random.rand(
            input_channels, kernel, kernel, output_channels)\
            .astype(np.float32) - 0.5
        b = np.random.rand(output_channels).astype(np.float32) - 0.5
        outputs = {}
        for order in ["NCHW", "NHWC"]:
            op = core.CreateOperator(
                "ConvTranspose",
                ["X", "w", "b"] if use_bias else ["X", "w"],
                ["Y"],
                stride_h=stride_h,
                stride_w=stride_w,
                kernel=kernel,
                pad_t=pad_t,
                pad_l=pad_l,
                pad_b=pad_b,
                pad_r=pad_r,
                adj_h=adj_h,
                adj_w=adj_w,
                order=order,
                engine=engine,
                device_option=gc,
            )
            if order == "NCHW":
                X_f = utils.NHWC2NCHW(X)
                w_f = utils.NHWC2NCHW(w)
            else:
                X_f = X
                w_f = w
            self.assertDeviceChecks(
                dc,
                op,
                [X_f, w_f, b] if use_bias else [X_f, w_f],
                [0])
            self.ws.create_blob("X").feed(X_f, device_option=gc)
            self.ws.create_blob("w").feed(w_f, device_option=gc)
            self.ws.create_blob("b").feed(b, device_option=gc)
            self.ws.run(op)
            outputs[order] = self.ws.blobs["Y"].fetch()
        output_h = (size - 1) * stride_h + kernel + adj_h - pad_t - pad_b
        output_w = (size - 1) * stride_w + kernel + adj_w - pad_l - pad_r
        self.assertEqual(
            outputs["NCHW"].shape,
            (batch_size, output_channels, output_h, output_w))
        np.testing.assert_allclose(
            outputs["NCHW"],
            utils.NHWC2NCHW(outputs["NHWC"]),
            atol=1e-4,
            rtol=1e-4)

    @given(stride=st.integers(1, 3),
           pad=st.integers(0, 3),
           kernel=st.integers(1, 5),
           adj=st.integers(0, 2),
           size=st.integers(7, 10),
           input_channels=st.integers(1, 8),
           output_channels=st.integers(1, 8),
           batch_size=st.integers(1, 3),
           order=st.sampled_from(["NCHW", "NHWC"]),
           engine=st.sampled_from(["", "CUDNN", "BLOCK"]),
           use_bias=st.booleans(),
           compute_dX=st.booleans(),
           **hu.gcs)
    @settings(max_examples=2, timeout=100)
    def test_convolution_transpose_gradients(self, stride, pad, kernel, adj,
                                             size, input_channels,
                                             output_channels, batch_size,
                                             order, engine, use_bias,
                                             compute_dX, gc, dc):
        assume(adj < stride)
        if hiputl.run_in_hip(gc, dc) and engine == "CUDNN":
            assume(order == "NCHW")
        X = np.random.rand(
            batch_size, size, size, input_channels).astype(np.float32) - 0.5
        w = np.random.rand(
            input_channels, kernel, kernel, output_channels)\
            .astype(np.float32) - 0.5
        b = np.random.rand(output_channels).astype(np.float32) - 0.5
        op = core.CreateOperator(
            "ConvTranspose",
            ["X", "w", "b"] if use_bias else ["X", "w"],
            ["Y"],
            stride=stride,
            kernel=kernel,
            pad=pad,
            adj=adj,
            order=order,
            engine=engine,
            no_gradient_to_input=not compute_dX,
        )
        if order == "NCHW":
            X = utils.NHWC2NCHW(X)
            w = utils.NHWC2NCHW(w)

        inputs = [X, w, b] if use_bias else [X, w]
        self.assertDeviceChecks(dc, op, inputs, [0])

        if use_bias and compute_dX:
            # w, b, X
            outputs_to_check = [1, 2, 0]
        elif use_bias:
            # w, b
            outputs_to_check = [1, 2]
        elif compute_dX:
            # w, X
            outputs_to_check = [1, 0]
        else:
            # w
            outputs_to_check = [1]
        for i in outputs_to_check:
            self.assertGradientChecks(gc, op, inputs, i, [0])

    # CUDNN does not support separate stride and pad so we skip it.
    @given(stride_h=st.integers(1, 3),
           stride_w=st.integers(1, 3),
           pad_t=st.integers(0, 3),
           pad_l=st.integers(0, 3),
           pad_b=st.integers(0, 3),
           pad_r=st.integers(0, 3),
           kernel=st.integers(1, 5),
           adj_h=st.integers(0, 2),
           adj_w=st.integers(0, 2),
           size=st.integers(7, 10),
           input_channels=st.integers(1, 8),
           output_channels=st.integers(1, 8),
           batch_size=st.integers(1, 3),
           order=st.sampled_from(["NCHW", "NHWC"]),
           engine=st.sampled_from(["", "BLOCK"]),
           use_bias=st.booleans(),
           compute_dX=st.booleans(),
           **hu.gcs)
    @settings(max_examples=2, timeout=100)
    def test_convolution_transpose_separate_stride_pad_adj_gradient(
            self, stride_h, stride_w, pad_t, pad_l, pad_b, pad_r, kernel,
            adj_h, adj_w, size, input_channels, output_channels, batch_size,
            order, engine, use_bias, compute_dX, gc, dc):
        assume(adj_h < stride_h)
        assume(adj_w < stride_w)
        X = np.random.rand(
            batch_size, size, size, input_channels).astype(np.float32) - 0.5
        w = np.random.rand(
            input_channels, kernel, kernel, output_channels)\
            .astype(np.float32) - 0.5
        b = np.random.rand(output_channels).astype(np.float32) - 0.5
        op = core.CreateOperator(
            "ConvTranspose",
            ["X", "w", "b"] if use_bias else ["X", "w"],
            ["Y"],
            stride_h=stride_h,
            stride_w=stride_w,
            kernel=kernel,
            pad_t=pad_t,
            pad_l=pad_l,
            pad_b=pad_b,
            pad_r=pad_r,
            adj_h=adj_h,
            adj_w=adj_w,
            order=order,
            engine=engine,
            no_gradient_to_input=not compute_dX,
        )
        if order == "NCHW":
            X = utils.NHWC2NCHW(X)
            w = utils.NHWC2NCHW(w)

        inputs = [X, w, b] if use_bias else [X, w]
        self.assertDeviceChecks(dc, op, inputs, [0])

        if use_bias and compute_dX:
            # w, b, X
            outputs_to_check = [1, 2, 0]
        elif use_bias:
            # w, b
            outputs_to_check = [1, 2]
        elif compute_dX:
            # w, X
            outputs_to_check = [1, 0]
        else:
            # w
            outputs_to_check = [1]
        for i in outputs_to_check:
            self.assertGradientChecks(gc, op, inputs, i, [0])

    @given(stride=st.integers(1, 3),
           pad=st.integers(0, 3),
           kernel=st.integers(1, 3),
           adj=st.integers(0, 2),
           size=st.integers(7, 10),
           input_channels=st.integers(1, 8),
           output_channels=st.integers(1, 8),
           batch_size=st.integers(1, 4),
           group=st.integers(1, 4),
           order=st.sampled_from(["NCHW", "NHWC"]),
           engine=st.sampled_from(["", "CUDNN", "BLOCK"]),
           shared_buffer=st.booleans(),
           use_bias=st.booleans(),
           **hu.gcs)
    def test_convolution_transpose_with_group(
            self, stride, pad, kernel, adj, size, input_channels,
            output_channels, batch_size, group, order, engine, shared_buffer,
            use_bias, gc, dc):
        assume(adj < stride)
        # TODO: Group conv_transpose in NHWC not implemented for GPU yet.
        assume(group == 1 or order == "NCHW" or
               gc.device_type == caffe2_pb2.CPU)
        if group != 1 and order == "NHWC":
            dc = [d for d in dc if d.device_type == caffe2_pb2.CPU]

        if hiputl.run_in_hip(gc, dc) and order == "NHWC":
            engine = ""

        op = core.CreateOperator(
            "ConvTranspose",
            ["X", "w", "b"] if use_bias else ["X", "w"],
            ["Y"],
            stride=stride,
            kernel=kernel,
            pad=pad,
            adj=adj,
            group=group,
            order=order,
            engine=engine,
            shared_buffer=int(shared_buffer),
            device_option=gc,
        )

        input_channels *= group
        output_channels *= group

        X = np.random.rand(
            batch_size, size, size, input_channels).astype(np.float32) - 0.5
        w = np.random.rand(
            input_channels, kernel, kernel, int(output_channels / group)) \
            .astype(np.float32) - 0.5
        b = np.random.rand(output_channels).astype(np.float32) - 0.5
        if order == "NCHW":
            X = utils.NHWC2NCHW(X)
            w = utils.NHWC2NCHW(w)

        inputs = [X, w, b] if use_bias else [X, w]
        self.assertDeviceChecks(dc, op, inputs, [0])
        for i in range(len(inputs)):
            self.assertGradientChecks(gc, op, inputs, i, [0])


if __name__ == "__main__":
    import unittest
    unittest.main()
