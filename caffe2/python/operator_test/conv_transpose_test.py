from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from hypothesis import assume, given, settings
import hypothesis.strategies as st

from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu


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
           **hu.gcs)
    def test_convolution_transpose_layout(self, stride, pad, kernel, adj,
                                          size, input_channels,
                                          output_channels, batch_size,
                                          engine, shared_buffer, gc, dc):
        assume(adj < stride)
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
                ["X", "w", "b"],
                ["Y"],
                stride=stride,
                kernel=kernel,
                pad=pad,
                adj=adj,
                order=order,
                engine=engine,
                shared_buffer=int(shared_buffer),
                device_option=gc,
            )
            if order == "NCHW":
                X_f = X.transpose((0, 3, 1, 2))
                w_f = w.transpose((0, 3, 1, 2))
            else:
                X_f = X
                w_f = w
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
            outputs["NHWC"].transpose((0, 3, 1, 2)),
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
           engine=st.sampled_from(["", "BLOCK"]), **hu.gcs)
    def test_convolution_transpose_separate_stride_pad_adj_layout(
            self, stride_h, stride_w, pad_t, pad_l, pad_b, pad_r, kernel,
            adj_h, adj_w, size, input_channels, output_channels, batch_size,
            engine, gc, dc):
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
                ["X", "w", "b"],
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
                X_f = X.transpose((0, 3, 1, 2))
                w_f = w.transpose((0, 3, 1, 2))
            else:
                X_f = X
                w_f = w
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
            outputs["NHWC"].transpose((0, 3, 1, 2)),
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
           engine=st.sampled_from(["", "CUDNN", "BLOCK"]), **hu.gcs)
    @settings(max_examples=2, timeout=100)
    def test_convolution_transpose_gradients(self, stride, pad, kernel, adj,
                                             size, input_channels,
                                             output_channels, batch_size,
                                             order, engine, gc, dc):
        assume(adj < stride)
        X = np.random.rand(
            batch_size, size, size, input_channels).astype(np.float32) - 0.5
        w = np.random.rand(
            input_channels, kernel, kernel, output_channels)\
            .astype(np.float32) - 0.5
        b = np.random.rand(output_channels).astype(np.float32) - 0.5
        op = core.CreateOperator(
            "ConvTranspose",
            ["X", "w", "b"],
            ["Y"],
            stride=stride,
            kernel=kernel,
            pad=pad,
            adj=adj,
            order=order,
            engine=engine,
        )
        if order == "NCHW":
            X = X.transpose((0, 3, 1, 2))
            w = w.transpose((0, 3, 1, 2))

        self.assertDeviceChecks(dc, op, [X, w, b], [0])
        for i in range(3):
            self.assertGradientChecks(gc, op, [X, w, b], i, [0])

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
           engine=st.sampled_from(["", "BLOCK"]), **hu.gcs)
    @settings(max_examples=2, timeout=100)
    def test_convolution_transpose_separate_stride_pad_adj_gradient(
            self, stride_h, stride_w, pad_t, pad_l, pad_b, pad_r, kernel,
            adj_h, adj_w, size, input_channels, output_channels, batch_size,
            order, engine, gc, dc):
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
            ["X", "w", "b"],
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
        )
        if order == "NCHW":
            X = X.transpose((0, 3, 1, 2))
            w = w.transpose((0, 3, 1, 2))

        self.assertDeviceChecks(dc, op, [X, w, b], [0])
        for i in range(3):
            self.assertGradientChecks(gc, op, [X, w, b], i, [0])

if __name__ == "__main__":
    import unittest
    unittest.main()
