from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from hypothesis import given
import hypothesis.strategies as st

from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu


class TestLocallyConnectedOp(hu.HypothesisTestCase):

    @given(kernel=st.integers(1, 3),
           size=st.integers(1, 5),
           input_channels=st.integers(1, 3),
           output_channels=st.integers(1, 3),
           batch_size=st.integers(1, 3),
           order=st.sampled_from(["NCHW", "NHWC"]),
           use_bias=st.booleans(),
           **hu.gcs)
    def test_lc_2d(
            self, kernel, size, input_channels, output_channels, batch_size,
            order, use_bias, gc, dc):
        if size < kernel:
            return

        op = core.CreateOperator(
            "LC2D",
            ["X", "W", "b"] if use_bias else ["X", "W"],
            ["Y"],
            kernel=kernel,
            order=order,
            engine="",
        )

        L = size - kernel + 1
        if order == "NCHW":
            X = np.random.rand(
                batch_size, input_channels, size, size).astype(np.float32) - 0.5
            W = np.random.rand(
                L, L, output_channels, input_channels, kernel, kernel
            ).astype(np.float32) - 0.5
        else:
            X = np.random.rand(
                batch_size, size, size, input_channels).astype(np.float32) - 0.5
            W = np.random.rand(
                L, L, output_channels, kernel, kernel, input_channels
            ).astype(np.float32) - 0.5
        b = np.random.rand(L, L, output_channels).astype(np.float32) - 0.5
        inputs = [X, W, b] if use_bias else [X, W]

        def lc_2d_nchw(X, W, b=None):
            N, C, XH, XW = X.shape
            YH, YW, M, _, KH, KW = W.shape

            def conv(n, m, yh, yw):
                sum = b[yh, yw, m] if b is not None else 0
                for c in range(C):
                    for kh in range(KH):
                        for kw in range(KW):
                            hh = yh + kh
                            ww = yw + kw
                            sum += X[n, c, hh, ww] * W[yh, yw, m, c, kh, kw]
                return sum

            output = np.zeros((N, M, YH, YW), dtype=np.float32)
            for n in range(N):
                for m in range(M):
                    for yh in range(YH):
                        for yw in range(YW):
                            output[n, m, yh, yw] = conv(n, m, yh, yw)
            return [output]

        def lc_2d_nhwc(X, W, b=None):
            XT = np.transpose(X, [0, 3, 1, 2])
            WT = np.transpose(W, [0, 1, 2, 5, 3, 4])
            output = lc_2d_nchw(XT, WT, b)
            return [np.transpose(output[0], [0, 2, 3, 1])]

        ref_op = lc_2d_nchw if order == "NCHW" else lc_2d_nhwc

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=inputs,
            reference=ref_op,
        )
        for i in range(len(inputs)):
            self.assertGradientChecks(gc, op, inputs, i, [0])

    @given(kernel=st.integers(1, 3),
           size=st.integers(1, 5),
           input_channels=st.integers(1, 3),
           output_channels=st.integers(1, 3),
           batch_size=st.integers(1, 3),
           use_bias=st.booleans(),
           **hu.gcs)
    def test_lc_1d(
            self, kernel, size, input_channels, output_channels, batch_size,
            use_bias, gc, dc):
        if size < kernel:
            return

        op = core.CreateOperator(
            "LC1D",
            ["X", "W", "b"] if use_bias else ["X", "W"],
            ["Y"],
            kernels=[kernel],
            order="NCHW",
            engine="",
        )

        L = size - kernel + 1
        X = np.random.rand(
            batch_size, input_channels, size).astype(np.float32) - 0.5
        W = np.random.rand(
            L, output_channels, input_channels, kernel).astype(np.float32) - 0.5
        b = np.random.rand(L, output_channels).astype(np.float32) - 0.5
        inputs = [X, W, b] if use_bias else [X, W]

        def lc_1d_nchw(X, W, b=None):
            N, C, XL = X.shape
            YL, M, _, KL = W.shape

            def conv(n, m, yl):
                sum = b[yl, m] if b is not None else 0
                for c in range(C):
                    for kl in range(KL):
                        ll = yl + kl
                        sum += X[n, c, ll] * W[yl, m, c, kl]
                return sum

            output = np.zeros((N, M, YL), dtype=np.float32)
            for n in range(N):
                for m in range(M):
                    for yl in range(YL):
                        output[n, m, yl] = conv(n, m, yl)
            return [output]

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=inputs,
            reference=lc_1d_nchw,
        )
        for i in range(len(inputs)):
            self.assertGradientChecks(gc, op, inputs, i, [0])


if __name__ == "__main__":
    import unittest
    unittest.main()
