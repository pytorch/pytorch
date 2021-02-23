



import numpy as np
from hypothesis import given, settings, assume
import hypothesis.strategies as st

from caffe2.python import core, utils, workspace
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial



class TestLocallyConnectedOp(serial.SerializedTestCase):
    @given(N=st.integers(1, 3),
           C=st.integers(1, 3),
           H=st.integers(1, 5),
           W=st.integers(1, 5),
           M=st.integers(1, 3),
           kernel=st.integers(1, 3),
           op_name=st.sampled_from(["LC", "LC2D"]),
           order=st.sampled_from(["NCHW", "NHWC"]),
           use_bias=st.booleans(),
           **hu.gcs)
    @settings(deadline=10000)
    def test_lc_2d(
            self, N, C, H, W, M, kernel, op_name, order, use_bias, gc, dc):
        if H < kernel:
            kernel = H
        if W < kernel:
            kernel = W

        assume(C == kernel * N)

        op = core.CreateOperator(
            op_name,
            ["X", "W", "b"] if use_bias else ["X", "W"],
            ["Y"],
            kernels=[kernel, kernel],
            order=order,
            engine="",
        )

        Y_H = H - kernel + 1
        Y_W = W - kernel + 1
        if order == "NCHW":
            X = np.random.rand(N, C, H, W).astype(np.float32) - 0.5
            W = np.random.rand(Y_H, Y_W, M, C, kernel,
                               kernel).astype(np.float32) - 0.5
        else:
            X = np.random.rand(N, H, W, C).astype(np.float32) - 0.5
            W = np.random.rand(Y_H, Y_W, M, kernel, kernel,
                               C).astype(np.float32) - 0.5
        b = np.random.rand(Y_H, Y_W, M).astype(np.float32) - 0.5
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
            XT = utils.NHWC2NCHW(X)
            WT = np.transpose(W, [0, 1, 2, 5, 3, 4])
            output = lc_2d_nchw(XT, WT, b)
            return [utils.NCHW2NHWC(output[0])]

        ref_op = lc_2d_nchw if order == "NCHW" else lc_2d_nhwc

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=inputs,
            reference=ref_op,
        )
        self.assertDeviceChecks(dc, op, inputs, [0])
        for i in range(len(inputs)):
            self.assertGradientChecks(gc, op, inputs, i, [0])

    @given(N=st.integers(1, 3),
           C=st.integers(1, 3),
           size=st.integers(1, 5),
           M=st.integers(1, 3),
           kernel=st.integers(1, 3),
           op_name=st.sampled_from(["LC", "LC1D"]),
           use_bias=st.booleans(),
           **hu.gcs)
    @settings(deadline=5000)
    # Increased timeout from 1 second to 5 for ROCM
    def test_lc_1d(self, N, C, size, M, kernel, op_name, use_bias, gc, dc):
        if size < kernel:
            kernel = size

        op = core.CreateOperator(
            op_name,
            ["X", "W", "b"] if use_bias else ["X", "W"],
            ["Y"],
            kernels=[kernel],
            order="NCHW",
            engine="",
        )

        L = size - kernel + 1
        X = np.random.rand(N, C, size).astype(np.float32) - 0.5
        W = np.random.rand(L, M, C, kernel).astype(np.float32) - 0.5
        b = np.random.rand(L, M).astype(np.float32) - 0.5
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
        self.assertDeviceChecks(dc, op, inputs, [0])
        for i in range(len(inputs)):
            self.assertGradientChecks(gc, op, inputs, i, [0])

    @given(N=st.integers(1, 1),
           C=st.integers(1, 1),
           T=st.integers(2, 2),
           H=st.integers(2, 2),
           W=st.integers(2, 2),
           M=st.integers(1, 1),
           kernel=st.integers(2, 2),
           op_name=st.sampled_from(["LC", "LC3D"]),
           use_bias=st.booleans(),
           **hu.gcs)
    @settings(deadline=1000)
    def test_lc_3d(self, N, C, T, H, W, M, kernel, op_name, use_bias, gc, dc):
        if T < kernel:
            kernel = T
        if H < kernel:
            kernel = H
        if W < kernel:
            kernel = W

        op = core.CreateOperator(
            op_name,
            ["X", "W", "b"] if use_bias else ["X", "W"],
            ["Y"],
            kernels=[kernel, kernel, kernel],
            order="NCHW",
            engine="",
        )

        Y_T = T - kernel + 1
        Y_H = H - kernel + 1
        Y_W = W - kernel + 1
        X = np.random.rand(N, C, T, H, W).astype(np.float32) - 0.5
        W = np.random.rand(Y_T, Y_H, Y_W, M, C, kernel,
                           kernel, kernel).astype(np.float32) - 0.5
        b = np.random.rand(Y_T, Y_H, Y_W, M).astype(np.float32) - 0.5
        inputs = [X, W, b] if use_bias else [X, W]

        def lc_3d_nchw(X, W, b=None):
            N, C, XT, XH, XW = X.shape
            YT, YH, YW, M, _, KT, KH, KW = W.shape

            def conv(n, m, yt, yh, yw):
                sum = b[yt, yh, yw, m] if b is not None else 0
                for c in range(C):
                    for kt in range(KT):
                        for kh in range(KH):
                            for kw in range(KW):
                                tt = yt + kt
                                hh = yh + kh
                                ww = yw + kw
                                sum += X[n, c, tt, hh, ww] * \
                                    W[yt, yh, yw, m, c, kt, kh, kw]
                return sum

            output = np.zeros((N, M, YT, YH, YW), dtype=np.float32)
            for n in range(N):
                for m in range(M):
                    for yt in range(YT):
                        for yh in range(YH):
                            for yw in range(YW):
                                output[n, m, yt, yh, yw] = conv(
                                    n, m, yt, yh, yw)
            return [output]

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=inputs,
            reference=lc_3d_nchw,
        )
        self.assertDeviceChecks(dc, op, inputs, [0])
        for i in range(len(inputs)):
            self.assertGradientChecks(gc, op, inputs, i, [0])
