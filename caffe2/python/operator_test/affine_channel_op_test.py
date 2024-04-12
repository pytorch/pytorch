



from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
from hypothesis import given, settings
import hypothesis.strategies as st
import numpy as np


class TestAffineChannelOp(serial.SerializedTestCase):
    def affine_channel_nchw_ref(self, X, scale, bias):
        dims = X.shape
        N = dims[0]
        C = dims[1]
        X = X.reshape(N, C, -1)
        scale = scale.reshape(C, 1)
        bias = bias.reshape(C, 1)
        Y = X * scale + bias
        return [Y.reshape(dims)]

    def affine_channel_nhwc_ref(self, X, scale, bias):
        dims = X.shape
        N = dims[0]
        C = dims[-1]
        X = X.reshape(N, -1, C)
        Y = X * scale + bias
        return [Y.reshape(dims)]

    @serial.given(N=st.integers(1, 5), C=st.integers(1, 5),
            H=st.integers(1, 5), W=st.integers(1, 5),
            order=st.sampled_from(["NCHW", "NHWC"]), is_learnable=st.booleans(),
            in_place=st.booleans(), **hu.gcs)
    def test_affine_channel_2d(
            self, N, C, H, W, order, is_learnable, in_place, gc, dc):
        op = core.CreateOperator(
            "AffineChannel",
            ["X", "scale", "bias"],
            ["X"] if in_place and not is_learnable else ["Y"],
            order=order,
            is_learnable=is_learnable,
        )

        if order == "NCHW":
            X = np.random.randn(N, C, H, W).astype(np.float32)
        else:
            X = np.random.randn(N, H, W, C).astype(np.float32)
        scale = np.random.randn(C).astype(np.float32)
        bias = np.random.randn(C).astype(np.float32)
        inputs = [X, scale, bias]

        def ref_op(X, scale, bias):
            if order == "NCHW":
                return self.affine_channel_nchw_ref(X, scale, bias)
            else:
                return self.affine_channel_nhwc_ref(X, scale, bias)

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=inputs,
            reference=ref_op,
        )
        self.assertDeviceChecks(dc, op, inputs, [0])
        num_grad = len(inputs) if is_learnable else 1
        for i in range(num_grad):
            self.assertGradientChecks(gc, op, inputs, i, [0])

    @given(N=st.integers(1, 5), C=st.integers(1, 5), T=st.integers(1, 3),
           H=st.integers(1, 3), W=st.integers(1, 3),
           order=st.sampled_from(["NCHW", "NHWC"]), is_learnable=st.booleans(),
           in_place=st.booleans(), **hu.gcs)
    @settings(deadline=10000)
    def test_affine_channel_3d(
            self, N, C, T, H, W, order, is_learnable, in_place, gc, dc):
        op = core.CreateOperator(
            "AffineChannel",
            ["X", "scale", "bias"],
            ["X"] if in_place and not is_learnable else ["Y"],
            order=order,
            is_learnable=is_learnable,
        )

        if order == "NCHW":
            X = np.random.randn(N, C, T, H, W).astype(np.float32)
        else:
            X = np.random.randn(N, T, H, W, C).astype(np.float32)
        scale = np.random.randn(C).astype(np.float32)
        bias = np.random.randn(C).astype(np.float32)
        inputs = [X, scale, bias]

        def ref_op(X, scale, bias):
            if order == "NCHW":
                return self.affine_channel_nchw_ref(X, scale, bias)
            else:
                return self.affine_channel_nhwc_ref(X, scale, bias)

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=inputs,
            reference=ref_op,
        )
        self.assertDeviceChecks(dc, op, inputs, [0])
        num_grad = len(inputs) if is_learnable else 1
        for i in range(num_grad):
            self.assertGradientChecks(gc, op, inputs, i, [0])
