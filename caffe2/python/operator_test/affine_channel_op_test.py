from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from caffe2.python import core
from hypothesis import given

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st


class TestAffineChannelOp(hu.HypothesisTestCase):
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

    @given(N=st.integers(1, 5), C=st.integers(1, 5), H=st.integers(1, 5),
           W=st.integers(1, 5), order=st.sampled_from(["NCHW", "NHWC"]),
           is_learnable=st.booleans(), engine=st.sampled_from(["", "CUDNN"]),
           in_place=st.booleans(), **hu.gcs)
    def test_affine_channel_2d(
            self, N, C, H, W, order, is_learnable, engine, in_place, gc, dc):
        op = core.CreateOperator(
            "AffineChannel",
            ["X", "scale", "bias"],
            ["X"] if in_place and not is_learnable else ["Y"],
            order=order,
            is_learnable=is_learnable,
            engine=engine,
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
           engine=st.sampled_from(["", "CUDNN"]), in_place=st.booleans(),
           **hu.gcs)
    def test_affine_channel_3d(
            self, N, C, T, H, W, order, is_learnable, engine, in_place, gc, dc):
        op = core.CreateOperator(
            "AffineChannel",
            ["X", "scale", "bias"],
            ["X"] if in_place and not is_learnable else ["Y"],
            order=order,
            is_learnable=is_learnable,
            engine=engine,
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
