




from caffe2.python import brew, core, workspace
from caffe2.python.model_helper import ModelHelper
from functools import partial
from hypothesis import given, settings
from typing import Optional, Tuple

import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st

import numpy as np
import torch

import unittest


def _layer_norm_ref(axis, epsilon, X):
    left = int(np.prod(X.shape[:axis]))
    reshaped = np.reshape(X, [left, -1])
    mean = np.mean(reshaped, axis=1).reshape([left, 1])
    std = np.sqrt(np.mean(np.square(reshaped), axis=1).reshape(
        [left, 1]) - np.square(mean) + epsilon)
    Y = (reshaped - mean) / (std)
    Y = np.reshape(Y, X.shape)
    mean = np.reshape(mean, X.shape[:axis] + (1,))
    std = np.reshape(std, X.shape[:axis] + (1,))
    return (Y, mean, std)


def _layer_norm_with_affine_ref(axis, epsilon, X, gamma, beta):
    Y, mean, std = _layer_norm_ref(axis, epsilon, X)
    Y = Y * gamma + beta
    return (Y, mean, std)


def _layer_norm_grad_ref(axis, gout_full, norm, mean_full, stdev_full, X_full):
    left = int(np.prod(X_full.shape[:axis]))
    right = int(np.prod(X_full.shape[axis:]))
    X = np.reshape(X_full, [left, right])
    stdev = np.reshape(stdev_full, [left, 1])
    mean = np.reshape(mean_full, [left, 1])
    gout = np.reshape(gout_full, [left, right])
    dstdev_end = (-1.0) / np.power(stdev, 2.0) \
        * np.sum((X - mean) * gout, axis=1).reshape([left, 1])
    dmean_end = np.sum(-1.0 / stdev * gout, axis=1).reshape([left, 1])
    dx_end = 1.0 / stdev * gout

    # stdev block
    dmean_stdev = -1.0 * mean / stdev * dstdev_end
    dx_stdev = X / (right * stdev) * dstdev_end

    # mean block
    dmean = dmean_end + dmean_stdev
    dxmean = (1.0 / right) * dmean

    # final outputs
    dx = dx_end + dx_stdev + dxmean
    dx = dx.reshape(X_full.shape)

    return [dx]


class TestLayerNormOp(serial.SerializedTestCase):
    @given(X=hu.tensor(min_dim=2), **hu.gcs)
    @settings(deadline=10000)
    def test_layer_norm_grad_op(self, X, gc, dc):
        axis = np.random.randint(0, len(X.shape))
        epsilon = 1e-4
        op = core.CreateOperator(
            "LayerNormGradient",
            ["gout", "out", "mean", "stdev", "in"],
            ["gin"],
            axis=axis,
            epsilon=epsilon,
        )

        norm, mean, stdev = _layer_norm_ref(axis, epsilon, X)
        gout = norm

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[gout, norm, mean, stdev, X],
            reference=partial(_layer_norm_grad_ref, axis)
        )
        self.assertDeviceChecks(
            device_options=dc,
            op=op,
            inputs=[gout, norm, mean, stdev, X],
            outputs_to_check=[0],
        )

    @given(X=hu.tensor(min_dim=2),
           eps=st.floats(1e-5, 1e-3),
           elementwise_affine=st.booleans(),
           **hu.gcs)
    def test_layer_norm_op(self, X, eps, elementwise_affine, gc, dc):
        axis = np.random.randint(0, len(X.shape))

        op = core.CreateOperator(
            "LayerNorm",
            ["X", "gamma", "beta"] if elementwise_affine else ["X"],
            ["Y", "mean", "std"],
            axis=axis,
            epsilon=eps,
            elementwise_affine=elementwise_affine,
        )

        if elementwise_affine:
            ref = partial(_layer_norm_with_affine_ref, axis, eps)
        else:
            ref = partial(_layer_norm_ref, axis, eps)

        if elementwise_affine:
            gamma = np.random.randn(*X.shape[axis:]).astype(np.float32)
            beta = np.random.randn(*X.shape[axis:]).astype(np.float32)
            inputs = [X, gamma, beta]
        else:
            inputs = [X]

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=inputs,
            reference=ref,
        )
        self.assertDeviceChecks(
            device_options=dc,
            op=op,
            inputs=inputs,
            outputs_to_check=[0, 1, 2],
        )

    @given(M=st.integers(1, 10),
           N=st.integers(10, 20),
           axis=st.integers(0, 1),
           eps=st.floats(1e-5, 1e-3),
           elementwise_affine=st.booleans(),
           **hu.gcs)
    @settings(deadline=10000)
    def test_layer_norm_grad(
            self, M, N, axis, eps, elementwise_affine, gc, dc):
        op = core.CreateOperator(
            "LayerNorm",
            ["X", "gamma", "beta"] if elementwise_affine else ["X"],
            ["Y", "mean", "std"],
            axis=axis,
            epsilon=eps,
            elementwise_affine=elementwise_affine,
        )

        X = np.arange(M * N).astype(np.float32)
        np.random.shuffle(X)
        X = X.reshape((M, N))
        if elementwise_affine:
            gamma = np.random.randn(*X.shape[axis:]).astype(np.float32)
            beta = np.random.randn(*X.shape[axis:]).astype(np.float32)
            inputs = [X, gamma, beta]
        else:
            inputs = [X]

        for i in range(len(inputs)):
            self.assertGradientChecks(gc, op, inputs, i, [0])

    @unittest.skipIf(workspace.has_hip_support,
                     "Operator cross-calling doesn't work with hip yet")
    @given(X=hu.tensor(min_dim=2),
           eps=st.floats(1e-5, 1e-3),
           elementwise_affine=st.booleans(),
           **hu.gcs)
    @settings(deadline=10000)
    def test_layer_norm_op_c10(self, X, eps, elementwise_affine, gc, dc):
        axis = np.random.randint(0, len(X.shape))

        op = core.CreateOperator(
            "C10LayerNorm_DontUseThisOpYet",
            ["X", "gamma", "beta"] if elementwise_affine else ["X"],
            ["Y", "mean", "std"],
            axis=axis,
            epsilon=eps,
            elementwise_affine=elementwise_affine,
        )

        if elementwise_affine:
            ref = partial(_layer_norm_with_affine_ref, axis, eps)
        else:
            ref = partial(_layer_norm_ref, axis, eps)

        if elementwise_affine:
            gamma = np.random.randn(*X.shape[axis:]).astype(np.float32)
            beta = np.random.randn(*X.shape[axis:]).astype(np.float32)
            inputs = [X, gamma, beta]
        else:
            inputs = [X]

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=inputs,
            reference=ref,
        )
        self.assertDeviceChecks(
            device_options=dc,
            op=op,
            inputs=inputs,
            outputs_to_check=[0, 1, 2],
        )

    @unittest.skipIf(workspace.has_hip_support,
                     "Operator cross-calling doesn't work with hip yet")
    @given(X=hu.tensor(min_dim=2),
           eps=st.floats(1e-5, 1e-3),
           elementwise_affine=st.booleans(),
           **hu.gcs)
    def test_layer_norm_op_c10_preallocated_outputs(
            self, X, eps, elementwise_affine, gc, dc):
        # This test case ensures that it works correctly when output tensors are
        # preallocated.
        axis = np.random.randint(0, len(X.shape))

        self.ws.create_blob("X").feed(X)
        if elementwise_affine:
            gamma = np.random.randn(*X.shape[axis:]).astype(np.float32)
            beta = np.random.randn(*X.shape[axis:]).astype(np.float32)
            self.ws.create_blob("gamma").feed(gamma)
            self.ws.create_blob("beta").feed(beta)

        m = ModelHelper(name="test")
        m.net.C10LayerNorm_DontUseThisOpYet(
            ["X", "gamma", "beta"] if elementwise_affine else ["X"],
            ["Y", "mean", "std"],
            axis=axis,
            epsilon=eps,
            elementwise_affine=elementwise_affine,
        )
        self.ws.create_net(m.param_init_net).run()
        net = self.ws.create_net(m.net)
        # run two times to be extra sure that the outputs are preallocated
        net.run()
        net.run()

        if elementwise_affine:
            expected_norm, expected_mean, expected_std = \
                _layer_norm_with_affine_ref(axis, eps, X, gamma, beta)
        else:
            expected_norm, expected_mean, expected_std = _layer_norm_ref(
                axis, eps, X)
        actual_norm = self.ws.fetch_blob('Y')
        actual_mean = self.ws.fetch_blob('mean')
        actual_std = self.ws.fetch_blob('std')

        torch.testing.assert_allclose(
            expected_norm, actual_norm, rtol=1e-4, atol=1e-4)
        torch.testing.assert_allclose(expected_mean, actual_mean)
        torch.testing.assert_allclose(expected_std, actual_std)

    @given(X=hu.tensor(min_dim=2),
           eps=st.floats(1e-5, 1e-3),
           elementwise_affine=st.booleans(),
           **hu.gcs)
    def test_layer_norm_op_pytorch(self, X, eps, elementwise_affine, gc, dc):
        axis = np.random.randint(0, len(X.shape))

        if elementwise_affine:
            gamma = np.random.randn(*X.shape[axis:]).astype(np.float32)
            beta = np.random.randn(*X.shape[axis:]).astype(np.float32)
            expected_norm, expected_mean, expected_std = \
                _layer_norm_with_affine_ref(axis, eps, X, gamma, beta)
            actual_norm, actual_mean, actual_std = torch.ops._caffe2.LayerNorm(
                torch.tensor(X), torch.tensor(gamma), torch.tensor(beta),
                axis, eps, True)
        else:
            expected_norm, expected_mean, expected_std = _layer_norm_ref(
                axis, eps, X)
            actual_norm, actual_mean, actual_std = torch.ops._caffe2.LayerNorm(
                torch.tensor(X), None, None, axis, eps)

        torch.testing.assert_allclose(
            expected_norm, actual_norm, rtol=1e-4, atol=1e-4)
        torch.testing.assert_allclose(expected_mean, actual_mean)
        torch.testing.assert_allclose(expected_std, actual_std)

    # Test case is using workspace.has_cuda_support and not
    # workspace.has_gpu_support to exclude it from HIP because tensor interop
    # doesn't work for HIP tensors yet
    @unittest.skipIf(not workspace.has_cuda_support, "No cuda support")
    @given(X=hu.tensor(min_dim=2),
           eps=st.floats(1e-5, 1e-3),
           elementwise_affine=st.booleans())
    def test_layer_norm_op_pytorch_cuda(self, X, eps, elementwise_affine):
        axis = np.random.randint(0, len(X.shape))

        if elementwise_affine:
            gamma = np.random.randn(*X.shape[axis:]).astype(np.float32)
            beta = np.random.randn(*X.shape[axis:]).astype(np.float32)
            expected_norm, expected_mean, expected_std = \
                _layer_norm_with_affine_ref(axis, eps, X, gamma, beta)
            actual_norm, actual_mean, actual_std = torch.ops._caffe2.LayerNorm(
                torch.tensor(X).cuda(),
                torch.tensor(gamma).cuda(),
                torch.tensor(beta).cuda(),
                axis,
                eps,
                True)
        else:
            expected_norm, expected_mean, expected_std = _layer_norm_ref(
                axis, eps, X)
            actual_norm, actual_mean, actual_std = torch.ops._caffe2.LayerNorm(
                torch.tensor(X).cuda(), None, None, axis, eps)

        torch.testing.assert_allclose(
            expected_norm, actual_norm.cpu(), rtol=1e-4, atol=1e-4)
        torch.testing.assert_allclose(expected_mean, actual_mean.cpu())
        torch.testing.assert_allclose(expected_std, actual_std.cpu())

    @given(X=hu.tensor(min_dim=2),
           eps=st.floats(1e-5, 1e-3),
           elementwise_affine=st.booleans(),
           **hu.gcs)
    @settings(deadline=1000)
    def test_layer_norm_op_jit(self, X, eps, elementwise_affine, gc, dc):
        @torch.jit.script
        def jit_layer_norm(
                X: torch.Tensor,
                gamma: Optional[torch.Tensor] = None,
                beta: Optional[torch.Tensor] = None,
                axis: int = 1,
                eps: float = 1e-5,
                elementwise_affine: bool = False,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return torch.ops._caffe2.LayerNorm(
                X, gamma, beta, axis, eps, elementwise_affine)

        axis = np.random.randint(0, len(X.shape))

        if elementwise_affine:
            gamma = np.random.randn(*X.shape[axis:]).astype(np.float32)
            beta = np.random.randn(*X.shape[axis:]).astype(np.float32)
            expected_norm, expected_mean, expected_std = \
                _layer_norm_with_affine_ref(axis, eps, X, gamma, beta)
            actual_norm, actual_mean, actual_std = jit_layer_norm(
                torch.Tensor(X), torch.tensor(gamma), torch.tensor(beta),
                axis, eps, elementwise_affine)
        else:
            expected_norm, expected_mean, expected_std = _layer_norm_ref(
                axis, eps, X)
            actual_norm, actual_mean, actual_std = jit_layer_norm(
                torch.tensor(X), None, None, axis, eps, elementwise_affine)

        torch.testing.assert_allclose(
            expected_norm, actual_norm, rtol=1e-4, atol=1e-4)
        torch.testing.assert_allclose(expected_mean, actual_mean)
        torch.testing.assert_allclose(expected_std, actual_std)

    @given(X=hu.tensor(min_dim=2), **hu.gcs)
    def test_layer_norm_brew_wrapper(self, X, gc, dc):
        axis = np.random.randint(0, len(X.shape))
        scale_dim = [1] * np.ndim(X)
        scale_dim[axis] = X.shape[axis]

        self.ws.create_blob('input').feed(X)

        model = ModelHelper(name='test_layer_norm_brew_wrapper')
        brew.layer_norm(
            model,
            'input',
            'output',
            dim_in=X.shape[axis:],
            axis=axis,
            epsilon=1e-4,
        )

        self.ws.create_net(model.param_init_net).run()
        self.ws.create_net(model.net).run()

    @given(N=st.integers(1, 10), elementwise_affine=st.booleans(), **hu.gcs)
    @settings(deadline=None)
    def test_layer_norm_with_empty_batch(self, N, elementwise_affine, gc, dc):
        X = np.random.randn(0, N).astype(np.float32)
        gamma = np.random.rand(N).astype(np.float32)
        beta = np.random.rand(N).astype(np.float32)

        op = core.CreateOperator(
            "LayerNorm",
            ["X", "gamma", "beta"] if elementwise_affine else ["X"],
            ["Y", "mean", "sigma"],
            elementwise_affine=elementwise_affine,
        )

        def ref(X, gamma=None, beta=None):
            Y = np.zeros_like(X)
            axis = 1
            mean = np.zeros(X.shape[:axis] + (1,), dtype=X.dtype)
            sigma = np.zeros(X.shape[:axis] + (1,), dtype=X.dtype)
            return Y, mean, sigma


        inputs = [X, gamma, beta] if elementwise_affine else [X]
        self.assertReferenceChecks(gc, op, inputs, ref)
        self.assertDeviceChecks(dc, op, inputs, [0, 1])
        for i in range(len(inputs)):
            self.assertGradientChecks(gc, op, inputs, i, [0])


if __name__ == "__main__":
    unittest.main()
