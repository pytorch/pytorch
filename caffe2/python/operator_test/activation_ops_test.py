




import numpy as np

from hypothesis import given, assume, settings
import hypothesis.strategies as st

from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.mkl_test_util as mu
import caffe2.python.serialized_test.serialized_test_util as serial

from scipy.stats import norm

import unittest


class TestActivations(serial.SerializedTestCase):
    @given(X=hu.tensor(), in_place=st.booleans(),
                  engine=st.sampled_from(["", "CUDNN"]), **mu.gcs)
    @settings(deadline=10000)
    def test_relu(self, X, in_place, engine, gc, dc):
        if gc == mu.mkl_do:
            in_place = False

        op = core.CreateOperator(
            "Relu",
            ["X"],
            ["X"] if in_place else ["Y"],
            engine=engine,
        )

        def relu_ref(X):
            return [np.maximum(X, 0.0)]

        # go away from the origin point to avoid kink problems
        X += 0.02 * np.sign(X)
        X[X == 0.0] += 0.02

        self.assertReferenceChecks(gc, op, [X], relu_ref, ensure_outputs_are_inferred=True)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0], ensure_outputs_are_inferred=True)

    @given(N=st.integers(1, 10), M=st.integers(1, 10), in_place=st.booleans(),
           **hu.gcs)
    def test_relu_empty_input(self, N, M, in_place, gc, dc):
        op = core.CreateOperator(
            "Relu",
            ["X"],
            ["X"] if in_place else ["Y"],
        )

        def relu_ref(X):
            return [np.maximum(X, 0.0)]

        X = np.random.randn(0, N, M).astype(np.float32)

        self.assertReferenceChecks(gc, op, [X], relu_ref, ensure_outputs_are_inferred=True)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0], ensure_outputs_are_inferred=True)

    @unittest.skipIf(not workspace.has_gpu_support,
                     "Relu for float16 can only run on GPU now.")
    @given(X=hu.tensor(dtype=np.float16), in_place=st.booleans(),
           engine=st.sampled_from(["", "CUDNN"]), **hu.gcs)
    def test_relu_fp16(self, X, in_place, engine, gc, dc):
        # fp16 is only supported on CUDA/HIP
        assume(core.IsGPUDeviceType(gc.device_type))
        op = core.CreateOperator(
            "Relu",
            ["X"],
            ["X"] if in_place else ["Y"],
            engine=engine,
        )

        def relu_ref(X):
            return [np.maximum(X, 0.0)]

        def relu_grad_ref(g_out, outputs, fwd_inputs):
            dY = g_out
            [Y] = outputs
            dX = dY
            dX[Y == 0] = 0
            return [dX]

        # go away from the origin point to avoid kink problems
        X += 0.02 * np.sign(X)
        X[X == 0.0] += 0.02

        self.assertReferenceChecks(
            gc,
            op,
            [X],
            relu_ref,
            output_to_grad="X" if in_place else "Y",
            grad_reference=relu_grad_ref)

    @serial.given(X=hu.tensor(elements=hu.floats(-3.0, 3.0)),
                  n=hu.floats(min_value=0.5, max_value=2.0),
                  in_place=st.booleans(), **hu.gcs)
    def test_relu_n(self, X, n, in_place, gc, dc):
        op = core.CreateOperator(
            "ReluN",
            ["X"],
            ["X"] if in_place else ["Y"],
            n=n,
        )

        def relu_n_ref(X):
            return [np.minimum(np.maximum(X, 0.0), n)]

        # go away from 0 and n to avoid kink problems
        X += 0.04 * np.sign(X)
        X[X == 0.0] += 0.04
        X -= n
        X += 0.02 * np.sign(X)
        X[X == 0.0] -= 0.02
        X += n

        self.assertReferenceChecks(gc, op, [X], relu_n_ref, ensure_outputs_are_inferred=True)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0], stepsize=0.005,
                                  ensure_outputs_are_inferred=True)

    @serial.given(X=hu.tensor(),
                  alpha=hu.floats(min_value=0.1, max_value=2.0),
                  in_place=st.booleans(), engine=st.sampled_from(["", "CUDNN"]),
                  **hu.gcs)
    def test_elu(self, X, alpha, in_place, engine, gc, dc):
        op = core.CreateOperator(
            "Elu",
            ["X"],
            ["X"] if in_place else ["Y"],
            alpha=alpha,
            engine=engine,
        )

        def elu_ref(X):
            Y = X
            Y[X < 0] = alpha * (np.exp(X[X < 0]) - 1.0)
            return [Y]

        # go away from the origin point to avoid kink problems
        X += 0.04 * np.sign(X)
        X[X == 0.0] += 0.04

        self.assertReferenceChecks(gc, op, [X], elu_ref, ensure_outputs_are_inferred=True)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0], stepsize=1e-2, ensure_outputs_are_inferred=True)

    @given(X=hu.tensor(min_dim=4, max_dim=4),
           alpha=hu.floats(min_value=0.1, max_value=2.0),
           inplace=st.booleans(),
           shared=st.booleans(),
           order=st.sampled_from(["NCHW", "NHWC"]),
           seed=st.sampled_from([20, 100]),
           **hu.gcs)
    @settings(deadline=10000)
    def test_prelu(self, X, alpha, inplace, shared, order, seed, gc, dc):
        np.random.seed(seed)
        W = np.random.randn(
            X.shape[1] if order == "NCHW" else X.shape[3]).astype(np.float32)

        if shared:
            W = np.random.randn(1).astype(np.float32)

        # go away from the origin point to avoid kink problems
        X += 0.04 * np.sign(X)
        X[X == 0.0] += 0.04

        def prelu_ref(X, W):
            Y = X.copy()
            W = W.reshape(1, -1, 1, 1) if order == "NCHW" \
                else W.reshape(1, 1, 1, -1)
            assert len(X.shape) == 4
            neg_indices = X <= 0
            assert len(neg_indices.shape) == 4
            assert X.shape == neg_indices.shape
            Y[neg_indices] = (Y * W)[neg_indices]
            return (Y,)

        op = core.CreateOperator(
            "PRelu", ["X", "W"], ["Y" if not inplace else "X"],
            alpha=alpha, order=order)
        self.assertReferenceChecks(gc, op, [X, W], prelu_ref, ensure_outputs_are_inferred=True)
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [X, W], [0])

        if not inplace:
            # Gradient check wrt X
            self.assertGradientChecks(gc, op, [X, W], 0, [0], stepsize=1e-2, ensure_outputs_are_inferred=True)
            # Gradient check wrt W
            self.assertGradientChecks(gc, op, [X, W], 1, [0], stepsize=1e-2, ensure_outputs_are_inferred=True)

    @serial.given(X=hu.tensor(),
                  alpha=hu.floats(min_value=0.1, max_value=2.0),
                  inplace=st.booleans(),
                  **hu.gcs)
    def test_leaky_relu(self, X, alpha, inplace, gc, dc):
        # go away from the origin point to avoid kink problems
        X += 0.04 * np.sign(X)
        X[X == 0.0] += 0.04

        def leaky_relu_ref(X):
            Y = X.copy()
            neg_indices = X <= 0
            Y[neg_indices] = Y[neg_indices] * alpha
            return (Y,)

        op = core.CreateOperator(
            "LeakyRelu",
            ["X"], ["Y" if not inplace else "X"],
            alpha=alpha)
        self.assertReferenceChecks(gc, op, [X], leaky_relu_ref,
                                   ensure_outputs_are_inferred=True)
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [X], [0])

    @given(X=hu.tensor(),
           inplace=st.booleans(),
           **hu.gcs)
    def test_leaky_relu_default(self, X, inplace, gc, dc):
        # go away from the origin point to avoid kink problems
        X += 0.04 * np.sign(X)
        X[X == 0.0] += 0.04

        def leaky_relu_ref(X):
            Y = X.copy()
            neg_indices = X <= 0
            Y[neg_indices] = Y[neg_indices] * 0.01
            return (Y,)

        op = core.CreateOperator(
            "LeakyRelu",
            ["X"], ["Y" if not inplace else "X"])
        self.assertReferenceChecks(gc, op, [X], leaky_relu_ref)
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [X], [0])

    @given(X=hu.tensor(),
           fast_gelu=st.booleans(),
           **hu.gcs)
    @settings(deadline=1000)
    def test_gelu(self, X, fast_gelu, gc, dc):
        op = core.CreateOperator(
            "Gelu",
            ["X"],
            ["Y"],
            fast_gelu=fast_gelu,
        )

        def gelu_ref(X):
            return (X * norm.cdf(X),)

        tol = 1e-3 if fast_gelu else 1e-4
        self.assertReferenceChecks(gc, op, [X], gelu_ref, threshold=tol,
                                   ensure_outputs_are_inferred=True)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0],
                                  ensure_outputs_are_inferred=True)


    @given(n=st.integers(0, 6), m=st.integers(4, 6),
           seed=st.integers(0, 1000), **hu.gcs_cpu_only)
    def test_mish(self, n, m, gc, dc, seed):
        np.random.seed(seed)
        X = np.random.rand(n, m).astype(np.float32)

        def mish_ref(X):
            return (X * np.tanh(np.log1p(np.exp(X))),)

        op = core.CreateOperator(
            "Mish",
            ["X"],
            ["Y"]
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=mish_ref,
            ensure_outputs_are_inferred=True,
        )

        self.assertGradientChecks(
            gc, op, [X], 0, [0], ensure_outputs_are_inferred=True)


if __name__ == "__main__":
    unittest.main()
