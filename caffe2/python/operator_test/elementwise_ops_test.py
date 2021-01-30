




from caffe2.python import core, workspace
from hypothesis import given, assume, settings
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np

import unittest

class TestElementwiseOps(hu.HypothesisTestCase):

    @given(X=hu.tensor(dtype=np.float32), **hu.gcs)
    @settings(deadline=10000)
    def test_abs(self, X, gc, dc):
        op = core.CreateOperator(
            "Abs",
            ["X"],
            ["Y"],
        )

        def abs_ref(X):
            return [np.absolute(X)]

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=abs_ref,
            ensure_outputs_are_inferred=True,
        )
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0], ensure_outputs_are_inferred=True)

    @given(X=hu.tensor(dtype=np.float32), inplace=st.booleans(), **hu.gcs)
    @settings(deadline=10000)
    def test_exp(self, X, inplace, gc, dc):
        op = core.CreateOperator(
            "Exp",
            ["X"],
            ["X"] if inplace else ["Y"],
        )

        def exp_ref(X):
            return [np.exp(X)]

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=exp_ref,
            ensure_outputs_are_inferred=True,
        )
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0], ensure_outputs_are_inferred=True)

    @given(n=st.integers(0, 6), m=st.integers(4, 6),
           seed=st.integers(0, 1000), **hu.gcs)
    @settings(deadline=1000)
    def test_log(self, n, m, gc, dc, seed):
        np.random.seed(seed)
        X = np.random.rand(n, m).astype(np.float32) + 1.0

        def log_op(X):
            return [np.log(X)]

        op = core.CreateOperator(
            "Log",
            ["X"],
            ["Z"]
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=log_op,
            ensure_outputs_are_inferred=True,
        )

        self.assertGradientChecks(
            gc, op, [X], 0, [0], stepsize=1e-4, threshold=1e-2,
            ensure_outputs_are_inferred=True)

    @given(n=st.integers(0, 10), m=st.integers(4, 6),
           d=st.integers(2, 3), seed=st.integers(0, 1000), **hu.gcs)
    @settings(deadline=10000)
    def test_powt(self, n, m, d, gc, dc, seed):
        np.random.seed(seed)
        X = np.random.rand(n, m, d).astype(np.float32) + 1.0
        Y = np.random.rand(n, m, d).astype(np.float32) + 2.0

        def powt_op(X, Y):
            return [np.power(X, Y)]

        #two gradients Y*X^(Y-1) and X^Y * ln(X)
        def powt_grad(g_out, outputs, fwd_inputs):
            [X, Y] = fwd_inputs
            Z = outputs[0]
            return ([Y * np.power(X, Y - 1), Z * np.log(X)] * g_out)

        op = core.CreateOperator(
            "Pow",
            ["X", "Y"],
            ["Z"]
        )

        self.assertReferenceChecks(device_option=gc,
                                   op=op,
                                   inputs=[X, Y],
                                   reference=powt_op,
                                   output_to_grad="Z",
                                   grad_reference=powt_grad,
                                   ensure_outputs_are_inferred=True)

    @given(n=st.integers(0, 6), m=st.integers(4, 6),
           seed=st.integers(0, 1000), **hu.gcs)
    @settings(deadline=10000)
    def test_sqr(self, n, m, gc, dc, seed):
        np.random.seed(seed)
        X = np.random.rand(n, m).astype(np.float32)

        def sqr_op(X):
            return [np.square(X)]

        op = core.CreateOperator(
            "Sqr",
            ["X"],
            ["Z"]
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=sqr_op,
            ensure_outputs_are_inferred=True,
        )

        self.assertGradientChecks(
            gc, op, [X], 0, [0], stepsize=1e-4, threshold=1e-2,
            ensure_outputs_are_inferred=True)

    @given(
        X=hu.tensor(
            elements=hu.floats(min_value=0.1, max_value=10),
            # allow empty tensor
            min_value=0),
        inplace=st.booleans(),
        **hu.gcs
    )
    @settings(deadline=10000)
    def test_sqrt(self, X, inplace, gc, dc):
        def sqrt_op(X):
            return [np.sqrt(X)]

        op = core.CreateOperator(
            "Sqrt",
            ["X"],
            ["X"] if inplace else ["Y"]
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=sqrt_op,
            ensure_outputs_are_inferred=True,
        )
        self.assertDeviceChecks(dc, op, [X], [0])
        # stepsize need to be smaller than the possible minimum X, so the
        # sqrt is well defined
        self.assertGradientChecks(
            gc, op, [X], 0, [0], stepsize=1e-2, ensure_outputs_are_inferred=True)

    @given(X=hu.tensor(dtype=np.float32), inplace=st.booleans(), **hu.gcs)
    @settings(deadline=10000)
    def test_softsign(self, X, inplace, gc, dc):
        op = core.CreateOperator(
            "Softsign",
            ["X"],
            ["X"] if inplace else ["Y"],
        )

        def softsign_ref(X):
            return [X / (1.0 + np.absolute(X))]

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=softsign_ref,
            ensure_outputs_are_inferred=True,
        )
        self.assertDeviceChecks(dc, op, [X], [0])
        if not inplace:
            self.assertGradientChecks(
                gc, op, [X], 0, [0],
                ensure_outputs_are_inferred=True,
            )

    @given(X=hu.tensor(elements=hu.floats(min_value=0.1, max_value=10.0), dtype=np.float32),
           inplace=st.booleans(), **hu.gcs)
    @settings(deadline=10000)
    def test_rsqrt(self, X, inplace, gc, dc):
        op = core.CreateOperator(
            "Rsqrt",
            ["X"],
            ["X"] if inplace else ["Y"],
        )

        def rsqrt_ref(X):
            return [1.0 / np.sqrt(X)]

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=rsqrt_ref,
            ensure_outputs_are_inferred=True,
        )
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(
            gc, op, [X], 0, [0], stepsize=5e-3,
            ensure_outputs_are_inferred=True,
        )

    @given(X=hu.tensor(dtype=np.float32), **hu.gcs)
    @settings(deadline=10000)
    def test_cube(self, X, gc, dc):
        op = core.CreateOperator(
            "Cube",
            ["X"],
            ["Y"],
        )

        def cube_ref(X):
            return [np.power(X, 3)]

        def cube_grad_ref(g_out, outputs, fwd_inputs):
            dY = g_out
            [X] = fwd_inputs
            return [dY * np.square(X) * 3]

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=cube_ref,
            output_to_grad="Y",
            grad_reference=cube_grad_ref,
            ensure_outputs_are_inferred=True,
        )
        self.assertDeviceChecks(dc, op, [X], [0])

    @given(X=hu.tensor(dtype=np.float32), in_place=st.booleans(), **hu.gcs)
    @settings(deadline=10000)
    def test_cbrt(self, X, in_place, gc, dc):
        op = core.CreateOperator(
            "Cbrt",
            ["X"],
            ["X"] if in_place else ["Y"],
        )

        def cbrt_ref(X):
            return [np.cbrt(X)]

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=cbrt_ref,
            ensure_outputs_are_inferred=True,
        )

    @given(X=hu.tensor(elements=hu.floats(min_value=1.0, max_value=10.0), dtype=np.float32),
           in_place=st.booleans(), **hu.gcs)
    @settings(deadline=10000)
    def test_cbrt_grad(self, X, in_place, gc, dc):
        op = core.CreateOperator(
            "Cbrt",
            ["X"],
            ["X"] if in_place else ["Y"],
        )

        self.assertGradientChecks(
            gc, op, [X], 0, [0],
            ensure_outputs_are_inferred=True,
        )
        self.assertGradientChecks(
            gc, op, [-X], 0, [0],
            ensure_outputs_are_inferred=True,
        )


    @given(n=st.integers(0, 6), m=st.integers(4, 6),
           seed=st.integers(0, 1000), **hu.gcs)
    @settings(deadline=10000)
    def test_swish(self, n, m, gc, dc, seed):
        np.random.seed(seed)
        X = np.random.rand(n, m).astype(np.float32)

        def swish(X):
            return [np.divide(X, (1. + np.exp(-X)))]

        op = core.CreateOperator(
            "Swish",
            ["X"],
            ["Z"]
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=swish,
            ensure_outputs_are_inferred=True,
        )

        self.assertGradientChecks(
            gc, op, [X], 0, [0], stepsize=1e-4, threshold=1e-2,
            ensure_outputs_are_inferred=True)

    @given(n=st.integers(0, 6), m=st.integers(4, 6),
           seed=st.integers(0, 1000), **hu.gcs)
    @settings(deadline=1000)
    def test_swish_gradient_inplace(self, n, m, gc, dc, seed):
        np.random.seed(seed)

        def swish(X):
            return [np.divide(X, (1. + np.exp(-X)))]

        def swish_gradient(X, Y, dY):
            return [dY * (Y + np.divide(1. - Y, 1. + np.exp(-X)))]

        X = np.random.rand(n, m).astype(np.float32)
        Y = swish(X)[0]
        dY = np.random.rand(n, m).astype(np.float32)
        op = core.CreateOperator(
            "SwishGradient",
            ["X", "Y", "grad"],
            "grad"
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X, Y, dY],
            reference=swish_gradient,
        )

    @given(X=hu.tensor(dtype=np.float32), inplace=st.booleans(),
           engine=st.sampled_from(["", "CUDNN"]), **hu.gcs)
    @settings(deadline=1000)
    def test_sigmoid(self, X, inplace, engine, gc, dc):
        op = core.CreateOperator(
            "Sigmoid",
            ["X"],
            ["X"] if inplace else ["Y"],
            engine=engine,
        )

        def sigmoid_ref(X):
            return [1.0 / (1.0 + np.exp(-X))]

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=sigmoid_ref,
            ensure_outputs_are_inferred=True,
        )
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0], ensure_outputs_are_inferred=True)

    @given(X=hu.tensor(dtype=np.float32),
           inplace=st.booleans(),
           alpha=hu.floats(min_value=-100.0, max_value=100.0),
           beta=hu.floats(min_value=-100.0, max_value=100.0),
           engine=st.sampled_from([""]),
           **hu.gcs)
    @settings(deadline=10000)
    def test_hard_sigmoid(self, X, inplace, alpha, beta, engine, gc, dc):
        # Prevent alpha and beta from mutually being 0 to avoid a division
        # error when adjusting our inputs
        assume(alpha != 0.0 or beta != 0.0)
        op = core.CreateOperator(
            "HardSigmoid",
            ["X"],
            ["X"] if inplace else ["Y"],
            alpha=alpha,
            beta=beta,
            engine=engine,
        )

        def hard_sigmoid_ref(X):
            return [np.minimum(1.0, np.maximum(0.0, X * alpha + beta))]

        # Adjust inputs to avoid differentitating at inflection points
        if abs(alpha) > 0.001:
            Y = X * alpha + beta
            Y += 0.04 * np.sign(Y)
            Y[Y == 0.0] += 0.1
            Y[Y == 1.0] -= 0.1
            X = (Y - beta) / alpha

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=hard_sigmoid_ref,
            ensure_outputs_are_inferred=True,
        )
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(
            gc, op, [X], 0, [0], stepsize=1e-4, threshold=1e-2,
            ensure_outputs_are_inferred=True)

    @given(n=st.integers(0, 6), m=st.integers(4, 6), **hu.gcs)
    @settings(deadline=10000)
    def test_eq(self, n, m, gc, dc):
        # Set broadcast and no axis, i.e. broadcasting last dimensions.
        X = np.random.randint(2, size=(n, m))
        Y = np.random.randint(2, size=(n, m))
        op = core.CreateOperator("EQ", ["X", "Y"], "out", broadcast=1)

        def eq(X, Y):
            return [X == Y]

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X, Y],
            reference=eq,
            ensure_outputs_are_inferred=True,
        )

        workspace.FeedBlob('X', X)
        workspace.FeedBlob('Y', Y)

        net = core.Net("batch_bucket_one_hot_test")
        result = net.EQ(["X", "Y"], 1)
        (shapes, types) = workspace.InferShapesAndTypes([net])
        workspace.RunNetOnce(net)

        self.assertEqual(shapes[result], list(workspace.blobs[result].shape))
        self.assertEqual(shapes[result], list(X.shape))
        self.assertEqual(types[result], core.DataType.BOOL)

    @given(n=st.integers(0, 6), m=st.integers(4, 6), **hu.gcs)
    @settings(deadline=10000)
    def test_eq_bcast(self, n, m, gc, dc):
        # Set broadcast and no axis, i.e. broadcasting last dimensions.
        X = np.random.randint(2, size=(n, m))
        Y = np.random.randint(2, size=(m,))
        op = core.CreateOperator("EQ", ["X", "Y"], "out", broadcast=1)

        def eq(X, Y):
            return [X == Y]

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X, Y],
            reference=eq,
            ensure_outputs_are_inferred=True,
        )

        workspace.FeedBlob('X', X)
        workspace.FeedBlob('Y', Y)

        net = core.Net("eq_bast")
        result = net.EQ(["X", "Y"], 1, broadcast=1)
        (shapes, types) = workspace.InferShapesAndTypes([net])
        workspace.RunNetOnce(net)
        self.assertTrue(str(result) in shapes)
        self.assertEqual(shapes[result], list(workspace.blobs[result].shape))
        self.assertEqual(shapes[result], list(X.shape))
        self.assertEqual(types[result], core.DataType.BOOL)

        net_2 = core.Net("eq_bast_invalid")
        result_2 = net_2.EQ(["X", "Y"], 1)
        (shapes, types) = workspace.InferShapesAndTypes([net])
        self.assertTrue(str(result_2) not in shapes)

    def _run_single_test(
            self, op, ref, A, B, reverse_inputs, test_grad, gc, dc):
        inputs = [A, B]
        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=inputs,
            reference=ref,
            ensure_outputs_are_inferred=True,
        )
        self.assertDeviceChecks(dc, op, inputs, [0])
        if test_grad:
            for i in range(len(inputs)):
                self.assertGradientChecks(
                    gc, op, inputs, i, [0],
                    ensure_outputs_are_inferred=True,
                )

        if reverse_inputs:
            inputs = [B, A]
            self.assertReferenceChecks(
                device_option=gc,
                op=op,
                inputs=inputs,
                reference=ref,
                ensure_outputs_are_inferred=True,
           )
            self.assertDeviceChecks(dc, op, inputs, [0])
            if test_grad:
                for i in range(len(inputs)):
                    self.assertGradientChecks(
                        gc, op, inputs, i, [0],
                        ensure_outputs_are_inferred=True,
                    )

    def _test_binary_op(
            self, op_name, np_ref, n, m, k, t, bias, test_grad, gc, dc):
        op = core.CreateOperator(
            op_name,
            ["A", "B"],
            ["C"],
        )

        def ref(A, B):
            return [np_ref(A, B)]

        A = np.random.rand(n, m, k, t).astype(np.float32) + bias
        B = np.random.rand(n, m, k, t).astype(np.float32) + bias
        self._run_single_test(op, ref, A, B, True, test_grad, gc, dc)

        A = np.random.rand(1).astype(np.float32) + bias
        B = np.random.rand(n, m, k, t).astype(np.float32) + bias
        self._run_single_test(op, ref, A, B, True, test_grad, gc, dc)

        A = np.random.rand(k, t).astype(np.float32) + bias
        B = np.random.rand(n, m, k, t).astype(np.float32) + bias
        self._run_single_test(op, ref, A, B, True, test_grad, gc, dc)

        A = np.random.rand(n, m, 1, 1).astype(np.float32) + bias
        B = np.random.rand(n, m, k, t).astype(np.float32) + bias
        self._run_single_test(op, ref, A, B, True, test_grad, gc, dc)

        A = np.random.rand(1, m, k, 1).astype(np.float32) + bias
        B = np.random.rand(n, m, k, t).astype(np.float32) + bias
        self._run_single_test(op, ref, A, B, True, test_grad, gc, dc)

        A = np.random.rand(m, 1, t).astype(np.float32) + bias
        B = np.random.rand(n, m, k, t).astype(np.float32) + bias
        self._run_single_test(op, ref, A, B, True, test_grad, gc, dc)

        A = np.random.rand(1, m, 1, t).astype(np.float32) + bias
        B = np.random.rand(n, 1, k, 1).astype(np.float32) + bias
        self._run_single_test(op, ref, A, B, True, test_grad, gc, dc)

    def _test_binary_op_in_place(
            self, op_name, np_ref, n, m, bias, test_grad, in_place_2nd, gc, dc):
        def ref(A, B):
            return [np_ref(A, B)]

        op = core.CreateOperator(
            op_name,
            ["A", "B"],
            ["A"],
        )
        A = np.random.rand(n, m).astype(np.float32) + bias
        B = np.random.rand(m).astype(np.float32) + bias

        self._run_single_test(op, ref, A, B, False, test_grad, gc, dc)
        A = np.random.rand(n, m).astype(np.float32) + bias
        B = np.random.rand(n, 1).astype(np.float32) + bias
        self._run_single_test(op, ref, A, B, False, test_grad, gc, dc)

        if in_place_2nd:
            op = core.CreateOperator(
                op_name,
                ["A", "B"],
                ["B"],
            )
            A = np.random.rand(m).astype(np.float32) + bias
            B = np.random.rand(n, m).astype(np.float32) + bias
            self._run_single_test(op, ref, A, B, False, test_grad, gc, dc)
            A = np.random.rand(n, 1).astype(np.float32) + bias
            B = np.random.rand(n, m).astype(np.float32) + bias
            self._run_single_test(op, ref, A, B, False, test_grad, gc, dc)

    @given(n=st.integers(0, 5), m=st.integers(0, 5), k=st.integers(0, 5),
           t=st.integers(0, 5), **hu.gcs)
    @settings(deadline=None, max_examples=50)
    def test_add(self, n, m, k, t, gc, dc):
        self._test_binary_op("Add", np.add, n, m, k, t, -0.5, True, gc, dc)
        self._test_binary_op_in_place(
            "Add", np.add, n, m, -0.5, True, True, gc, dc)

    @given(n=st.integers(0, 5), m=st.integers(0, 5), k=st.integers(0, 5),
           t=st.integers(0, 5), **hu.gcs)
    @settings(deadline=None, max_examples=50)
    def test_sub(self, n, m, k, t, gc, dc):
        self._test_binary_op("Sub", np.subtract, n, m,
                             k, t, -0.5, True, gc, dc)
        self._test_binary_op_in_place(
            "Sub", np.subtract, n, m, -0.5, True, True, gc, dc)

    @given(n=st.integers(0, 5), m=st.integers(0, 5), k=st.integers(0, 5),
           t=st.integers(0, 5), **hu.gcs)
    @settings(deadline=None, max_examples=50)
    def test_mul(self, n, m, k, t, gc, dc):
        self._test_binary_op("Mul", np.multiply, n, m,
                             k, t, -0.5, True, gc, dc)

    @given(n=st.integers(0, 5), m=st.integers(0, 5), k=st.integers(0, 5),
           t=st.integers(0, 5), **hu.gcs)
    @settings(deadline=None, max_examples=50)
    def test_div(self, n, m, k, t, gc, dc):
        self._test_binary_op("Div", np.divide, n, m, k, t, 1.0, True, gc, dc)
        self._test_binary_op_in_place(
            "Div", np.divide, n, m, 1.0, True, False, gc, dc)

    @given(n=st.integers(1, 5), m=st.integers(1, 5), broadcast=st.booleans(),
           **hu.gcs)
    @settings(deadline=10000)
    def test_div_legacy_grad(self, n, m, broadcast, gc, dc):
        op = core.CreateOperator(
            "DivGradient",
            ["B", "C", "dC"],
            ["dA", "dB"],
        )

        def div_grad_ref(B, C, dC):
            dA = dC / B
            dB = -dC * C / B
            if broadcast:
                dB = np.sum(dB, axis=0)
            return [dA, dB]

        if broadcast:
            B = np.random.rand(m).astype(np.float32) + 1.0
        else:
            B = np.random.rand(n, m).astype(np.float32) + 1.0
        C = np.random.randn(n, m).astype(np.float32)
        dC = np.random.randn(n, m).astype(np.float32)
        inputs = [B, C, dC]
        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=inputs,
            reference=div_grad_ref,
        )
        self.assertDeviceChecks(dc, op, inputs, [0, 1])

    def _test_bitwise_binary_op(self, op_name, np_ref, n, m, k, t, gc, dc):
        op = core.CreateOperator(
            op_name,
            ["A", "B"],
            ["C"],
        )

        def ref(A, B):
            return [np_ref(A, B)]

        A = np.random.randint(128, size=(n, m, k, t))
        B = np.random.randint(128, size=(n, m, k, t))
        self._run_single_test(op, ref, A, B, True, False, gc, dc)

        A = np.random.randint(128, size=1)
        B = np.random.randint(128, size=(n, m, k, t))
        self._run_single_test(op, ref, A, B, True, False, gc, dc)

        A = np.random.randint(128, size=(k, t))
        B = np.random.randint(128, size=(n, m, k, t))
        self._run_single_test(op, ref, A, B, True, False, gc, dc)

        A = np.random.randint(128, size=(n, m, 1, 1))
        B = np.random.randint(128, size=(n, m, k, t))
        self._run_single_test(op, ref, A, B, True, False, gc, dc)

        A = np.random.randint(128, size=(1, m, k, 1))
        B = np.random.randint(128, size=(n, m, k, t))
        self._run_single_test(op, ref, A, B, True, False, gc, dc)

        A = np.random.randint(128, size=(m, 1, t))
        B = np.random.randint(128, size=(n, m, k, t))
        self._run_single_test(op, ref, A, B, True, False, gc, dc)

        A = np.random.randint(128, size=(1, m, 1, t))
        B = np.random.randint(128, size=(n, 1, k, 1))
        self._run_single_test(op, ref, A, B, True, False, gc, dc)

    @given(n=st.integers(1, 5), m=st.integers(1, 5), k=st.integers(1, 5),
           t=st.integers(1, 5), **hu.gcs)
    @settings(deadline=10000)
    def test_bitwise_and(self, n, m, k, t, gc, dc):
        self._test_bitwise_binary_op(
            "BitwiseAnd", np.bitwise_and, n, m, k, t, gc, dc)

    @given(n=st.integers(1, 5), m=st.integers(1, 5), k=st.integers(1, 5),
           t=st.integers(1, 5), **hu.gcs)
    @settings(deadline=10000)
    def test_bitwise_or(self, n, m, k, t, gc, dc):
        self._test_bitwise_binary_op(
            "BitwiseOr", np.bitwise_or, n, m, k, t, gc, dc)

    @given(n=st.integers(1, 5), m=st.integers(1, 5), k=st.integers(1, 5),
           t=st.integers(1, 5), **hu.gcs)
    @settings(deadline=10000)
    def test_bitwise_xor(self, n, m, k, t, gc, dc):
        self._test_bitwise_binary_op(
            "BitwiseXor", np.bitwise_xor, n, m, k, t, gc, dc)

    @given(X=hu.tensor(elements=hu.floats(min_value=0.5, max_value=2), dtype=np.float32),
           inplace=st.booleans(), **hu.gcs)
    @settings(deadline=10000)
    def test_reciprocal(self, X, inplace, gc, dc):
        def reciprocal_op(X):
            return [np.reciprocal(X)]

        op = core.CreateOperator(
            "Reciprocal",
            ["X"],
            ["X"] if inplace else ["Y"]
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=reciprocal_op,
            ensure_outputs_are_inferred=True,
        )
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(
            gc, op, [X], 0, [0], stepsize=1e-3, threshold=0.05,
            ensure_outputs_are_inferred=True)

    @given(X=hu.tensor(dtype=np.bool), **hu.gcs)
    @settings(deadline=10000)
    def test_not(self, X, gc, dc):
        def not_op(X):
            return [np.logical_not(X)]

        op = core.CreateOperator(
            "Not",
            ["X"],
            ["Y"],
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=not_op,
            ensure_outputs_are_inferred=True,
        )
        self.assertDeviceChecks(dc, op, [X], [0])


if __name__ == "__main__":
    unittest.main()
