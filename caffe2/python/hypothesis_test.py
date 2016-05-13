from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from hypothesis import given, assume
import hypothesis.strategies as st

import unittest

from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu

core.GlobalInit(["python"])


class TestOperators(hu.HypothesisTestCase):
    @given(inputs=hu.tensors(n=2), in_place=st.booleans(), **hu.gcs)
    def test_sum(self, inputs, in_place, gc, dc):
        op = core.CreateOperator("Sum", ["X1", "X2"],
                                        ["Y" if not in_place else "X1"])
        X1, X2 = inputs
        self.assertDeviceChecks(dc, op, [X1, X2], [0])
        self.assertGradientChecks(gc, op, [X1, X2], 0, [0])

    @given(X=hu.tensor(), **hu.gcs)
    def test_relu(self, X, gc, dc):
        op = core.CreateOperator("Relu", ["X"], ["Y"])
        # go away from the origin point to avoid kink problems
        X += 0.02 * np.sign(X)
        X[X == 0.0] += 0.02
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0])

    @unittest.skipIf(not workspace.has_gpu_support,
                     "Recurrent only implemented on GPU")
    @given(hidden_size=st.integers(min_value=1, max_value=3),
           num_layers=st.integers(min_value=1, max_value=3),
           bidirectional=st.booleans(),
           rnn_mode=st.sampled_from(["gru", "lstm"]),
           input_mode=st.sampled_from(["linear"]),
           dropout=st.floats(min_value=0.0, max_value=0.0),
           T=st.integers(min_value=1, max_value=4),
           N=st.integers(min_value=1, max_value=4),
           D=st.integers(min_value=1, max_value=4))
    def test_recurrent(self, hidden_size, num_layers, bidirectional, rnn_mode,
                       input_mode, dropout, T, N, D):
        init_op = core.CreateOperator("RecurrentInit",
            ["INPUT"],
            ["WEIGHT", "DROPOUT_STATES"],
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            rnn_mode=rnn_mode,
            dropout=dropout,
            input_mode=input_mode,
            num_layers=num_layers,
            device_option=hu.gpu_do)

        op = core.CreateOperator("Recurrent",
            ["INPUT", "HIDDEN_INPUT", "CELL_INPUT", "WEIGHT"],
            ["OUTPUT", "HIDDEN_OUTPUT", "CELL_OUTPUT",
             "RNN_SCRATCH", "DROPOUT_STATES"],
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            rnn_mode=rnn_mode,
            dropout=dropout,
            input_mode=input_mode,
            num_layers=num_layers)
        num_directions = 2 if bidirectional else 1
        X = np.random.randn(T, N, D).astype(np.float32)
        workspace.FeedBlob("INPUT", X, device_option=hu.gpu_do)
        workspace.RunOperatorOnce(init_op)
        W = workspace.FetchBlob("WEIGHT")
        H = np.random.randn(
            hidden_size, N, num_layers * num_directions).astype(
                np.float32)
        C = np.random.randn(
            hidden_size, N, num_layers * num_directions).astype(
                np.float32) if rnn_mode == "lstm" else \
            np.empty((1,)).astype(np.float32)  # unused in GRU
        inputs = [X, H, C, W]
        input_idxs = [i for (i, _) in enumerate(inputs)] \
            if rnn_mode == "lstm" else [0, 1, 3]  # ignore C
        for input_idx in input_idxs:
            self.assertGradientChecks(
                hu.gpu_do, op, inputs, input_idx, [0, 1, 2])

    @given(stride=st.integers(1, 3),
           pad=st.integers(0, 3),
           kernel=st.integers(1, 5),
           size=st.integers(7, 10),
           input_channels=st.integers(1, 8),
           output_channels=st.integers(1, 8),
           batch_size=st.integers(1, 3),
           order=st.sampled_from(["NCHW", "NHWC"]),
           engine=st.sampled_from(["", "CUDNN"]), **hu.gcs)
    def test_convolution_gradients(self, stride, pad, kernel, size,
                                   input_channels, output_channels, batch_size,
                                   order, engine, gc, dc):
        assume(stride <= kernel)
        op = core.CreateOperator("Conv",
            ["X", "w", "b"],
            ["Y"],
            stride=stride,
            kernel=kernel,
            pad=pad,
            order=order,
            engine=engine,
        )
        X = np.random.rand(
            batch_size, size, size, input_channels).astype(np.float32) - 0.5
        w = np.random.rand(
            output_channels, kernel, kernel, input_channels).astype(np.float32) - 0.5
        b = np.random.rand(output_channels).astype(np.float32) - 0.5
        if order == "NCHW":
            X = X.transpose((0, 3, 1, 2))
            w = w.transpose((0, 3, 1, 2))

        self.assertDeviceChecks(dc, op, [X, w, b], [0])
        for i in range(3):
            self.assertGradientChecks(gc, op, [X, w, b], i, [0])

    @given(stride=st.integers(1, 3),
           pad=st.integers(0, 3),
           kernel=st.integers(1, 5),
           size=st.integers(7, 10),
           input_channels=st.integers(1, 8),
           output_channels=st.integers(1, 8),
           batch_size=st.integers(1, 3),
           engine=st.sampled_from(["", "CUDNN"]), **hu.gcs)
    def test_convolution_layout(self, stride, pad, kernel, size,
                                input_channels, output_channels, batch_size,
                                engine, gc, dc):
        assume(stride <= kernel)
        X = np.random.rand(
            batch_size, size, size, input_channels).astype(np.float32) - 0.5
        w = np.random.rand(
            output_channels, kernel, kernel, input_channels).astype(np.float32) - 0.5
        b = np.random.rand(output_channels).astype(np.float32) - 0.5
        outputs = {}
        for order in ["NCHW", "NHWC"]:
            op = core.CreateOperator("Conv",
                ["X", "w", "b"],
                ["Y"],
                stride=stride,
                kernel=kernel,
                pad=pad,
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
            workspace.FeedBlob("X", X_f, device_option=gc)
            workspace.FeedBlob("w", w_f, device_option=gc)
            workspace.FeedBlob("b", b, device_option=gc)
            workspace.RunOperatorOnce(op)
            outputs[order] = workspace.FetchBlob("Y")
        np.testing.assert_allclose(
            outputs["NCHW"],
            outputs["NHWC"].transpose((0, 3, 1, 2)),
            atol=1e-4,
            rtol=1e-4)

    @given(inputs=hu.tensors(n=3),
           in_place=st.booleans(),
           beta1=st.floats(min_value=0.1, max_value=0.9),
           beta2=st.floats(min_value=0.1, max_value=0.9),
           lr=st.floats(min_value=0.1, max_value=0.9),
           iters=st.integers(min_value=1, max_value=10000),
           epsilon=st.floats(min_value=1e-5, max_value=1e-2),
           **hu.gcs)
    def test_adam(self, inputs, in_place, beta1, beta2, lr, iters, epsilon,
                  gc, dc):
        grad, m1, m2 = inputs
        m2 += np.abs(m2) + 0.01
        lr = np.asarray([lr], dtype=np.float32)
        iters = np.asarray([iters], dtype=np.int32)
        op = core.CreateOperator("Adam",
            ["grad", "m1", "m2", "lr", "iters"],
            ["grad" if in_place else "grad_o",
             "m1" if in_place else "m1_o",
             "m2" if in_place else "m2_o"],
            beta1=beta1, beta2=beta2, epsilon=epsilon,
            device_option=gc)
        input_device_options = {"lr": hu.cpu_do, "iters": hu.cpu_do}
        self.assertDeviceChecks(
            dc, op, [grad, m1, m2, lr, iters], [0], input_device_options)

        # Reference
        def adam(grad, m1, m2, lr, iters):
            lr = lr[0]
            iters = iters[0]
            t = iters + 1
            corrected_local_rate = lr * np.sqrt(1. - np.power(beta2, t)) / \
                (1. - np.power(beta1, t))

            m1_o = (beta1 * m1) + (1. - beta1) * grad
            m2_o = (beta2 * m2) + (1. - beta2) * np.square(grad)
            grad_o = corrected_local_rate * m1_o / \
                (np.sqrt(m2_o) + epsilon)

            return (grad_o, m1_o, m2_o)

        self.assertReferenceChecks(gc, op, [grad, m1, m2, lr, iters],
                                   adam, input_device_options)

    @given(inputs=hu.tensors(n=3),
           in_place=st.booleans(),
           decay=st.floats(min_value=0.1, max_value=0.9),
           momentum=st.floats(min_value=0.1, max_value=0.9),
           lr=st.floats(min_value=0.1, max_value=0.9),
           epsilon=st.floats(min_value=1e-5, max_value=1e-2),
           **hu.gcs)
    def test_rmsprop(self, inputs, in_place, decay, momentum, lr, epsilon,
                     gc, dc):
        grad, ms, mom = inputs
        ms = np.abs(ms) + 0.01
        lr = np.asarray([lr], dtype=np.float32)
        op = core.CreateOperator("RmsProp",
            ["grad", "ms", "mom", "lr"],
            ["grad" if in_place else "grad_o",
             "ms" if in_place else "ms_o",
             "mom" if in_place else "mom_o"],
            momentum=momentum, decay=decay, epsilon=epsilon, device_option=gc)
        input_device_options = {"lr": hu.cpu_do}
        self.assertDeviceChecks(
            dc, op, [grad, ms, mom, lr], [0], input_device_options)

        def rmsprop(grad, ms, mom, lr):
            lr = lr[0]
            ms_o = ms + (1. - decay) * (np.square(grad) - ms)
            mom_o = momentum * mom + lr * grad / np.sqrt(epsilon + ms_o)
            grad_o = mom_o
            return (grad_o, ms_o, mom_o)
        self.assertReferenceChecks(gc, op, [grad, ms, mom, lr],
                                   rmsprop, input_device_options)

    @given(inputs=hu.tensors(n=2),
           in_place=st.booleans(),
           lr=st.floats(min_value=0.1, max_value=0.9),
           epsilon=st.floats(min_value=1e-5, max_value=1e-2),
           **hu.gcs)
    def test_adagrad(self, inputs, in_place, lr, epsilon,
                     gc, dc):
        grad, h = inputs
        h = np.abs(h) + 0.01
        lr = np.asarray([lr], dtype=np.float32)
        op = core.CreateOperator("Adagrad")(
            ["grad", "h", "lr"],
            ["grad" if in_place else "grad_o",
             "h" if in_place else "h_o"],
            epsilon=epsilon, device_option=gc)
        input_device_options = {"lr": hu.cpu_do}
        self.assertDeviceChecks(
            dc, op, [grad, h, lr], [0], input_device_options)

        def adagrad(grad, h, lr):
            lr = lr[0]
            h_o = h + np.square(grad)
            grad_o = lr * grad / (np.sqrt(h_o) + epsilon)
            return (grad_o, h_o)
        self.assertReferenceChecks(gc, op, [grad, h, lr],
                                   adagrad, input_device_options)
