from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy
import time
from functools import partial, reduce
from future.utils import viewitems, viewkeys
from hypothesis import assume, given, settings, HealthCheck
import hypothesis.strategies as st
import unittest
import os

from caffe2.python import core, workspace, tt_core, dyndep
import caffe2.python.hypothesis_test_util as hu
from caffe2.proto import caffe2_pb2

dyndep.InitOpsLibrary('@/caffe2/caffe2/fb/optimizers:sgd_simd_ops')


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@st.composite
def _tensor_and_prefix(draw, dtype, elements, min_dim=1, max_dim=4, **kwargs):
    dims_ = draw(
        st.lists(hu.dims(**kwargs), min_size=min_dim, max_size=max_dim))
    extra_ = draw(
        st.lists(hu.dims(**kwargs), min_size=min_dim, max_size=max_dim))
    assume(len(dims_) + len(extra_) < max_dim)
    return (draw(hu.arrays(dims_ + extra_, dtype, elements)),
            draw(hu.arrays(extra_, dtype, elements)))


def _tensor_and_indices(min_dim=1, max_dim=4, dtype=np.float32,
                        elements=None, **kwargs):
    """ generates a tensor and a list of indices of larger tensor of same dim"""
    data_dims_ = st.lists(hu.dims(**kwargs), min_size=min_dim, max_size=max_dim)
    original_dim = st.integers(min_value=2, max_value=10)
    return st.tuples(data_dims_, original_dim).flatmap(lambda pair: st.tuples(
        st.just(pair[1]),  # original dimension
        hu.arrays(pair[0], dtype, elements),  # data tensor
        hu.arrays(pair[0][0], dtype=np.int64, elements=st.integers(
            min_value=0, max_value=pair[1] - 1)),
    ))


_NUMPY_TYPE_TO_ENUM = {
    np.float32: core.DataType.FLOAT,
    np.int32: core.DataType.INT32,
    np.bool: core.DataType.BOOL,
    np.uint8: core.DataType.UINT8,
    np.int8: core.DataType.INT8,
    np.uint16: core.DataType.UINT16,
    np.int16: core.DataType.INT16,
    np.int64: core.DataType.INT64,
    np.float64: core.DataType.DOUBLE,
}


def _dtypes(dtypes=None):
    dtypes = dtypes if dtypes else [np.int32, np.int64, np.float32]
    return st.sampled_from(dtypes)


def _test_binary(name, ref, filter_=None, gcs=hu.gcs,
                 test_gradient=False, allow_inplace=False, dtypes=_dtypes):
    @given(
        inputs=dtypes().flatmap(
            lambda dtype: hu.tensors(
                n=2, dtype=dtype,
                elements=hu.elements_of_type(dtype, filter_=filter_))),
        out=st.sampled_from(('Y', 'X1', 'X2') if allow_inplace else ('Y',)),
        **gcs)
    @settings(max_examples=3, timeout=100)
    def test_binary(self, inputs, out, gc, dc):
        op = core.CreateOperator(name, ["X1", "X2"], [out])
        X1, X2 = inputs
        self.assertDeviceChecks(dc, op, [X1, X2], [0])
        # We only do gradient check with float32 types.
        if test_gradient and X1.dtype == np.float32:
            self.assertGradientChecks(gc, op, [X1, X2], 0, [0])
        self.assertReferenceChecks(gc, op, [X1, X2], ref)

    return test_binary


def _test_binary_broadcast(name, ref, filter_=None,
                           gcs=hu.gcs, allow_inplace=False, dtypes=_dtypes):
    @given(
        inputs=dtypes().flatmap(lambda dtype: _tensor_and_prefix(
            dtype=dtype,
            elements=hu.elements_of_type(dtype, filter_=filter_))),
        in_place=(st.booleans() if allow_inplace else st.just(False)),
        **gcs)
    @settings(max_examples=3, timeout=100)
    def test_binary_broadcast(self, inputs, in_place, gc, dc):
        op = core.CreateOperator(
            name, ["X1", "X2"], ["X1" if in_place else "Y"], broadcast=1)
        X1, X2 = inputs
        self.assertDeviceChecks(dc, op, [X1, X2], [0])

        def cast_ref(x, y):
            return (np.array(ref(x, y)[0], dtype=x.dtype), )

        # gradient not implemented yet
        # self.assertGradientChecks(gc, op, [X1, X2], 0, [0])
        self.assertReferenceChecks(gc, op, [X1, X2], cast_ref)

    return test_binary_broadcast


class TestOperators(hu.HypothesisTestCase):

    def test_comparison_ops(self):
        ops = {"LT": lambda x1, x2: [x1 < x2],
               "LE": lambda x1, x2: [x1 <= x2],
               "GT": lambda x1, x2: [x1 > x2],
               "GE": lambda x1, x2: [x1 >= x2]}
        for name, ref in viewitems(ops):
            _test_binary(name, ref, gcs=hu.gcs_cpu_only)(self)
            _test_binary_broadcast(name, ref, gcs=hu.gcs_cpu_only)(self)

    @given(inputs=hu.tensors(n=2), in_place=st.booleans(), **hu.gcs)
    def test_sum(self, inputs, in_place, gc, dc):
        op = core.CreateOperator("Sum", ["X1", "X2"],
                                        ["Y" if not in_place else "X1"])
        X1, X2 = inputs
        self.assertDeviceChecks(dc, op, [X1, X2], [0])
        self.assertGradientChecks(gc, op, [X1, X2], 0, [0])

    @given(inputs=hu.tensors(n=2, min_dim=2, max_dim=2), **hu.gcs_cpu_only)
    def test_row_mul(self, inputs, gc, dc):
        op = core.CreateOperator("RowMul", ["X1", "X2"], ["Y"])
        X1, Xtmp = inputs
        X2 = Xtmp[:, 0]

        def ref(x, y):
            ret = np.zeros(shape=x.shape, dtype=x.dtype)
            for i in range(y.size):
                ret[i, ] = x[i, ] * y[i]
            return [ret]

        self.assertDeviceChecks(dc, op, [X1, X2], [0])
        for i in range(2):
            self.assertGradientChecks(gc, op, [X1, X2], i, [0])
        self.assertReferenceChecks(gc, op, [X1, X2], ref)

    @given(inputs=hu.tensors(n=2), **hu.gcs_cpu_only)
    def test_max(self, inputs, gc, dc):
        op = core.CreateOperator("Max", ["X1", "X2"], ["Y"])

        X1, X2 = inputs
        # Make X1 and X2 far from each other, since X1=X2 is not differentiable
        # and the step size of gradient checker is 0.05
        X1[np.logical_and(X1 >= X2 - 0.05, X1 <= X2)] -= 0.05
        X1[np.logical_and(X1 <= X2 + 0.05, X1 >= X2)] += 0.05
        self.assertDeviceChecks(dc, op, [X1, X2], [0])
        for i in range(2):
            self.assertGradientChecks(gc, op, [X1, X2], i, [0])

        def elementwise_max(X, Y):
            return [np.maximum(X, Y)]
        self.assertReferenceChecks(gc, op, [X1, X2], elementwise_max)

    def test_add(self):
        def not_overflow(x):
            if not isinstance(x, float):
                return abs(x) < (1 << 30) - 1
            return True

        def ref(x, y):
            return (x + y, )
        _test_binary("Add", ref, filter_=not_overflow, test_gradient=True)(self)
        _test_binary_broadcast("Add", ref, filter_=not_overflow)(self)

    def test_sub(self):
        def ref(x, y):
            return (x - y, )
        # TODO(jiayq): enable gradient test when implemented.
        _test_binary("Sub", ref, test_gradient=True)(self)
        _test_binary_broadcast("Sub", ref)(self)

    def test_mul(self):
        def not_overflow(x):
            if not isinstance(x, float):
                return abs(x) < (1 << 15) - 1
            return True

        def ref(x, y):
            return (x * y, )
        _test_binary("Mul", ref, filter_=not_overflow, test_gradient=True)(self)
        _test_binary_broadcast("Mul", ref, filter_=not_overflow)(self)

    def test_div(self):
        def ref(x, y):
            return (x / y, )

        def non_zero(x):
            return abs(x) > 1e-2

        def div_dtypes():
            return st.sampled_from([np.float32, np.float64])

        _test_binary(
            "Div", ref, filter_=non_zero, test_gradient=True,
            dtypes=div_dtypes, gcs=hu.gcs_cpu_only
        )(self)
        _test_binary(
            "Div", ref, filter_=non_zero, test_gradient=False,
            dtypes=div_dtypes
        )(self)
        _test_binary_broadcast(
            "Div", ref, filter_=non_zero, dtypes=div_dtypes)(self)

    @given(X=hu.tensor(), in_place=st.booleans(), **hu.gcs)
    def test_negative(self, X, in_place, gc, dc):
        op = core.CreateOperator("Negative", ["X"],
                                 ["Y" if not in_place else "X"])
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0])

    @given(X=hu.tensor(), **hu.gcs)
    def test_tanh(self, X, gc, dc):
        op = core.CreateOperator("Tanh", "X", "Y")
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0])

    @given(X=hu.tensor(), **hu.gcs)
    def test_averaged_loss(self, X, gc, dc):
        op = core.CreateOperator("AveragedLoss", ["X"], ["loss"])
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0])

    @given(X=hu.tensor(), inplace=st.booleans(), **hu.gcs)
    def test_softsign(self, X, inplace, gc, dc):
        op = core.CreateOperator("Softsign", ["X"], ["X" if inplace else "Y"])

        def softsign(X):
            return (X / (1 + np.abs(X)),)

        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertReferenceChecks(gc, op, [X], softsign)
        if inplace:
            with self.assertRaises(Exception):
                self.assertGradientChecks(gc, op, [X], 0, [0])
        else:
            self.assertGradientChecks(gc, op, [X], 0, [0])

    @given(
        device_options=st.lists(
            min_size=2,
            max_size=4,
            elements=st.sampled_from(hu.expanded_device_options)),
        set_seed=st.booleans())
    def test_random_seed_behaviour(self, device_options, set_seed):
        # Assume we are always operating on CUDA or CPU, since RNG is
        # inconsistent between CPU and GPU.
        device_options = copy.deepcopy(device_options)
        assume(len({do.device_type for do in device_options}) == 1)
        if set_seed:
            for do in device_options:
                do.random_seed = 1000

        def run(do):
            # Reset each time because 'Y' may already exist in the workspace
            #   on a different device
            workspace.ResetWorkspace()
            ws = workspace.C.Workspace()
            op = core.CreateOperator(
                "XavierFill", [], ["Y"],
                device_option=do,
                shape=[2])
            ws.run(op)
            return ws.blobs["Y"].fetch()

        ys = [run(do) for do in device_options]
        for y in ys[1:]:
            if set_seed:
                np.testing.assert_array_equal(ys[0], y)
            else:
                with self.assertRaises(AssertionError):
                    np.testing.assert_array_equal(ys[0], y)

    @given(axis=st.integers(min_value=1, max_value=4),
           num_output=st.integers(min_value=4, max_value=8),
           engine=st.sampled_from(["", "PACKED"]),
           **hu.gcs)
    def test_fully_connected_axis(self, axis, num_output, engine, gc, dc):
        np.random.seed(1)
        X = np.random.randn(1, 2, 3, 2, 1).astype(np.float32)

        def prod(xs):
            p = 1
            for x in xs:
                p *= x
            return p

        K = prod(list(X.shape)[axis:])
        N = num_output
        W = np.random.randn(N, K).astype(np.float32)
        b = np.random.randn(N).astype(np.float32)

        op = core.CreateOperator(
            "FC",
            ["X", "W", "b"],
            ["Y"],
            engine=engine,
            axis=axis)
        for name, param in [("X", X), ("W", W), ("b", b)]:
            self.ws.create_blob(name).feed(param)
        self.ws.run(op)
        Y = self.ws.blobs["Y"].fetch()
        self.assertEqual(list(Y.shape), list(X.shape)[:axis] + [N])

        inputs = [X, W, b]
        self.assertDeviceChecks(dc, op, inputs, [0])
        for param, _ in enumerate(inputs):
            self.assertGradientChecks(gc, op, inputs, param, [0])

    @unittest.skipIf(not workspace.has_gpu_support,
                     "Skipping test due to no gpu present.")
    @given(hidden_size=st.integers(min_value=1, max_value=3),
           num_layers=st.integers(min_value=1, max_value=3),
           bidirectional=st.booleans(),
           rnn_mode=st.sampled_from(["lstm"]),   # TODO: "gru"
           input_mode=st.sampled_from(["linear"]),
           dropout=st.floats(min_value=1.0, max_value=1.0),
           T=st.integers(min_value=2, max_value=6),
           N=st.integers(min_value=1, max_value=4),
           D=st.integers(min_value=1, max_value=4))
    def test_recurrent(self, hidden_size, num_layers, bidirectional, rnn_mode,
                       input_mode, dropout, T, N, D):
        #there's a bug in miopen for N=1 which would be resolved in the next release.
        if workspace.has_hip_support:
            assume(N>1)
        # Random seed, this one happens to pass
        seed = 1234
        np.random.seed(seed)
        # set device option
        if workspace.has_hip_support:
           device_option = hu.hip_do
           engine = 'MIOPEN'
        else:
           device_option = hu.gpu_do
           engine = 'CUDNN'
        input_weight_size = hidden_size * D
        upper_layer_input_weight_size = hidden_size * hidden_size
        if bidirectional:
            upper_layer_input_weight_size *= 2
        recurrent_weight_size = hidden_size * hidden_size
        input_bias_size = hidden_size
        recurrent_bias_size = hidden_size
        num_directions = 2 if bidirectional else 1
        first_layer_sz = input_weight_size + recurrent_weight_size + \
                         input_bias_size + recurrent_bias_size
        upper_layer_sz = upper_layer_input_weight_size + \
                         recurrent_weight_size + input_bias_size + \
                         recurrent_bias_size
        total_sz = 4 * (first_layer_sz + (num_layers - 1) * upper_layer_sz)
        total_sz *= num_directions

        W = np.random.rand(total_sz).astype(np.float32)
        self.ws.create_blob("WEIGHT").feed(W, device_option=device_option)

        op = core.CreateOperator(
            "Recurrent",
            ["INPUT", "HIDDEN_INPUT", "CELL_INPUT", "WEIGHT"],
            ["OUTPUT", "HIDDEN_OUTPUT", "CELL_OUTPUT",
             "RNN_SCRATCH", "DROPOUT_STATES"],
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            rnn_mode=rnn_mode,
            dropout=dropout,
            input_mode=input_mode,
            num_layers=num_layers,
            seed=seed,
            engine=engine)
        X = np.random.randn(T, N, D).astype(np.float32)
        self.ws.create_blob("INPUT").feed(X, device_option=device_option)
        W = self.ws.blobs["WEIGHT"].fetch()
        H = np.random.randn(
            num_layers, N, hidden_size * num_directions).astype(
                np.float32)
        C = np.random.randn(
            num_layers, N, hidden_size * num_directions).astype(
                np.float32) if rnn_mode == "lstm" else \
            np.empty((1,)).astype(np.float32)  # unused in GRU
        inputs = [X, H, C, W]
        input_idxs = [i for (i, _) in enumerate(inputs)] \
            if rnn_mode == "lstm" else [0, 1, 3]  # ignore C
        for input_idx in input_idxs:
            self.assertGradientChecks(
                device_option, op, inputs, input_idx, [0],
                stepsize=0.01, threshold=0.01)

    @given(ndim=st.integers(1, 4),
           axis=st.integers(0, 3),
           add_axis=st.integers(0, 1),
           num_inputs=st.integers(2, 4), **hu.gcs)
    def test_depth_concat(self, ndim, axis, add_axis, num_inputs, gc, dc):
        assume(axis < ndim)
        input_names = ['X0', 'X1', 'X2', 'X3'][:num_inputs]
        shape = [2, 3, 5, 7][:ndim]
        individual_dims = [1, 2, 3, 4, 5][:num_inputs]
        inputs = []
        for i in range(num_inputs):
            if add_axis == 0:
                # Sets a unique dim and create the input.
                shape[axis] = individual_dims[i]
            inputs.append(np.random.randn(*shape).astype(np.float32))
        op = core.CreateOperator("Concat", input_names, ["Y", "Y_dims"],
                                 axis=axis, add_axis=add_axis)
        self.assertDeviceChecks(dc, op, inputs, [0])
        for i in range(num_inputs):
            self.assertGradientChecks(gc, op, inputs, i, [0])

        # Reference
        def depth_concat(*inputs):
            inputs = list(inputs)
            if add_axis:
                for i in range(len(inputs)):
                    inputs[i] = np.expand_dims(inputs[i], axis)
            input_dims = np.array([np.shape(x)[axis] for x in inputs])
            return [np.concatenate(inputs, axis=axis), input_dims]

        self.assertReferenceChecks(gc, op, inputs, depth_concat)

    @given(num_inputs=st.integers(2, 4),
           order=st.sampled_from([("NCHW", 1), ("NHWC", 3)]),
           **hu.gcs)
    def test_depth_concat_with_order(self, num_inputs, order, gc, dc):
        input_names = ['X0', 'X1', 'X2', 'X3'][:num_inputs]
        shape = [2, 3, 5, 7]
        individual_dims = [1, 2, 3, 4][:num_inputs]
        inputs = []
        for i in range(num_inputs):
            # Sets a unique dim and create the input.
            shape[order[1]] = individual_dims[i]
            inputs.append(np.random.rand(*shape).astype(np.float32))
        op = core.CreateOperator("Concat", input_names, ["Y", "Y_dims"],
                                 order=order[0])
        self.assertDeviceChecks(dc, op, inputs, [0])
        for i in range(num_inputs):
            self.assertGradientChecks(gc, op, inputs, i, [0])

        # Reference
        def depth_concat_with_order(*inputs):
            inputs = list(inputs)
            axis = order[1]
            input_dims = np.array([np.shape(x)[axis] for x in inputs])
            return [np.concatenate(inputs, axis=axis), input_dims]

        self.assertReferenceChecks(gc, op, inputs, depth_concat_with_order)

    @given(X=hu.arrays(dims=[5, 2],
                       elements=st.floats(min_value=1.0, max_value=10.0)),
           **hu.gcs_cpu_only)
    def test_last_n_windows(self, X, gc, dc):
        workspace.FeedBlob('input', X)
        workspace.FeedBlob('next', np.array(0, dtype=np.int32))
        workspace.CreateBlob('output')
        collect_net = core.Net('collect_net')
        collect_net.LastNWindowCollector(
            ['output', 'next', 'input'],
            ['output', 'next'],
            num_to_collect=7,
        )
        plan = core.Plan('collect_data')
        plan.AddStep(core.execution_step('collect_data',
                                         [collect_net], num_iter=2))
        workspace.RunPlan(plan)
        output = workspace.FetchBlob('output')
        inputs = workspace.FetchBlob('input')
        new_output = np.zeros([7, inputs.shape[1]])
        for i in range(inputs.shape[0] * 2):
            new_output[i % 7] = inputs[i % inputs.shape[0]]
        import numpy.testing as npt
        npt.assert_almost_equal(output, new_output, decimal=5)

    @given(dtype=st.sampled_from([np.float32, np.float64, np.int32, np.bool]))
    def test_print(self, dtype):
        data = np.random.permutation(6).astype(dtype)
        self.ws.create_blob("data").feed(data)
        op = core.CreateOperator("Print", "data", [])
        self.ws.run(op)

    @given(inputs=hu.tensors(n=2),
           in_place=st.booleans(),
           momentum=st.floats(min_value=0.1, max_value=0.9),
           nesterov=st.booleans(),
           lr=st.floats(min_value=0.1, max_value=0.9),
           **hu.gcs)
    def test_momentum_sgd(
            self, inputs, in_place, momentum, nesterov, lr, gc, dc):
        grad, m = inputs
        lr = np.asarray([lr], dtype=np.float32)
        op = core.CreateOperator(
            "MomentumSGD",
            ["grad", "m", "lr"],
            ["grad" if in_place else "grad_o",
             "m" if in_place else "m_o"],
            momentum=momentum,
            nesterov=int(nesterov),
            device_option=gc)
        self.assertDeviceChecks(
            dc, op, [grad, m, lr], [0])

        # Reference
        def momentum_sgd(grad, m, lr):
            lr = lr[0]
            if not nesterov:
                adjusted_gradient = lr * grad + momentum * m
                return (adjusted_gradient, adjusted_gradient)
            else:
                m_new = momentum * m + lr * grad
                return ((1 + momentum) * m_new - momentum * m, m_new)

        self.assertReferenceChecks(gc, op, [grad, m, lr], momentum_sgd)

    @given(inputs=hu.tensors(n=3),
           in_place=st.booleans(),
           decay=st.floats(min_value=0.1, max_value=0.9),
           momentum=st.floats(min_value=0.1, max_value=0.9),
           lr=st.floats(min_value=0.1, max_value=0.9),
           epsilon=st.floats(min_value=1e-5, max_value=1e-2),
           **hu.gcs)
    def test_rmsprop_sgd(self, inputs, in_place, decay, momentum, lr, epsilon,
                         gc, dc):
        grad, ms, mom = inputs
        ms = np.abs(ms) + 0.01
        lr = np.asarray([lr], dtype=np.float32)
        op = core.CreateOperator(
            "RmsProp",
            ["grad", "ms", "mom", "lr"],
            ["grad" if in_place else "grad_o",
             "ms" if in_place else "ms_o",
             "mom" if in_place else "mom_o"],
            momentum=momentum, decay=decay, epsilon=epsilon, device_option=gc)
        self.assertDeviceChecks(dc, op, [grad, ms, mom, lr], [0])

        def rmsprop(grad, ms, mom, lr):
            lr = lr[0]
            ms_o = ms + (1. - decay) * (np.square(grad) - ms)
            mom_o = momentum * mom + lr * grad / np.sqrt(epsilon + ms_o)
            grad_o = mom_o
            return (grad_o, ms_o, mom_o)
        self.assertReferenceChecks(gc, op, [grad, ms, mom, lr], rmsprop)

    # Reference
    @staticmethod
    def _dense_ftrl(alpha, beta, lambda1, lambda2, w, nz, g):
        if isinstance(alpha, np.ndarray):
            alpha = np.asscalar(alpha)
        n = np.take(nz, 0, axis=-1)
        z = np.take(nz, 1, axis=-1)
        # python port of Sigrid's implementation
        g2 = g * g
        sigma = (np.sqrt(n + g2) - np.sqrt(n)) / alpha
        z += g - sigma * w
        n += g2
        w = (np.sign(z) * lambda1 - z) / (
            (beta + np.sqrt(n)) / alpha + lambda2)
        w[np.abs(z) <= lambda1] = 0
        return (w, np.stack([n, z], axis=-1))

    @given(inputs=hu.tensors(n=4),
           in_place=st.booleans(),
           alpha=st.floats(min_value=0.01, max_value=0.1),
           beta=st.floats(min_value=0.1, max_value=0.9),
           lambda1=st.floats(min_value=0.001, max_value=0.1),
           lambda2=st.floats(min_value=0.001, max_value=0.1),
           engine=st.sampled_from([None, "SIMD"]),
           **hu.gcs_cpu_only)
    def test_ftrl_sgd(self, inputs, in_place, alpha, beta, lambda1, lambda2,
                      engine, gc, dc):
        var, n, z, grad = inputs
        n = np.abs(n)
        nz = np.stack([n, z], axis=-1)
        op = core.CreateOperator(
            "Ftrl",
            ["var", "nz", "grad"],
            ["var" if in_place else "var_o",
             "nz" if in_place else "nz_o"],
            alpha=alpha, beta=beta, lambda1=lambda1, lambda2=lambda2,
            engine=engine,
            device_option=gc)
        self.assertDeviceChecks(
            dc, op, [var, nz, grad], [0])

        self.assertReferenceChecks(
            gc, op, [var, nz, grad],
            partial(self._dense_ftrl, alpha, beta, lambda1, lambda2))

    # Reference
    @staticmethod
    def _dense_gftrl(alpha, beta, lambda1, lambda2, w, nz, g):
        if isinstance(alpha, np.ndarray):
            alpha = np.asscalar(alpha)

        old_shape = g.shape

        n = np.take(nz, 0, axis=-1)
        z = np.take(nz, 1, axis=-1)

        output_dim = g.shape[0]

        w = w.reshape(output_dim, -1)
        g = g.reshape(output_dim, -1)

        n = n.reshape(output_dim, -1)
        z = z.reshape(output_dim, -1)

        input_dim = g.shape[1]

        g2 = g * g
        sigma = (np.sqrt(n + g2) - np.sqrt(n)) / alpha
        z += g - sigma * w
        n += g2

        z_norms = np.linalg.norm(z, 2, axis=0)

        z_norms = z_norms + 1e-6
        w = z * ((lambda1 * np.sqrt(output_dim)) / z_norms - 1) / \
                    ((beta + np.sqrt(n)) / alpha + lambda2)
        for i in range(input_dim):
            if z_norms[i] <= lambda1 * np.sqrt(output_dim):
                w[:, i] = 0

        w = w.reshape(old_shape)
        n = n.reshape(old_shape)
        z = z.reshape(old_shape)
        return (w, np.stack([n, z], axis=-1))

    @given(inputs=hu.tensors(n=4),
           in_place=st.booleans(),
           alpha=st.floats(min_value=0.01, max_value=0.1),
           beta=st.floats(min_value=0.1, max_value=0.9),
           lambda1=st.floats(min_value=0.001, max_value=0.1),
           lambda2=st.floats(min_value=0.001, max_value=0.1),
           engine=st.sampled_from([None, "SIMD"]),
           **hu.gcs_cpu_only)
    def test_gftrl_sgd(self, inputs, in_place, alpha, beta, lambda1, lambda2,
                      engine, gc, dc):
        var, n, z, grad = inputs
        n = np.abs(n)
        nz = np.stack([n, z], axis=-1)
        op = core.CreateOperator(
            "GFtrl",
            ["var", "nz", "grad"],
            ["var" if in_place else "var_o",
             "nz" if in_place else "nz_o"],
            alpha=alpha, beta=beta, lambda1=lambda1, lambda2=lambda2,
            engine=engine,
            device_option=gc)
        self.assertDeviceChecks(
            dc, op, [var, nz, grad], [0])

        self.assertReferenceChecks(
            gc, op, [var, nz, grad],
            partial(self._dense_gftrl, alpha, beta, lambda1, lambda2))

    @given(inputs=hu.tensors(n=4),
           alpha=st.floats(min_value=0.01, max_value=0.1),
           beta=st.floats(min_value=0.1, max_value=0.9),
           lambda1=st.floats(min_value=0.001, max_value=0.1),
           lambda2=st.floats(min_value=0.001, max_value=0.1),
           engine=st.sampled_from([None, "SIMD"]),
           **hu.gcs_cpu_only)
    def test_sparse_ftrl_sgd(self, inputs, alpha, beta, lambda1, lambda2,
                             engine, gc, dc):
        var, n, z, grad = inputs
        # generate fake subset manually because hypothesis is too complicated :)
        indices = np.arange(var.shape[0])
        indices = indices[indices % 2 == 0]
        grad = grad[indices]
        n = np.abs(n)
        nz = np.stack([n, z], axis=-1)
        op = core.CreateOperator(
            "SparseFtrl",
            ["var", "nz", "indices", "grad"],
            ["var", "nz"],
            alpha=alpha, beta=beta, lambda1=lambda1, lambda2=lambda2,
            engine=engine,
            device_option=gc)
        self.assertDeviceChecks(
            dc, op, [var, nz, indices, grad], [0])

        # Reference
        def ftrl(w, nz, i, g):
            sw, snz = self._dense_ftrl(alpha, beta, lambda1, lambda2,
                                       w[i], nz[i], g)
            w[i] = sw
            nz[i] = snz
            return (w, nz)

        self.assertReferenceChecks(gc, op, [var, nz, indices, grad], ftrl)

    # Reference
    @staticmethod
    def _dense_ftrl_send_alpha_by_input(beta, lambda1, lambda2, w, nz, g, alpha):
        return TestOperators._dense_ftrl(alpha, beta, lambda1, lambda2, w, nz,
                                         g)

    @given(inputs=hu.tensors(n=4),
           in_place=st.booleans(),
           alpha=st.floats(min_value=0.01, max_value=0.1),
           beta=st.floats(min_value=0.1, max_value=0.9),
           lambda1=st.floats(min_value=0.001, max_value=0.1),
           lambda2=st.floats(min_value=0.001, max_value=0.1),
           engine=st.sampled_from([None, "SIMD"]),
           **hu.gcs_cpu_only)
    def test_ftrl_sgd_send_alpha_by_input(self, inputs, in_place, alpha, beta,
                                          lambda1, lambda2, engine, gc, dc):
        var, n, z, grad = inputs
        n = np.abs(n)
        nz = np.stack([n, z], axis=-1)
        alpha = np.array(alpha).astype(np.float32)
        op = core.CreateOperator(
            "Ftrl",
            ["var", "nz", "grad", "alpha"],
            ["var" if in_place else "var_o",
             "nz" if in_place else "nz_o"],
            beta=beta, lambda1=lambda1, lambda2=lambda2,
            engine=engine,
            device_option=gc)
        self.assertDeviceChecks(
            dc, op, [var, nz, grad, alpha], [0])

        self.assertReferenceChecks(
            gc, op, [var, nz, grad, alpha],
            partial(self._dense_ftrl_send_alpha_by_input, beta, lambda1, lambda2))

    @given(inputs=hu.tensors(n=4),
           alpha=st.floats(min_value=0.01, max_value=0.1),
           beta=st.floats(min_value=0.1, max_value=0.9),
           lambda1=st.floats(min_value=0.001, max_value=0.1),
           lambda2=st.floats(min_value=0.001, max_value=0.1),
           engine=st.sampled_from([None, "SIMD"]),
           **hu.gcs_cpu_only)
    def test_sparse_ftrl_sgd_send_alpha_by_input(self, inputs, alpha, beta,
                                                 lambda1, lambda2, engine, gc,
                                                 dc):
        var, n, z, grad = inputs
        # generate fake subset manually because hypothesis is too complicated :)
        indices = np.arange(var.shape[0])
        indices = indices[indices % 2 == 0]
        grad = grad[indices]
        n = np.abs(n)
        nz = np.stack([n, z], axis=-1)
        alpha = np.array(alpha).astype(np.float32)
        op = core.CreateOperator(
            "SparseFtrl",
            ["var", "nz", "indices", "grad", "alpha"],
            ["var", "nz"],
            beta=beta, lambda1=lambda1, lambda2=lambda2,
            engine=engine,
            device_option=gc)
        self.assertDeviceChecks(
            dc, op, [var, nz, indices, grad, alpha], [0])

        # Reference
        def ftrl(w, nz, i, g, alpha):
            sw, snz = self._dense_ftrl_send_alpha_by_input(beta, lambda1,
                                                           lambda2, w[i], nz[i],
                                                           g, alpha)
            w[i] = sw
            nz[i] = snz
            return (w, nz)

        self.assertReferenceChecks(gc, op, [var, nz, indices, grad, alpha],
                                   ftrl)

    # TODO: (bddppq) test_unique keeps running into segfault on rocm 1.8.2
    @given(input=hu.tensor(max_value=20,
                           max_dim=1,
                           dtype=np.int32,
                           elements=st.integers(min_value=0, max_value=10)),
           with_remapping=st.booleans(),
           **hu.gcs_no_hip)
    def test_unique(self, input, with_remapping, gc, dc):
        op = core.CreateOperator(
            "Unique",
            ["input"],
            ["unique"] + (["remapping"] if with_remapping else []),
            device_option=gc)
        self.assertDeviceChecks(dc, op, [input], [0])

        # Validator
        def unique_valid(input, unique, remapping=None):
            self.assertEqual(unique.size, len(set(input)))
            self.assertEqual(sorted(unique), sorted(set(input)))
            if with_remapping:
                self.assertEqual(remapping.shape, input.shape)
                remapped = [unique[remapping[i]] for i in range(len(input))]
                np.testing.assert_array_equal(remapped, input)

        self.assertValidationChecks(gc, op, [input], unique_valid)

    @given(prediction=hu.arrays(dims=[10, 3],
                                elements=st.floats(allow_nan=False,
                                                   allow_infinity=False,
                                                   min_value=0,
                                                   max_value=1)),
           labels=hu.arrays(dims=[10],
                            dtype=np.int32,
                            elements=st.integers(min_value=0,
                                                 max_value=3 - 1)),
           top_k=st.integers(min_value=1, max_value=3),
           **hu.gcs)
    def test_accuracy(self, prediction, labels, top_k, gc, dc):
        if(top_k > 1):
            gc = hu.cpu_do

        op = core.CreateOperator(
            "Accuracy",
            ["prediction", "labels"],
            ["accuracy"],
            top_k=top_k,
            device_option=gc
        )

        def op_ref(prediction, labels, top_k):
            N = prediction.shape[0]
            correct = 0
            for i in range(0, len(prediction)):
                pred_sorted = sorted(
                    ([item, j] for j, item in enumerate(prediction[i])),
                    key=lambda x: x[0],
                    reverse=True
                )
                max_ids = [x[1] for x in pred_sorted[0:top_k]]
                for m in max_ids:
                    if m == labels[i]:
                        correct += 1
            accuracy = correct / N
            return (accuracy,)

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[prediction, labels, top_k],
            reference=op_ref)

    @given(target_probabilities=hu.arrays(
        dims=[10], elements=st.floats(allow_nan=False,
                                      allow_infinity=False,
                                      min_value=0.01,
                                      max_value=1)),
           **hu.gcs)
    def test_perplexity(self, target_probabilities, gc, dc):
        op = core.CreateOperator(
            "Perplexity",
            ["target_probabilities"],
            ["perplexity"]
        )

        def op_ref(target_probabilities):
            N = target_probabilities.shape[0]
            perplexities = np.power(target_probabilities, -1.0 / N)
            perplexity = reduce(lambda x, y: x * y, perplexities)
            return (perplexity,)

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[target_probabilities],
            reference=op_ref)

    @given(lengths=st.lists(st.integers(min_value=0, max_value=10),
                            min_size=0,
                            max_size=10),
           **hu.gcs_cpu_only)
    def test_lengths_to_segment_ids(self, lengths, gc, dc):
        op = core.CreateOperator(
            "LengthsToSegmentIds",
            ["lengths"],
            ["segment_ids"])

        def op_ref(lengths):
            sids = []
            for i, l in enumerate(lengths):
                sids.extend(l * [i])
            return (np.array(sids, dtype=np.int32), )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[np.array(lengths, dtype=np.int32)],
            reference=op_ref)

    @given(lengths=st.lists(st.integers(min_value=0, max_value=10),
                            min_size=0,
                            max_size=10),
           **hu.gcs_cpu_only)
    def test_lengths_range_fill(self, lengths, gc, dc):
        op = core.CreateOperator(
            "LengthsRangeFill",
            ["lengths"],
            ["increasing_seq"])

        def op_ref(lengths):
            sids = []
            for _, l in enumerate(lengths):
                sids.extend(list(range(l)))
            return (np.array(sids, dtype=np.int32), )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[np.array(lengths, dtype=np.int32)],
            reference=op_ref)

    @given(**hu.gcs_cpu_only)
    def test_segment_ids_to_ranges(self, gc, dc):
        lengths = [4, 6, 3, 2, 0, 4]
        op = core.CreateOperator(
            "SegmentIdsToRanges",
            ["segment_ids"],
            ["ranges"])

        def op_ref(segment_ids):
            ranges = [np.array([0, 0], dtype=np.int32)]
            prev = 0
            for i, sid in enumerate(segment_ids):
                while sid != prev:
                    prev += 1
                    ranges.append(np.array([i, 0], dtype=np.int32))
                ranges[-1][1] += 1
            return (np.array(ranges, dtype=np.int32), )

        def lengths_to_segment_ids(lengths):
            sids = []
            for i, l in enumerate(lengths):
                sids.extend(l * [i])
            return (np.array(sids, dtype=np.int32), )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=np.array(lengths_to_segment_ids(lengths), dtype=np.int32),
            reference=op_ref)

    @given(lengths=st.lists(st.integers(min_value=0, max_value=10),
                            min_size=0,
                            max_size=10),
           **hu.gcs_cpu_only)
    def test_lengths_to_ranges(self, lengths, gc, dc):
        op = core.CreateOperator(
            "LengthsToRanges",
            ["lengths"],
            ["ranges"])

        def op_ref(x):
            if not x.size:
                return (x.reshape((0, 2)), )
            return (np.column_stack((np.concatenate(([0], np.cumsum(x)[:-1])),
                                     x)), )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[np.array(lengths, dtype=np.int32)],
            reference=op_ref)

    @given(prediction=hu.arrays(dims=[10, 3],
                                elements=st.floats(allow_nan=False,
                                                   allow_infinity=False,
                                                   min_value=0,
                                                   max_value=1)),
           labels=hu.arrays(dims=[10],
                            dtype=np.int32,
                            elements=st.integers(min_value=0,
                                                 max_value=3 - 1)),
            **hu.gcs)
    def test_multi_class_accuracy(self, prediction, labels, gc, dc):
        op = core.CreateOperator(
            "MultiClassAccuracy",
            ["prediction", "labels"],
            ["accuracies", "amounts"]
        )

        def op_ref(prediction, labels):
            N = prediction.shape[0]
            D = prediction.shape[1]
            accuracies = np.empty(D, dtype=float)
            accuracies.fill(0)
            amounts = np.empty(D, dtype=int)
            amounts.fill(0)
            max_ids = np.argmax(prediction, axis=1)
            for i in range(0, N):
                max_id = max_ids[i]
                label_id = labels[i]
                if max_id == label_id:
                    accuracies[label_id] += 1
                amounts[label_id] += 1
            for i in range(0, D):
                amount = amounts[i]
                if amount:
                    accuracies[i] /= amount
            return (accuracies, amounts,)

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[prediction, labels],
            reference=op_ref)

    @given(lengths=st.lists(st.integers(min_value=0, max_value=10),
                            min_size=0,
                            max_size=10),
           **hu.gcs_cpu_only)
    def test_segment_ids_to_lengths(self, lengths, gc, dc):
        op = core.CreateOperator(
            "SegmentIdsToLengths",
            ["segment_ids"],
            ["lengths"])

        def lengths_to_ids(lengths):
            sids = []
            for i, l in enumerate(lengths):
                sids.extend(l * [i])
            return sids

        segment_ids = lengths_to_ids(lengths)

        def ids_to_lengths(ids):
            ids_length = len(ids)
            if ids_length == 0:
                return (np.array([], dtype=np.int32),)

            lengths = []
            # segment id starts with 0
            prev_id = -1
            tmp_length = 0
            for idx in range(ids_length):
                cur_id = ids[idx]
                if cur_id != prev_id:
                    if idx != 0:
                        lengths.append(tmp_length)
                    while prev_id + 1 != cur_id:
                        lengths.append(0)
                        prev_id += 1
                    prev_id = cur_id
                    tmp_length = 0
                tmp_length += 1
            lengths.append(tmp_length)
            return (np.array(lengths, dtype=np.int32),)

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[np.array(segment_ids, dtype=np.int32)],
            reference=ids_to_lengths)

    @given(lengths=st.lists(st.integers(min_value=1, max_value=10),
                            min_size=0,
                            max_size=10),
            power=st.sampled_from([0.5, 1.0, 1.5, 2.0]),
           **hu.gcs_cpu_only)
    def test_lengths_to_weights(self, lengths, power, gc, dc):
        op = core.CreateOperator(
            "LengthsToWeights",
            ["lengths"],
            ["weights"],
            power=power)

        def lengths_to_weights(lengths):
            weighted_length = []
            for l in lengths:
                weighted_length.extend(l * [1 / pow(l, power)])

            return (np.array(weighted_length, dtype=float),)

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[np.array(lengths, dtype=np.int32)],
            reference=lengths_to_weights)

    @given(input_tensor=hu.arrays(
        dims=[10], elements=st.floats(allow_nan=False,
                                      allow_infinity=False)),
           **hu.gcs)
    def test_abs(self, input_tensor, gc, dc):
        op = core.CreateOperator(
            "Abs",
            ["input"],
            ["output"]
        )

        def abs_ref(input_tensor):
            return (np.abs(input_tensor),)

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[input_tensor],
            reference=abs_ref)

    @given(input_tensor=hu.arrays(
        dims=[10], elements=st.floats(min_value=-10,
                                      max_value=10)),
           **hu.gcs)
    def test_cos(self, input_tensor, gc, dc):
        op = core.CreateOperator(
            "Cos",
            ["input"],
            ["output"]
        )

        def cos_ref(input_tensor):
            return (np.cos(input_tensor),)

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[input_tensor],
            reference=cos_ref)

    @given(input_tensor=hu.arrays(
        dims=[10], elements=st.floats(min_value=-10,
                                      max_value=10)),
           **hu.gcs)
    def test_sin(self, input_tensor, gc, dc):
        op = core.CreateOperator(
            "Sin",
            ["input"],
            ["output"]
        )

        def sin_ref(input_tensor):
            return (np.sin(input_tensor),)

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[input_tensor],
            reference=sin_ref)

    @given(input_tensor=hu.arrays(
        dims=[10], elements=st.floats(allow_nan=False,
                                      allow_infinity=False)),
           **hu.gcs)
    def test_exp(self, input_tensor, gc, dc):
        op = core.CreateOperator(
            "Exp",
            ["input"],
            ["output"]
        )

        def exp_ref(input_tensor):
            return (np.exp(input_tensor),)

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[input_tensor],
            reference=exp_ref)

    @given(input_tensor=hu.arrays(
        dims=[10], elements=st.floats(min_value=1,
                                      max_value=10000)),
           **hu.gcs_cpu_only)
    def test_log(self, input_tensor, gc, dc):
        op = core.CreateOperator(
            "Log",
            ["input"],
            ["output"]
        )

        def log_ref(input_tensor):
            return (np.log(input_tensor),)

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[input_tensor],
            reference=log_ref)
        self.assertGradientChecks(gc, op, [input_tensor], 0, [0])

    def test_blobs_dequeue_timeout(self):
        op = core.CreateOperator(
            "CreateBlobsQueue",
            [],
            ["queue"],
            capacity=5,
            num_blobs=1)
        self.ws.run(op)
        t = time.time()
        op = core.CreateOperator(
            "DequeueBlobs",
            ["queue"],
            ["out"],
            timeout_secs=0.2)
        self.assertRaises(RuntimeError, lambda: self.ws.run(op))
        t = time.time() - t
        self.assertGreater(t, 0.19)

    @given(num_threads=st.integers(1, 10),  # noqa
           num_elements=st.integers(1, 100),
           capacity=st.integers(1, 5),
           num_blobs=st.integers(1, 3),
           do=st.sampled_from(hu.device_options))
    def test_blobs_queue_threading(self, num_threads, num_elements,
                                   capacity, num_blobs, do):
        """
        - Construct matrices of size N x D
        - Start K threads
        - Push all N rows into the queue of capacity C
        - Pull all N rows out of the queue.
        - Verify that the output matrices are permutation of the rows of the
          original matrices.
        """
        import threading
        try:
            import queue
        except ImportError:
            # Py3
            import Queue as queue
        op = core.CreateOperator(
            "CreateBlobsQueue",
            [],
            ["queue"],
            capacity=capacity,
            num_blobs=num_blobs,
            device_option=do)
        self.ws.run(op)

        xs = [np.random.randn(num_elements, 5).astype(np.float32)
              for _ in range(num_blobs)]
        q = queue.Queue()
        for i in range(num_elements):
            q.put([x[i] for x in xs])

        def enqueue(t):
            while True:
                feed_blobs = ["x_{}_{}".format(i, t) for i in range(num_blobs)]
                op = core.CreateOperator(
                    "EnqueueBlobs",
                    ["queue"] + feed_blobs,
                    feed_blobs,
                    device_option=do)
                try:
                    elems = q.get_nowait()
                    for elem, feed_blob in zip(elems, feed_blobs):
                        self.ws.create_blob(feed_blob).feed(
                            elem, device_option=do)
                    self.ws.run(op)
                except queue.Empty:
                    return

        # Create all blobs before racing on multiple threads
        # (blob creation is not threadsafe)
        for t in range(num_threads):
            for i in range(num_blobs):
                self.ws.create_blob("x_{}_{}".format(i, t))

        threads = [threading.Thread(target=enqueue, args=(t,))
                   for t in range(num_threads)]
        for thread in threads:
            thread.start()

        for n in range(num_elements):
            dequeue_blobs = ["y_{}_{}".format(i, n) for i in range(num_blobs)]
            op = core.CreateOperator(
                "DequeueBlobs",
                ["queue"],
                dequeue_blobs,
                device_option=do)
            self.ws.run(op)
        for thread in threads:
            thread.join()
        op = core.CreateOperator("CloseBlobsQueue", ["queue"], [])
        self.ws.run(op)
        ys = [np.vstack([self.ws.blobs["y_{}_{}".format(i, n)].fetch()
                         for n in range(num_elements)])
              for i in range(num_blobs)]
        for i in range(num_blobs):
            self.assertEqual(ys[i].shape, xs[i].shape)
            for j in range(num_elements):
                # Verify that the rows of the returned blob are a
                # permutation. The order may be different due to
                # different threads racing.
                self.assertTrue(
                    any(np.array_equal(xs[i][j], ys[i][k])
                        for k in range(num_elements)))

    @given(num_producers=st.integers(1, 10),
           num_consumers=st.integers(1, 10),
           capacity=st.integers(1, 5),
           num_blobs=st.integers(1, 3),
           do=st.sampled_from(hu.device_options))
    def test_safe_blobs_queue(self, num_producers, num_consumers,
                              capacity, num_blobs, do):
        init_net = core.Net('init_net')
        queue = init_net.CreateBlobsQueue(
            [], 1, capacity=capacity, num_blobs=num_blobs)
        producer_steps = []
        truth = 0
        for i in range(num_producers):
            name = 'producer_%d' % i
            net = core.Net(name)
            blobs = [net.ConstantFill([], 1, value=1.0, run_once=False)
                     for times in range(num_blobs)]
            status = net.NextName()
            net.SafeEnqueueBlobs([queue] + blobs, blobs + [status])
            count = (i + 1) * 10
            step = core.execution_step(name, net, num_iter=count)
            truth += count
            producer_steps.append(step)
        producer_exit_net = core.Net('producer_exit_net')
        producer_exit_net.CloseBlobsQueue([queue], 0)
        producer_step = core.execution_step('producer', [
            core.execution_step(
                'producers', producer_steps, concurrent_substeps=True),
            core.execution_step('producer_exit', producer_exit_net)]
        )

        consumer_steps = []
        counters = []
        const_1 = init_net.ConstantFill([], 1, value=1.0)
        for i in range(num_consumers):
            name = 'consumer_%d' % i
            net1 = core.Net(name)
            blobs = net1.SafeDequeueBlobs([queue], num_blobs + 1)
            status = blobs[-1]

            net2 = core.Net(name + '_counter')
            counter = init_net.ConstantFill([], 1, value=0.0)
            counters.append(counter)
            net2.Add([counter, const_1], counter)
            consumer_steps.append(core.execution_step(
                name, [net1, net2], should_stop_blob=status))
        consumer_step = core.execution_step(
            'consumer', consumer_steps, concurrent_substeps=True)

        init_step = core.execution_step('init', init_net)
        worker_step = core.execution_step(
            'worker', [consumer_step, producer_step], concurrent_substeps=True)

        plan = core.Plan('test')
        plan.AddStep(init_step)
        plan.AddStep(worker_step)

        self.ws.run(plan)
        v = 0
        for counter in counters:
            v += self.ws.blobs[str(counter)].fetch().tolist()
        self.assertEqual(v, truth)

    @given(num_queues=st.integers(1, 5),
           num_iter=st.integers(5, 10),
           capacity=st.integers(1, 5),
           num_blobs=st.integers(1, 3))
    def test_weighted_sample_blobs_queue(
        self, num_queues, num_iter, capacity, num_blobs
    ):
        # Create BlobsQueue for each input queue
        print("num_queues", num_queues)
        init_net = core.Net('init_net')
        queues = [
            init_net.CreateBlobsQueue(
                [], 1, capacity=capacity, num_blobs=num_blobs
            ) for _ in range(num_queues)
        ]

        # Create multiple producer nets and one producer exist net
        producer_steps = []
        producer_exit_nets = []
        for i in range(num_queues):
            name = 'producer_%d' % i
            net = core.Net(name)
            blobs = [net.ConstantFill([], 1, value=1.0, run_once=False)
                     for _ in range(num_blobs)]
            status = net.NextName()
            net.SafeEnqueueBlobs([queues[i]] + blobs, blobs + [status])

            exit_net = core.Net('producer_exit_%d' % i)
            exit_net.CloseBlobsQueue(queues[i], 0)
            producer_exit_nets.append(exit_net)

            step = core.execution_step(
                name, [
                    core.execution_step(
                        'producer_%d' % i, [net], num_iter=num_iter
                    ),
                    core.execution_step('producer_exit_%d' % i, [exit_net]),
                ]
            )
            producer_steps.append(step)

        producer_step = core.execution_step(
            'producer', [
                core.execution_step(
                    'producers',
                    producer_steps,
                    concurrent_substeps=True,
                ),
            ]
        )

        status_lst = []

        def append(ins, outs):
            status_lst.append(ins)

        # Create one consumer dequeue net and one consumer exist net
        consumer_net = core.Net('weight_sample_dequeue_net')
        table_idx_blob = np.random.randint(low=-1, high=num_blobs, size=1)
        blobs = consumer_net.WeightedSampleDequeueBlobs(
            queues,
            num_blobs + 1,
            weights=np.random.uniform(low=0.0, high=1.0, size=(num_queues,)),
            table_idx_blob=table_idx_blob[0],
        )
        status = blobs[-1]
        consumer_net.Python(append)(status)

        consumer_step = core.execution_step(
            'consumer',
            [
                core.execution_step(
                    'consumer', [consumer_net], should_stop_blob=status
                ),
                core.execution_step('producer_exit', producer_exit_nets)
            ]
        )

        init_step = core.execution_step('init', init_net)
        worker_step = core.execution_step(
            'worker', [producer_step, consumer_step], concurrent_substeps=True)

        plan = core.Plan('test')
        plan.AddStep(init_step)
        plan.AddStep(worker_step)

        self.ws.run(plan)
        assert len(status_lst) >= num_iter + 1
        assert len(status_lst) <= num_iter * num_queues + 1

    @given(
        data=hu.tensor(),
        **hu.gcs_cpu_only)
    def test_squeeze_expand_dims(self, data, gc, dc):
        dims = [0, 0]
        if len(data.shape) > 2:
            dims.append(2)
        op = core.CreateOperator(
            "ExpandDims",
            ["data"],
            ["expanded"],
            dims=dims)

        def expand_dims_ref(data, *args, **kw):
            inc_dims = list(set(dims))
            inc_dims.sort()
            r = data
            for dim in inc_dims:
                r = np.expand_dims(r, axis=dim)
            return (r, )

        def squeeze_ref(data, *args, **kw):
            dec_dims = list(set(dims))
            dec_dims.sort(reverse=True)
            r = data
            for dim in dec_dims:
                r = np.squeeze(r, axis=dim)
            return (r, )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[data],
            reference=expand_dims_ref,
            output_to_grad='expanded',
            grad_reference=squeeze_ref)

    @given(**hu.gcs_cpu_only)
    def test_tt_layer(self, gc, dc):
        seed = 1234
        np.random.seed(seed)

        inp_sizes = [2, 2, 2, 2]
        out_sizes = [2, 2, 2, 2]
        tt_ranks = [1, 3, 3, 3, 1]

        op = core.CreateOperator(
            "TT",
            ["X", "b", "cores"],
            ["Y"],
            inp_sizes=inp_sizes,
            out_sizes=out_sizes,
            tt_ranks=tt_ranks,
        )

        X = np.expand_dims(
            np.random.rand(16).astype(np.float32), axis=0)
        b = np.array([0] * 16).astype(np.float32)
        cores = tt_core.init_tt_cores(inp_sizes, out_sizes, tt_ranks)

        self.ws.create_blob("X").feed(X)
        self.ws.create_blob("b").feed(b)
        self.ws.create_blob("cores").feed(cores)
        self.ws.run(op)

        Y = self.ws.blobs[("Y")].fetch()
        Y = Y.reshape([16])

        golden = np.array([-9.51763490e-07, -1.28442286e-06,
                           -2.86281141e-07, 2.28865644e-07,
                           -1.96180017e-06, -1.78920531e-06,
                           9.31094666e-07, -2.04273989e-07,
                           1.70017107e-06, 1.64845711e-06,
                           -1.06099132e-06, -4.69111137e-07,
                           6.57552358e-08, -1.28942040e-08,
                           -2.29114004e-07, -1.04262714e-06])

        # This golden array is dependent on the specified inp_sizes, out_sizes,
        # tt_ranks, and seed. Changing these will cause the test to fail.
        self.assertAlmostEqual(np.linalg.norm(golden - Y), 0, delta=1e-10)

    @given(num_workers=st.integers(1, 10),
           net_type=st.sampled_from(
               ["simple", "dag"] +
               (["async_dag"] if workspace.has_gpu_support else [])),
           **hu.gcs)
    def test_dag_net_forking(self, net_type, num_workers, gc, dc):
        from caffe2.python.model_helper import ModelHelper
        from caffe2.python import brew
        m = ModelHelper(name="test_model")
        n = 10
        d = 2
        depth = 2
        iters = 5
        np.random.seed(1701)
        # Build a binary tree of FC layers, summing at each node.
        for i in reversed(range(depth)):
            for j in range(2 ** i):
                bottom_1 = "{}_{}".format(i + 1, 2 * j)
                bottom_2 = "{}_{}".format(i + 1, 2 * j + 1)
                mid_1 = "{}_{}_m".format(i + 1, 2 * j)
                mid_2 = "{}_{}_m".format(i + 1, 2 * j + 1)
                top = "{}_{}".format(i, j)
                brew.fc(
                    m,
                    bottom_1, mid_1,
                    dim_in=d, dim_out=d,
                    weight_init=('ConstantFill', dict(value=np.random.randn())),
                    bias_init=('ConstantFill', dict(value=np.random.randn())))
                brew.fc(
                    m,
                    bottom_2, mid_2,
                    dim_in=d, dim_out=d,
                    weight_init=('ConstantFill', dict(value=np.random.randn())),
                    bias_init=('ConstantFill', dict(value=np.random.randn())))
                m.net.Sum([mid_1, mid_2], top)
        m.net.SquaredL2Distance(["0_0", "label"], "xent")
        m.net.AveragedLoss("xent", "loss")
        input_to_grad = m.AddGradientOperators(["loss"])
        m.Proto().device_option.CopyFrom(gc)
        m.param_init_net.Proto().device_option.CopyFrom(gc)

        m.Proto().type = net_type
        m.Proto().num_workers = num_workers

        self.ws.run(m.param_init_net)

        print(str(m.Proto()))

        def run():
            import numpy as np
            np.random.seed(1701)
            input_blobs = ["{}_{}".format(depth, j) for j in range(2 ** depth)]
            for input_blob in input_blobs:
                self.ws.create_blob(input_blob).feed(
                    np.random.randn(n, d).astype(np.float32),
                    device_option=gc)
                self.ws.create_blob("label").feed(
                    np.random.randn(n, d).astype(np.float32),
                    device_option=gc)
            self.ws.run(m.net)
            gradients = [
                self.ws.blobs[str(input_to_grad[input_blob])].fetch()
                for input_blob in input_blobs]
            return gradients

        outputs = [run() for _ in range(iters)]
        for output in outputs[1:]:
            np.testing.assert_array_equal(outputs[0], output)
            self.assertAlmostEqual(np.sum(np.square(output)), 91.81752,
                                   delta=1e-2)

    @given(input=hu.tensor(min_dim=2, max_dim=6, dtype=np.int32,
                           elements=st.integers(min_value=0,
                                                max_value=2**32 - 1)),
           slice_dim=st.integers(),
           a=st.integers(),
           b=st.integers(),
           is_empty=st.booleans(),
           **hu.gcs_cpu_only)
    def test_slice(self, input, slice_dim, a, b, is_empty, gc, dc):
        slice_dim = slice_dim % len(input.shape)
        if (is_empty):
            input = np.random.rand(*([0] + list(input.shape))).astype(np.int32)
            slice_dim += 1

        a = a % input.shape[slice_dim]
        b = b % input.shape[slice_dim] + 1
        start_vec = np.zeros(len(input.shape), dtype=np.int32)
        end_vec = np.ones(len(input.shape), dtype=np.int32) * -1
        start_vec[slice_dim] = min(a, b)
        end_vec[slice_dim] = max(a, b)
        op = core.CreateOperator(
            "Slice",
            ["input", "start", "end"],
            ["output"])

        def slice_ref(x, s, e):
            if len(s.shape) == 0:
                return x
            slc = [slice(si, None if ei == -1 else ei) for si, ei in zip(s, e)]
            return (x[slc], )

        self.assertReferenceChecks(gc, op, [input, start_vec, end_vec],
                                   slice_ref)

    @given(data=hu.tensor(), **hu.gcs_cpu_only)
    def test_shape(self, data, gc, dc):
        op = core.CreateOperator("Shape", ["data"], ["shape"])
        self.assertReferenceChecks(gc, op, [data], lambda x: (x.shape, ))

    @given(data=hu.tensor(), **hu.gcs_cpu_only)
    def test_shape_with_axes(self, data, gc, dc):
        def shape_ref(x, y):
            return ([x.shape[i] for i in y],)
        axes = np.random.randint(len(data.shape), size=10).tolist()
        op = core.CreateOperator("Shape", ["data"], ["shape"], axes=axes)
        self.assertReferenceChecks(gc, op, [data, axes], shape_ref)

    @given(data=hu.tensor(), **hu.gcs_cpu_only)
    def test_has_elements(self, data, gc, dc):
        op = core.CreateOperator("HasElements", ["data"], ["has_elements"])
        self.assertReferenceChecks(gc, op, [data], lambda x: (len(x) > 0, ))

        op = core.CreateOperator("IsEmpty", ["data"], ["is_empty"])
        self.assertReferenceChecks(gc, op, [data], lambda x: (len(x) == 0, ))

    @given(initial_iters=st.integers(0, 100),
           max_iters=st.integers(0, 100))
    def test_should_stop_as_criteria_net_execution_step(
            self, initial_iters, max_iters):
        net = core.Net("net")
        net.Iter(["iter"], ["iter"])
        self.ws.create_blob("iter").feed(
            np.asarray([initial_iters]).astype(np.int64))
        self.ws.create_blob("num_iters").feed(
            np.asarray([max_iters]).astype(np.int64))
        criteria_net = core.Net("criteria")
        criteria_net.GE(["iter", "num_iters"], ["stop"])
        criteria_net.Proto().external_output.extend(["stop"])

        plan = core.Plan('plan')
        plan.AddStep(core.execution_step(
            'step', [criteria_net, net],
            should_stop_blob=core.BlobReference("stop")))
        self.ws.run(plan)
        iters = self.ws.blobs[("iter")].fetch()
        self.assertEqual(iters.dtype, np.int64)
        self.assertEqual(iters[0], max(initial_iters, max_iters))

    def test_disabled_execution_step(self):
        def createNets(i, disabled):
            should_stop = 'should_stop_{}'.format(i)
            output = 'output_{}'.format(i)

            # init content and stop signal
            init = core.Net("init_{}".format(i))
            init.ConstantFill(
                [],
                [output],
                shape=[1],
                value=0.0
            )
            init.Cast([output], [should_stop], to='bool')

            # decide if disabled or not
            criterion = core.Net("criterion_{}".format(i))
            tmp = criterion.ConstantFill(
                [],
                shape=[1],
                value=1.0 if disabled else 0.0
            )
            criterion.Cast([tmp], [should_stop], to='bool')
            criterion.Proto().external_output.extend([should_stop])

            # the body net is just to turn a 0 blob to 1
            net = core.Net("net_{}".format(i))
            net.ConstantFill(
                [],
                [output],
                shape=[1],
                value=1.0
            )

            # always end the loop
            ender = core.Net("ender_{}".format(i))
            tmp = ender.ConstantFill(
                [],
                shape=[1],
                value=1.0
            )
            ender.Cast([tmp], [should_stop], to='bool')
            ender.Proto().external_output.extend([should_stop])

            return [init, criterion, net, ender]

        nets = [createNets(1, False),
                createNets(2, True),
                createNets(3, False)]
        steps = [
            core.execution_step(
                'step_1', nets[0],
                should_stop_blob=core.BlobReference('should_stop_1')),
            core.execution_step(
                'step_2', nets[1],
                should_stop_blob=core.BlobReference('should_stop_2')),
            core.execution_step('step_3', nets[2])
        ]
        expected = [1.0, 0.0, 1.0]

        plan = core.Plan('plan')
        plan.AddStep(core.execution_step('all_steps', steps, num_iter=3))
        self.ws.run(plan)

        for i, _ in enumerate(nets):
            self.assertEqual(
                self.ws.blobs['output_{}'.format(i + 1)].fetch()[0],
                expected[i])

    @given(initial_iters=st.integers(0, 100),
           num_iters=st.integers(0, 100))
    def test_iter_count_with_execution_step(self, initial_iters, num_iters):
        net = core.Net("net")
        net.Iter(["iter"], ["iter"])
        self.ws.create_blob("iter").feed(
            np.asarray([initial_iters]).astype(np.int64))

        step = core.ExecutionStep("step", [net])
        step.SetIter(num_iters)

        plan = core.Plan("plan")
        plan.AddStep(step)
        self.ws.run(plan)
        iters = self.ws.blobs[("iter")].fetch()
        self.assertEqual(iters.dtype, np.int64)
        self.assertEqual(iters[0], initial_iters + num_iters)


    @given(initial_iters=st.integers(0, 100),
           num_iters=st.integers(0, 100),
           num_nets=st.integers(0, 5))
    def test_atomic_iter_with_concurrent_steps(self, initial_iters, num_iters,
                                               num_nets):
        init_net = core.Net("init_net")
        iter_mutex = init_net.CreateMutex([], ["iter_mutex"])
        self.ws.create_blob("iter").feed(
            np.asarray([initial_iters]).astype(np.int64))
        concurrent_steps = core.ExecutionStep("concurrent_steps",
                                              num_iter=num_iters)
        for i in range(num_nets):
            net = core.Net("net_{}".format(i))
            net.AtomicIter([iter_mutex, "iter"], ["iter"])
            step = core.ExecutionStep("step", [net])
            concurrent_steps.AddSubstep(step)

        concurrent_steps.SetConcurrentSubsteps(True)
        plan = core.Plan("plan")
        plan.AddStep(concurrent_steps)

        stats_net = core.Net("stats_net")
        stats_net.StatRegistryExport([], ["stats_key", "stats_val", "stats_ts"])

        self.ws.run(init_net)
        self.ws.run(plan)
        self.ws.run(stats_net)
        iters = self.ws.blobs[("iter")].fetch()
        self.assertEqual(iters.dtype, np.int64)
        self.assertEqual(iters[0], initial_iters + num_iters * num_nets)

        if num_iters * num_nets > 0:
            stats_key = self.ws.blobs[("stats_key")].fetch()
            atomic_iter_key = b'atomic_iter/stats/iter/num_iter'
            self.assertTrue(atomic_iter_key in stats_key)
            stat_val = self.ws.blobs[("stats_val")].fetch()
            self.assertEqual(num_iters * num_nets, stat_val[list(stats_key).index(atomic_iter_key)])


    @given(a=hu.tensor(),
           src=st.sampled_from(list(viewkeys(_NUMPY_TYPE_TO_ENUM))),
           dst=st.sampled_from(list(viewkeys(_NUMPY_TYPE_TO_ENUM))),
           use_name=st.booleans(),
           **hu.gcs)
    def test_cast(self, a, src, dst, use_name, gc, dc):
        a = a.astype(src)

        # Casting from a float type outside the range of the integral
        # type is UB.
        ftypes = [np.float32, np.float64]
        if src in ftypes and dst not in ftypes and dst is not np.bool:
            info = np.iinfo(dst)
            a = np.clip(a, info.min, info.max)

        def ref(data):
            return [data.astype(dst)]

        to = _NUMPY_TYPE_TO_ENUM[dst]
        if use_name:
            to = caffe2_pb2.TensorProto.DataType.Name(to).lower()
        op = core.CreateOperator('Cast', ["X"], ["Y"], to=to)
        self.assertDeviceChecks(dc, op, [a], [0])
        out, = self.assertReferenceChecks(gc, op, [a], ref)
        self.assertEqual(dst, out.dtype)

    @given(a=hu.tensor(),
           eps=st.floats(min_value=1e-4, max_value=1e-2),
           a_grad=hu.tensor(elements=st.floats(min_value=0.01, max_value=0.99)),
           eps_grad=st.floats(min_value=1e-4, max_value=1e-3),
           **hu.gcs)
    def test_logit(self, a, eps, a_grad, eps_grad, gc, dc):
        def ref(data):
            data = np.clip(data, eps, 1.0 - eps)
            return (np.log(data / (1 - data)), )
        # forward testing carried out in the full range of input
        # to ensure original test coverage.
        # gradient test carried out with reduced input range
        # because the sharp increase of the logit curve at 0 and 1
        # error increases dramtically when input is close to 0 or 1
        # and it will fail the test.
        # So we only run gradient test in the range of (0.01, 0.99)
        # very occationally, test may fail due to random accumulated error
        # reduce test range to (0.02, 0.98) will improve test stability
        op = core.CreateOperator('Logit', ["X"], ["Y"], eps=eps)
        self.assertDeviceChecks(dc, op, [a], [0])
        self.assertReferenceChecks(gc, op, [a], ref)
        op_grad = core.CreateOperator('Logit', ["X"], ["Y"], eps=eps_grad)
        self.assertGradientChecks(gc, op_grad, [a_grad], 0, [0],
                                  threshold=0.04, stepsize=2e-3)

    @given(a=hu.tensor(elements=st.floats(allow_nan=True)),
           value=st.floats(min_value=-10, max_value=10),
           **hu.gcs)
    def test_replace_nan(self, a, value, gc, dc):
        def ref(data):
            out = np.copy(data)
            out[np.isnan(data)] = value
            return (out, )

        op = core.CreateOperator('ReplaceNaN', ["X"], ["Y"], value=value)
        self.assertDeviceChecks(dc, op, [a], [0])
        self.assertReferenceChecks(gc, op, [a], ref)

    @given(data=_dtypes(dtypes=[np.int32, np.int64, np.float32, np.bool]).
           flatmap(lambda dtype: hu.tensor(
               min_dim=1, dtype=dtype, elements=hu.elements_of_type(dtype))),
           has_input=st.booleans(),
           has_extra_shape=st.booleans(),
           extra_shape=st.lists(
           min_size=1, max_size=5, elements=st.integers(1, 5)),
           **hu.gcs)
    def test_constant_fill(self, data, has_input, has_extra_shape, extra_shape,
                           gc, dc):
        dtype = data.dtype.type
        # in opt mode, np.bool is converted into np.bool_
        if data.dtype == np.dtype(np.bool):
            dtype = np.bool

        value = data.item(0)
        gt_shape = data.shape
        inputs = [data]
        enum_type = _NUMPY_TYPE_TO_ENUM[dtype]

        if has_input:
            if has_extra_shape:
                op = core.CreateOperator('ConstantFill', ["X"], ["Y"],
                                         dtype=enum_type,
                                         extra_shape=extra_shape,
                                         value=value)
                gt_shape += tuple(extra_shape)
            else:
                op = core.CreateOperator('ConstantFill', ["X"], ["Y"],
                                         dtype=enum_type,
                                         value=value)
        else:
            op = core.CreateOperator('ConstantFill', [], ["Y"],
                                     dtype=enum_type,
                                     value=value,
                                     shape=list(gt_shape))
            inputs = []

        def ref(inputs=None):
            outputs = np.full(shape=gt_shape, fill_value=value, dtype=dtype)
            return [outputs]

        self.assertDeviceChecks(dc, op, inputs, [0])
        out, = self.assertReferenceChecks(gc, op, inputs, ref)
        self.assertEqual(dtype, out.dtype)

    @given(t=st.integers(1, 5),
           n=st.integers(1, 5),
           d=st.integers(1, 5))
    def test_elman_recurrent_network(self, t, n, d):
        from caffe2.python import model_helper, brew
        np.random.seed(1701)
        step_net = model_helper.ModelHelper(name="Elman")
        # TODO: name scope external inputs and outputs
        step_net.Proto().external_input.extend(
            ["input_t", "seq_lengths", "timestep",
             "hidden_t_prev", "gates_t_w", "gates_t_b"])
        step_net.Proto().type = "simple"
        step_net.Proto().external_output.extend(["hidden_t", "gates_t"])
        brew.fc(step_net,
                "hidden_t_prev", "gates_t", dim_in=d, dim_out=d, axis=2)
        step_net.net.Sum(["gates_t", "input_t"], ["gates_t"])
        step_net.net.Sigmoid(["gates_t"], ["hidden_t"])

        # Initialize params for step net in the parent net
        for op in step_net.param_init_net.Proto().op:
            workspace.RunOperatorOnce(op)

        backward_ops, backward_mapping = core.GradientRegistry.GetBackwardPass(
            step_net.Proto().op, {"hidden_t": "hidden_t_grad"})
        backward_mapping = {
            str(k): str(v) for k, v in viewitems(backward_mapping)
        }
        backward_step_net = core.Net("ElmanBackward")
        del backward_step_net.Proto().op[:]
        backward_step_net.Proto().op.extend(backward_ops)
        assert backward_mapping["input_t"] == "gates_t_grad"
        links = [
            ("hidden_t_prev", "hidden", 0),
            ("hidden_t", "hidden", 1),
            ("input_t", "input", 0),
        ]
        link_internal, link_external, link_offset = zip(*links)
        backward_links = [
            ("hidden_t_prev_grad", "hidden_grad", 0),
            ("hidden_t_grad", "hidden_grad", 1),
            ("gates_t_grad", "input_grad", 0),
        ]
        backward_link_internal, backward_link_external, backward_link_offset = \
            zip(*backward_links)
        backward_step_net.Proto().external_input.extend(["hidden_t_grad"])
        backward_step_net.Proto().external_input.extend(
            step_net.Proto().external_input)
        backward_step_net.Proto().external_input.extend(
            step_net.Proto().external_output)
        inputs = ["input", "seq_lengths", "gates_t_w", "gates_t_b", "hidden_input"]
        recurrent_inputs = ["hidden_input"]
        op = core.CreateOperator(
            "RecurrentNetwork",
            inputs,
            ["output", "hidden", "hidden_output", "step_workspaces"],
            alias_src=["hidden", "hidden"],
            alias_dst=["output", "hidden_output"],
            alias_offset=[1, -1],
            recurrent_states=["hidden"],
            initial_recurrent_state_ids=[
                inputs.index(i) for i in recurrent_inputs
            ],
            link_internal=link_internal,
            link_external=link_external,
            link_offset=link_offset,
            backward_link_internal=backward_link_internal,
            backward_link_external=backward_link_external,
            backward_link_offset=backward_link_offset,
            param=[inputs.index(p) for p in step_net.params],
            step_net=step_net.Proto(),
            backward_step_net=backward_step_net.Proto(),
            outputs_with_grads=[0],
        )
        workspace.FeedBlob(
            "input", np.random.randn(t, n, d).astype(np.float32))
        workspace.FeedBlob(
            "hidden_input", np.random.randn(1, n, d).astype(np.float32))
        workspace.FeedBlob(
            "seq_lengths", np.random.randint(0, t, size=(n,)).astype(np.int32))

        def reference(input, seq_lengths, gates_w, gates_b, hidden_input):
            T = input.shape[0]
            N = input.shape[1]
            D = input.shape[2]
            hidden = np.zeros(shape=(T + 1, N, D))
            assert hidden.shape[0] == T + 1
            assert hidden.shape[1] == N
            assert hidden.shape[2] == D

            hidden[0, :, :] = hidden_input
            for t in range(T):
                input_t = input[t].reshape(1, N, D)
                hidden_t_prev = hidden[t].reshape(1, N, D)
                gates = np.dot(hidden_t_prev, gates_w.T)
                gates = gates.reshape(1, N, D) + input_t.reshape(1, N, D)
                hidden[t + 1] = sigmoid(gates)
            return hidden[1:], hidden, hidden[-1].reshape(1, N, D)

        self.assertReferenceChecks(
            hu.cpu_do,
            op,
            [workspace.FetchBlob(name)
             for name in ["input", "seq_lengths", "gates_t_w", "gates_t_b",
                          "hidden_input"]],
            reference,
            outputs_to_check=[0, 1, 2])

        for param in [0, 2, 3]:
            self.assertGradientChecks(
                hu.cpu_do,
                op,
                [workspace.FetchBlob(name)
                 for name in ["input", "seq_lengths", "gates_t_w", "gates_t_b",
                              "hidden_input"]],
                param,
                [0])

    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    @given(n=st.integers(1, 5),
           c=st.integers(1, 5),
           h=st.integers(1, 5),
           w=st.integers(1, 5),
           pad=st.integers(0, 2),
           block_size=st.integers(2, 3),
           **hu.gcs)
    def test_space_to_batch(self, n, c, h, w, pad, block_size, gc, dc):
        assume((h + 2 * pad) % block_size == 0)
        assume((w + 2 * pad) % block_size == 0)
        X = np.random.randn(n, c, h, w).astype(np.float32)
        op = core.CreateOperator("SpaceToBatch", ["X"], ["Y"],
                                 pad=pad, block_size=block_size)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0])

    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    @given(n=st.integers(1, 5),
           c=st.integers(1, 5),
           h=st.integers(1, 5),
           w=st.integers(1, 5),
           pad=st.integers(0, 2),
           block_size=st.integers(2, 3),
           **hu.gcs)
    def test_batch_to_space(self, n, c, h, w, pad, block_size, gc, dc):
        assume((h + 2 * pad) % block_size == 0)
        assume((w + 2 * pad) % block_size == 0)
        X = np.random.randn(
            n * block_size * block_size,
            c,
            (h + 2 * pad) // block_size,
            (w + 2 * pad) // block_size).astype(np.float32)
        op = core.CreateOperator("BatchToSpace", ["X"], ["Y"],
                                 pad=pad, block_size=block_size)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0])

    @given(X=hu.tensor(),
           in_place=st.booleans(),
           scale=st.floats(min_value=-2.0, max_value=2.0),
           **hu.gcs)
    def test_scale(self, X, in_place, scale, gc, dc):
        op = core.CreateOperator(
            "Scale", ["X"], ["Y" if not in_place else "X"],
            scale=scale)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0])

    @given(s=st.text())
    def test_string_serde(self, s):
        s = s.encode('ascii', 'ignore')
        self.ws.create_blob("a").feed(s)
        serialized = self.ws.blobs["a"].serialize("a")
        self.ws.create_blob("b").deserialize(serialized)
        self.assertEqual(s, self.ws.blobs[("a")].fetch())
        self.assertEqual(s, self.ws.blobs[("b")].fetch())

    @given(pad=st.integers(0, 3),
           size=st.integers(1, 10),
           input_channels=st.integers(1, 5),
           batch_size=st.integers(1, 5),
           order=st.sampled_from(["NCHW", "NHWC"]),
           mode=st.sampled_from(["constant", "reflect", "edge"]),
           **hu.gcs)
    def test_same_pad_image(self, pad, size, input_channels, batch_size, order,
                            mode, gc, dc):
        assume(size > pad)

        op = core.CreateOperator(
            "PadImage",
            ["X"],
            ["Y"],
            pad=pad,
            mode=mode,
            order=order,
        )
        if order == "NHWC":
            X = np.random.rand(
                batch_size, size, size, input_channels).astype(np.float32) - 0.5

            def numpy_pad_ref(x):
                return (np.pad(
                    x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode),)

        else:
            X = np.random.rand(
                batch_size, input_channels, size, size).astype(np.float32) - 0.5

            def numpy_pad_ref(x):
                return (np.pad(
                    x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode),)

        self.assertReferenceChecks(gc, op, [X], numpy_pad_ref)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0])

    @given(pad_t=st.integers(0, 3),
           pad_l=st.integers(0, 3),
           pad_b=st.integers(0, 3),
           pad_r=st.integers(0, 3),
           size=st.integers(1, 10),
           input_channels=st.integers(1, 5),
           batch_size=st.integers(1, 5),
           order=st.sampled_from(["NCHW", "NHWC"]),
           mode=st.sampled_from(["constant", "reflect", "edge"]),
           **hu.gcs)
    def test_pad_image(self, pad_t, pad_l, pad_b, pad_r, size, input_channels,
                       batch_size, order, mode, gc, dc):
        assume(size > max(pad_b, pad_r, pad_t, pad_l))

        op = core.CreateOperator(
            "PadImage",
            ["X"],
            ["Y"],
            pad_t=pad_t,
            pad_l=pad_l,
            pad_b=pad_b,
            pad_r=pad_r,
            mode=mode,
            order=order,
        )
        if order == "NHWC":
            X = np.random.rand(
                batch_size, size, size, input_channels).astype(np.float32) - 0.5

            def numpy_pad_ref(x):
                return (np.pad(
                    x, ((0, 0), (pad_t, pad_b), (pad_l, pad_r), (0, 0)),
                    mode),)

        else:
            X = np.random.rand(
                batch_size, input_channels, size, size).astype(np.float32) - 0.5

            def numpy_pad_ref(x):
                return (np.pad(
                    x, ((0, 0), (0, 0), (pad_t, pad_b), (pad_l, pad_r)),
                    mode),)

        self.assertReferenceChecks(gc, op, [X], numpy_pad_ref)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0])

    @given(size=st.integers(7, 10),
           input_channels=st.integers(1, 10),
           batch_size=st.integers(1, 3),
           order=st.sampled_from(["NCHW", "NHWC"]),
           epsilon=st.floats(min_value=1e-4, max_value=1e-2),
           **hu.gcs_cpu_only)
    def test_instance_norm(self, size, input_channels, batch_size, order,
                           epsilon, gc, dc):
        op = core.CreateOperator(
            "InstanceNorm",
            ["X", "scale", "bias"],
            ["Y"],
            order=order,
            epsilon=epsilon,
        )
        np.random.seed(1701)
        scale = np.random.rand(input_channels).astype(np.float32) + 0.5
        bias = np.random.rand(input_channels).astype(np.float32) - 0.5
        X = np.random.rand(
            batch_size, input_channels, size, size).astype(np.float32) - 0.5
        if order == "NHWC":
            X = X.swapaxes(1, 2).swapaxes(2, 3)

        def ref_nchw(x, scale, bias):
            x = x.reshape(batch_size * input_channels, size * size)
            y = (x - x.mean(1)[:, np.newaxis])
            y /= np.sqrt(x.var(1) + epsilon)[:, np.newaxis]
            y = y.reshape(batch_size, input_channels, size, size)
            y = y * scale.reshape(1, input_channels, 1, 1)
            y = y + bias.reshape(1, input_channels, 1, 1)
            return (y, )

        def ref_nhwc(x, scale, bias):
            x = x.swapaxes(2, 3).swapaxes(1, 2)
            y = ref_nchw(x, scale, bias)[0]
            return (y.swapaxes(1, 2).swapaxes(2, 3), )

        self.assertReferenceChecks(
            gc, op, [X, scale, bias],
            ref_nchw if order == "NCHW" else ref_nhwc)
        # TODO(jiayq): when there are backward and GPU implementations, enable
        # these two.
        # self.assertDeviceChecks(dc, op, [X, scale, bias], [0])
        # self.assertGradientChecks(gc, op, [X, scale, bias], 0, [0])

        ws = workspace.C.Workspace()
        feeds = [("X", X), ("scale", scale), ("bias", bias)]
        for blob, arr in feeds:
            ws.create_blob(blob).feed(arr)
        for _ in range(100):
            ws.run(op)
        for blob, arr in feeds:
            np.testing.assert_array_equal(ws.blobs[blob].fetch(), arr)

    @given(inp=_dtypes().flatmap(lambda dt: _tensor_and_indices(
        elements=st.floats(min_value=0, max_value=1), dtype=dt)),
        **hu.gcs)
    def test_sparse_to_dense(self, inp, gc, dc):
        first_dim, X, I = inp
        if X.dtype != np.dtype('float32') and gc.device_type in {caffe2_pb2.CUDA, caffe2_pb2.HIP} :
            # Cuda only support 32 bit float
            print("Bailout {}".format(X.dtype))
            return
        if gc.device_type in {caffe2_pb2.CUDA, caffe2_pb2.HIP}:
            # Cuda version only support int32
            I = I.astype(np.int32)

        # values don't matter
        D = np.zeros((first_dim,) + X.shape[1:]).astype(X.dtype)

        op = core.CreateOperator("SparseToDense", ["I", "X", "D"], ["Y"])

        def sparse_to_dense(I, X, D):
            O = np.zeros(D.shape)
            for i, p in enumerate(I):
                O[p] += X[i]
            return [O]

        self.assertReferenceChecks(gc, op, [I, X, D], sparse_to_dense)
        X = X.astype(np.float32)
        self.assertGradientChecks(gc, op, [I, X, D], 1, [0])

    @given(inputs=hu.tensors(n=2, min_dim=2, max_dim=2), **hu.gcs_cpu_only)
    def test_dot_product(self, inputs, gc, dc):
        X, Y = inputs
        op = core.CreateOperator("DotProduct", ["X", "Y"], 'out')

        def dotproduct(X, Y):
            return (np.sum(X * Y, axis=1), )

        self.assertReferenceChecks(gc, op, [X, Y], dotproduct)
        self.assertDeviceChecks(dc, op, [X, Y], [0])
        self.assertGradientChecks(gc, op, [X, Y], 0, [0])
        self.assertGradientChecks(gc, op, [X, Y], 1, [0])

    @given(N=st.integers(min_value=2, max_value=10),
           M=st.integers(min_value=2, max_value=10),
           K=st.integers(min_value=2, max_value=10),
           pad_value=st.floats(min_value=0.1, max_value=1.0),
           **hu.gcs_cpu_only)
    def test_dot_product_with_padding(self, N, M, K, pad_value, gc, dc):
        X = np.random.rand(N, M).astype(np.float32) - 0.5
        Y = np.random.rand(N, K).astype(np.float32) - 0.5
        op = core.CreateOperator("DotProductWithPadding", ["X", "Y"], 'out',
                                 pad_value=pad_value)

        def dotproduct(X, Y):
            Z = np.ones((N, max(M, K))).astype(np.float32) * pad_value
            if M < K:
                Z[:, :M] = X
                return (np.sum(Z * Y, axis=1), )
            else:
                Z[:, :K] = Y
                return (np.sum(Z * X, axis=1), )

        self.assertReferenceChecks(gc, op, [X, Y], dotproduct)
        self.assertDeviceChecks(dc, op, [X, Y], [0])
        self.assertGradientChecks(gc, op, [X, Y], 0, [0])
        self.assertGradientChecks(gc, op, [X, Y], 1, [0])

    @given(N=st.integers(min_value=2, max_value=10),
           M=st.integers(min_value=2, max_value=10),
           pad_value=st.floats(min_value=0.1, max_value=1.0),
           **hu.gcs_cpu_only)
    def test_dot_product_with_rep_padding(self, N, M, pad_value, gc, dc):
        K = 2 * M
        X = np.random.rand(N, M).astype(np.float32) - 0.5
        Y = np.random.rand(N, K).astype(np.float32) - 0.5
        op = core.CreateOperator("DotProductWithPadding", ["X", "Y"], 'out',
                                 replicate=True,
                                 pad_value=pad_value)

        def dotproduct(X, Y):
            import numpy.matlib as npm
            if M < K:
                Z = npm.repmat(X, 1, K // M)
                return (np.sum(Z * Y, axis=1), )
            else:
                Z = npm.repmat(Y, 1, M // K)
                return (np.sum(Z * X, axis=1), )

        self.assertReferenceChecks(gc, op, [X, Y], dotproduct)
        self.assertDeviceChecks(dc, op, [X, Y], [0])
        self.assertGradientChecks(gc, op, [X, Y], 0, [0])
        self.assertGradientChecks(gc, op, [X, Y], 1, [0])

    @given(N=st.integers(min_value=2, max_value=10),
           M=st.integers(min_value=2, max_value=10), **hu.gcs_cpu_only)
    def test_ensure_dense(self, N, M, gc, dc):
        # in place
        X = np.random.rand(N, M).astype(np.float32) - 0.5
        op = core.CreateOperator("EnsureDense", ["X"], "X")
        self.assertReferenceChecks(gc, op, [X], lambda x: [x])
        self.assertDeviceChecks(dc, op, [X], [0])
        # or not
        X = np.random.rand(N, M).astype(np.float32) - 0.5
        op = core.CreateOperator("EnsureDense", ["X"], "out")
        self.assertReferenceChecks(gc, op, [X], lambda x: [x])
        self.assertDeviceChecks(dc, op, [X], [0])

    @given(N=st.integers(min_value=10, max_value=100),
           M=st.integers(min_value=2, max_value=10),
           num_buckets=st.integers(min_value=1, max_value=5),
           **hu.gcs_cpu_only)
    def test_accumulate_histogram_op(self, N, M, num_buckets, gc, dc):
        X = np.random.rand(N, M).astype(np.float32)
        lower_bound, upper_bound = 0.1, 0.9
        op = core.CreateOperator("AccumulateHistogram", ["X"],
                                 ['cur_hist', 'acc_hist'],
                                 lower_bound=lower_bound,
                                 upper_bound=upper_bound,
                                 num_buckets=num_buckets)

        def histogram(X):
            hist = np.zeros((num_buckets + 2, ), dtype=np.int32)
            segment = (upper_bound - lower_bound) / num_buckets
            Y = np.zeros((N, M), dtype=np.int32)
            Y[X < lower_bound] = 0
            Y[X >= upper_bound] = num_buckets + 1
            Y[(X >= lower_bound) & (X < upper_bound)] = \
                ((X[(X >= lower_bound) & (X < upper_bound)] - lower_bound) /
                        segment + 1).astype(np.int32)

            for i in range(Y.shape[0]):
                for j in range(Y.shape[1]):
                    hist[Y[i][j]] += 1
            cur_hist, acc_hist = hist, hist

            return [cur_hist, acc_hist]

        self.assertDeviceChecks(dc, op, [X], [0, 1])
        self.assertReferenceChecks(gc, op, [X], histogram)


if __name__ == "__main__":
    unittest.main()
