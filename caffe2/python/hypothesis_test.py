from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy
from functools import reduce
from hypothesis import assume, given, settings
import hypothesis.strategies as st

from functools import partial
import unittest

from caffe2.python import core, workspace, tt_core
import caffe2.python.hypothesis_test_util as hu
from caffe2.proto.caffe2_pb2 import TensorProto


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x):
    return 2.0 * sigmoid(2.0 * x) - 1


def lstm_unit(cell_t_m_1, gates, seq_lengths, timestep):
    D = cell_t_m_1.shape[2]
    G = gates.shape[2]
    N = gates.shape[1]
    t = (timestep[0].reshape(1, 1) * np.ones(shape=(N, D))).astype(np.int32)
    assert t.shape == (N, D)
    seq_lengths = (np.ones(shape=(N, D)) *
                   seq_lengths.reshape(N, 1)).astype(np.int32)
    assert seq_lengths.shape == (N, D)
    assert G == 4 * D
    # Resize to avoid broadcasting inconsistencies with NumPy
    gates = gates.reshape(N, 4, D)
    cell_t_m_1 = cell_t_m_1.reshape(N, D)
    i_t = gates[:, 0, :].reshape(N, D)
    f_t = gates[:, 1, :].reshape(N, D)
    o_t = gates[:, 2, :].reshape(N, D)
    g_t = gates[:, 3, :].reshape(N, D)
    i_t = sigmoid(i_t)
    f_t = sigmoid(f_t)
    o_t = sigmoid(o_t)
    g_t = tanh(g_t)
    valid = (seq_lengths < t).astype(np.int32)
    assert valid.shape == (N, D)
    cell_t = ((f_t * cell_t_m_1) + (i_t * g_t)) * (valid) + \
        (1 - valid) * cell_t_m_1
    assert cell_t.shape == (N, D)
    hidden_t = (o_t * tanh(cell_t)) * valid
    hidden_t = hidden_t.reshape(1, N, D)
    cell_t = cell_t.reshape(1, N, D)
    return hidden_t, cell_t


@st.composite
def _tensor_and_prefix(draw, dtype, elements, min_dim=1, max_dim=4, **kwargs):
    dims_ = draw(
        st.lists(hu.dims(**kwargs), min_size=min_dim, max_size=max_dim))
    extra_ = draw(
        st.lists(hu.dims(**kwargs), min_size=min_dim, max_size=max_dim))
    return (draw(hu.arrays(dims_ + extra_, dtype, elements)),
            draw(hu.arrays(extra_, dtype, elements)))


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


def _dtypes():
    return st.sampled_from([np.int32, np.int64, np.float32, np.float64])


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
        for name, ref in ops.items():
            _test_binary(name, ref, gcs=hu.gcs_cpu_only)(self)
            _test_binary_broadcast(name, ref, gcs=hu.gcs_cpu_only)(self)

    @given(inputs=hu.tensors(n=2), in_place=st.booleans(), **hu.gcs)
    def test_sum(self, inputs, in_place, gc, dc):
        op = core.CreateOperator("Sum", ["X1", "X2"],
                                        ["Y" if not in_place else "X1"])
        X1, X2 = inputs
        self.assertDeviceChecks(dc, op, [X1, X2], [0])
        self.assertGradientChecks(gc, op, [X1, X2], 0, [0])

    def test_add(self):
        def ref(x, y):
            return (x + y, )
        _test_binary("Add", ref, test_gradient=True)(self)
        _test_binary_broadcast("Add", ref)(self)

    def test_sub(self):
        def ref(x, y):
            return (x - y, )
        # TODO(jiayq): enable gradient test when implemented.
        _test_binary("Sub", ref, test_gradient=True)(self)
        _test_binary_broadcast("Sub", ref)(self)

    def test_mul(self):
        def ref(x, y):
            return (x * y, )
        _test_binary("Mul", ref, test_gradient=True)(self)
        _test_binary_broadcast("Mul", ref)(self)

    def test_div(self):
        def ref(x, y):
            return (x / y, )

        def non_zero(x):
            return abs(x) > 10e-5

        def div_dtypes():
            return st.sampled_from([np.float32, np.float64])

        _test_binary(
            "Div", ref, filter_=non_zero, test_gradient=True, dtypes=div_dtypes
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
    def test_relu(self, X, gc, dc):
        op = core.CreateOperator("Relu", ["X"], ["Y"])
        # go away from the origin point to avoid kink problems
        X += 0.02 * np.sign(X)
        X[X == 0.0] += 0.02
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0])

    @given(X=hu.tensor(), **hu.gcs)
    def test_averaged_loss(self, X, gc, dc):
        op = core.CreateOperator("AveragedLoss", ["X"], ["loss"])
        self.assertDeviceChecks(dc, op, [X], [0])
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
            op = core.CreateOperator(
                "XavierFill", [], ["Y"],
                device_option=do,
                shape=[2])
            workspace.RunOperatorOnce(op)
            return workspace.FetchBlob("Y")
        ys = [run(do) for do in device_options]
        for y in ys[1:]:
            if set_seed:
                np.testing.assert_array_equal(ys[0], y)
            else:
                with self.assertRaises(AssertionError):
                    np.testing.assert_array_equal(ys[0], y)

    @given(axis=st.integers(min_value=1, max_value=4),
           num_output=st.integers(min_value=4, max_value=8),
           **hu.gcs)
    def test_fully_connected_axis(self, axis, num_output, gc, dc):
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
            axis=axis)
        for name, param in [("X", X), ("W", W), ("b", b)]:
            workspace.FeedBlob(name, param)
        workspace.RunOperatorOnce(op)
        Y = workspace.FetchBlob("Y")
        self.assertEqual(list(Y.shape), list(X.shape)[:axis] + [N])

        inputs = [X, W, b]
        self.assertDeviceChecks(dc, op, inputs, [0])
        for param, _ in enumerate(inputs):
            self.assertGradientChecks(gc, op, inputs, param, [0])

    @unittest.skipIf(True,
                     "Recurrent only works on CUDA 7.5 and above")
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
        init_op = core.CreateOperator(
            "RecurrentInit",
            ["INPUT"],
            ["WEIGHT", "DROPOUT_STATES"],
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            rnn_mode=rnn_mode,
            dropout=dropout,
            input_mode=input_mode,
            num_layers=num_layers,
            device_option=hu.gpu_do,
            engine="CUDNN")

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
            engine="CUDNN")
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

    @given(ndim=st.integers(1, 4),
           axis=st.integers(0, 3),
           num_inputs=st.integers(2, 4), **hu.gcs)
    def test_depth_concat(self, ndim, axis, num_inputs, gc, dc):
        if (axis >= ndim):
            return
        input_names = ['X0', 'X1', 'X2', 'X3'][:num_inputs]
        shape = [2, 3, 5, 7][:ndim]
        individual_dims = [11, 13, 17, 19][:num_inputs]
        inputs = []
        for i in range(num_inputs):
            # Sets a unique dim and create the input.
            shape[axis] = individual_dims[i]
            inputs.append(np.random.rand(*shape).astype(np.float32))
        op = core.CreateOperator("Concat", input_names, ["Y", "Y_dims"],
                                 axis=axis)
        self.assertDeviceChecks(dc, op, inputs, [0])
        for i in range(num_inputs):
            self.assertGradientChecks(gc, op, inputs, i, [0])

    @given(num_inputs=st.integers(2, 4),
           order=st.sampled_from([("NCHW", 1), ("NHWC", 3)]),
           **hu.gcs)
    def test_depth_concat_with_order(self, num_inputs, order, gc, dc):
        input_names = ['X0', 'X1', 'X2', 'X3'][:num_inputs]
        shape = [2, 3, 5, 7]
        individual_dims = [11, 13, 17, 19][:num_inputs]
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

    # CUDNN does NOT support different padding values and we skip it
    @given(stride_h=st.integers(1, 3),
            stride_w=st.integers(1, 3),
            pad_t=st.integers(0, 3),
            pad_l=st.integers(0, 3),
            pad_b=st.integers(0, 3),
            pad_r=st.integers(0, 3),
            kernel=st.integers(1, 5),
            size=st.integers(7, 10),
            input_channels=st.integers(1, 8),
            output_channels=st.integers(1, 8),
            batch_size=st.integers(1, 3),
            order=st.sampled_from(["NCHW", "NHWC"]),
            engine=st.sampled_from([""]),
            **hu.gcs)
    @settings(max_examples=2, timeout=100)
    def test_convolution_separate_stride_pad_gradients(self, stride_h, stride_w,
                                                        pad_t, pad_l, pad_b,
                                                        pad_r, kernel, size,
                                                        input_channels,
                                                        output_channels,
                                                        batch_size, order,
                                                        engine, gc, dc):
        assume(stride_h <= kernel)
        assume(stride_w <= kernel)
        op = core.CreateOperator(
            "Conv",
            ["X", "w", "b"],
            ["Y"],
            stride_h=stride_h,
            stride_w=stride_w,
            pad_t=pad_t,
            pad_l=pad_l,
            pad_b=pad_b,
            pad_r=pad_r,
            kernel=kernel,
            order=order,
            engine=engine,
        )
        X = np.random.rand(
            batch_size, size, size, input_channels).astype(np.float32) - 0.5
        w = np.random.rand(
            output_channels, kernel, kernel, input_channels).astype(np.float32)\
            - 0.5
        b = np.random.rand(output_channels).astype(np.float32) - 0.5
        if order == "NCHW":
            X = X.transpose((0, 3, 1, 2))
            w = w.transpose((0, 3, 1, 2))

        self.assertDeviceChecks(dc, op, [X, w, b], [0])
        for i in range(3):
            self.assertGradientChecks(gc, op, [X, w, b], i, [0])

    # CUDNN does NOT support different padding values and we skip it
    @given(stride_h=st.integers(1, 3),
            stride_w=st.integers(1, 3),
            pad_t=st.integers(0, 3),
            pad_l=st.integers(0, 3),
            pad_b=st.integers(0, 3),
            pad_r=st.integers(0, 3),
            kernel=st.integers(1, 5),
            size=st.integers(7, 10),
            input_channels=st.integers(1, 8),
            output_channels=st.integers(1, 8),
            batch_size=st.integers(1, 3),
            engine=st.sampled_from([""]), **hu.gcs)
    def test_convolution_separate_stride_pad_layout(self, stride_h, stride_w,
                                                    pad_t, pad_l, pad_b, pad_r,
                                                    kernel, size,
                                                    input_channels,
                                                    output_channels, batch_size,
                                                    engine, gc, dc):
        assume(stride_h <= kernel)
        assume(stride_w <= kernel)
        X = np.random.rand(
            batch_size, size, size, input_channels).astype(np.float32) - 0.5
        w = np.random.rand(
            output_channels, kernel, kernel, input_channels).astype(np.float32)\
            - 0.5
        b = np.random.rand(output_channels).astype(np.float32) - 0.5
        outputs = {}
        for order in ["NCHW", "NHWC"]:
            op = core.CreateOperator(
                "Conv",
                ["X", "w", "b"],
                ["Y"],
                stride_h=stride_h,
                stride_w=stride_w,
                kernel=kernel,
                pad_t=pad_t,
                pad_l=pad_l,
                pad_b=pad_b,
                pad_r=pad_r,
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

    @given(stride=st.integers(1, 3),
           pad=st.integers(0, 3),
           kernel=st.integers(1, 5),
           size=st.integers(7, 10),
           input_channels=st.integers(1, 8),
           output_channels=st.integers(1, 8),
           batch_size=st.integers(1, 3),
           order=st.sampled_from(["NCHW", "NHWC"]),
           engine=st.sampled_from(["", "CUDNN"]),
           **hu.gcs)
    @settings(max_examples=2, timeout=100)
    def test_convolution_gradients(self, stride, pad, kernel, size,
                                   input_channels, output_channels, batch_size,
                                   order, engine, gc, dc):
        assume(stride <= kernel)
        op = core.CreateOperator(
            "Conv",
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
            output_channels, kernel, kernel, input_channels).astype(np.float32)\
            - 0.5
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
            output_channels, kernel, kernel, input_channels).astype(np.float32)\
            - 0.5
        b = np.random.rand(output_channels).astype(np.float32) - 0.5
        outputs = {}
        for order in ["NCHW", "NHWC"]:
            op = core.CreateOperator(
                "Conv",
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

    @given(num_workers=st.integers(1, 4),
           net_type=st.sampled_from(
               ["simple", "dag"] +
               (["async_dag"] if workspace.has_gpu_support else [])),
           do=st.sampled_from(hu.device_options),
           engine=st.sampled_from(["CUDNN", ""]))
    def test_convolution_sync(self, net_type, num_workers, do, engine):
        from caffe2.python.cnn import CNNModelHelper
        m = CNNModelHelper()
        n = 1
        d = 2
        depth = 3
        iters = 5
        h = 5
        w = 5
        workspace.ResetWorkspace()

        np.random.seed(1701)
        # Build a binary tree of conv layers, summing at each node.
        for i in reversed(range(depth)):
            for j in range(2 ** i):
                bottom_1 = "{}_{}".format(i + 1, 2 * j)
                bottom_2 = "{}_{}".format(i + 1, 2 * j + 1)
                mid_1 = "{}_{}_m".format(i + 1, 2 * j)
                mid_2 = "{}_{}_m".format(i + 1, 2 * j + 1)
                top = "{}_{}".format(i, j)
                w1, b1, w2, b2 = np.random.randn(4).tolist()
                m.Conv(
                    bottom_1, mid_1,
                    dim_in=d, dim_out=d,
                    kernel=3,
                    weight_init=m.ConstantInit(w1),
                    bias_init=m.ConstantInit(b1),
                    cudnn_state=np.random.randint(0, 3),
                    stride=1,
                    pad=1,
                    deterministic=1,
                    engine=engine)
                m.Conv(
                    bottom_2, mid_2,
                    dim_in=d, dim_out=d,
                    kernel=3,
                    stride=1,
                    pad=1,
                    weight_init=m.ConstantInit(w2),
                    bias_init=m.ConstantInit(b2),
                    deterministic=1,
                    cudnn_state=np.random.randint(0, 3),
                    engine=engine)
                m.net.Sum([mid_1, mid_2], top)

        m.net.Flatten(["0_0"], ["0_0_flat"])
        m.net.SquaredL2Distance(["0_0_flat", "label"], "xent")
        m.net.AveragedLoss("xent", "loss")
        input_to_grad = m.AddGradientOperators(["loss"])
        m.Proto().device_option.CopyFrom(do)
        m.param_init_net.Proto().device_option.CopyFrom(do)
        m.Proto().type = net_type
        m.Proto().num_workers = num_workers
        workspace.RunNetOnce(m.param_init_net)

        def run():
            import numpy as np
            np.random.seed(1701)
            input_blobs = ["{}_{}".format(depth, j) for j in range(2 ** depth)]
            for input_blob in input_blobs:
                workspace.FeedBlob(
                    input_blob,
                    np.random.randn(n, d, h, w).astype(np.float32),
                    device_option=do)
                workspace.FeedBlob(
                    "label",
                    np.random.randn(n, d * h * w).astype(np.float32),
                    device_option=do)
            workspace.RunNetOnce(m.net)
            gradients = [
                workspace.FetchBlob(str(input_to_grad[input_blob]))
                for input_blob in input_blobs]
            return gradients

        outputs = [run() for _ in range(iters)]
        for output in outputs[1:]:
            np.testing.assert_array_equal(outputs[0], output)
            np.testing.assert_allclose(
                np.sum(np.square(output)),
                1763719461732352.0,
                rtol=1e-5)

    @given(stride=st.integers(1, 3),
            pad=st.integers(0, 3),
            kernel=st.integers(1, 5),
            size=st.integers(7, 10),
            input_channels=st.integers(1, 8),
            output_channels=st.integers(1, 8),
            batch_size=st.integers(1, 3),
            order=st.sampled_from(["NCHW", "NHWC"]),
            engine=st.sampled_from([""]), **hu.gcs)
    def test_convolution_transpose_gradients(self, stride, pad, kernel,
                                                size, input_channels,
                                                output_channels, batch_size,
                                                order, engine, gc, dc):
        assume(stride <= kernel)
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
            order=order,
            engine=engine,
        )
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
            engine=st.sampled_from([""]), **hu.gcs)
    def test_convolution_transpose_layout(self, stride, pad, kernel,
                                            size, input_channels,
                                            output_channels, batch_size,
                                            engine, gc, dc):
        assume(stride <= kernel)
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

    @given(dtype=st.sampled_from([np.float32, np.float64, np.int32, np.bool]))
    def test_print(self, dtype):
        data = np.random.permutation(6).astype(dtype)
        workspace.FeedBlob("data", data)
        op = core.CreateOperator("Print", "data", [])
        self.assertTrue(workspace.RunOperatorOnce(op))

    @given(inputs=hu.tensors(n=3),
           in_place=st.booleans(),
           beta1=st.floats(min_value=0.1, max_value=0.9),
           beta2=st.floats(min_value=0.1, max_value=0.9),
           lr=st.floats(min_value=0.1, max_value=0.9),
           iters=st.integers(min_value=1, max_value=10000),
           epsilon=st.floats(min_value=1e-5, max_value=1e-2),
           **hu.gcs)
    def test_adam_sgd(self, inputs, in_place, beta1, beta2, lr, iters, epsilon,
                      gc, dc):
        grad, m1, m2 = inputs
        m2 += np.abs(m2) + 0.01
        lr = np.asarray([lr], dtype=np.float32)
        iters = np.asarray([iters], dtype=np.int32)
        op = core.CreateOperator(
            "Adam",
            ["grad", "m1", "m2", "lr", "iters"],
            ["grad" if in_place else "grad_o",
             "m1" if in_place else "m1_o",
             "m2" if in_place else "m2_o"],
            beta1=beta1, beta2=beta2, epsilon=epsilon,
            device_option=gc)
        input_device_options = {"iters": hu.cpu_do}
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

    @staticmethod
    def _dense_adagrad(epsilon, grad, h, lr):
        lr = lr[0]
        h_o = h + np.square(grad)
        grad_o = lr * grad / (np.sqrt(h_o) + epsilon)
        return (grad_o, h_o)

    @given(inputs=hu.tensors(n=2),
           in_place=st.booleans(),
           lr=st.floats(min_value=0.1, max_value=0.9),
           epsilon=st.floats(min_value=1e-5, max_value=1e-2),
           **hu.gcs)
    def test_adagrad_sgd(self, inputs, in_place, lr, epsilon,
                         gc, dc):
        grad, h = inputs
        h = np.abs(h) + 0.01
        lr = np.asarray([lr], dtype=np.float32)
        op = core.CreateOperator(
            "Adagrad",
            ["grad", "h", "lr"],
            ["grad" if in_place else "grad_o",
             "h" if in_place else "h_o"],
            epsilon=epsilon, device_option=gc)
        self.assertDeviceChecks(dc, op, [grad, h, lr], [0])

        self.assertReferenceChecks(gc, op, [grad, h, lr],
                                   partial(self._dense_adagrad, epsilon))

    @given(inputs=hu.tensors(n=2),
           lr=st.floats(min_value=0.1, max_value=0.9),
           epsilon=st.floats(min_value=1e-5, max_value=1e-2),
           **hu.gcs_cpu_only)
    def test_sparse_adagrad_sgd(self, inputs, lr, epsilon,
                                gc, dc):
        grad, h = inputs
        indices = np.arange(h.shape[0])
        indices = indices[indices % 2 == 0]
        grad = grad[indices]
        h = np.abs(h)
        lr = np.asarray([lr], dtype=np.float32)
        op = core.CreateOperator(
            "SparseAdagrad",
            ["indices", "grad", "h", "lr"],
            ["grad", "h"],
            epsilon=epsilon,
            device_option=gc)
        self.assertDeviceChecks(
            dc, op, [indices, grad, h, lr], [0])

        def adagrad(i, grad, h, lr):
            sg, sh = self._dense_adagrad(epsilon, grad, h[i], lr)
            h[i] = sh
            return (sg, h)

        self.assertReferenceChecks(gc, op, [indices, grad, h, lr], adagrad)

    # Reference
    @staticmethod
    def _dense_ftrl(alpha, beta, lambda1, lambda2, w, nz, g):
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

    @given(input=hu.tensor(max_value=20,
                           max_dim=1,
                           dtype=np.int32,
                           elements=st.integers(min_value=0, max_value=10)),
           with_remapping=st.booleans(),
           **hu.gcs_cpu_only)
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
            **hu.gcs)
    def test_accuracy(self, prediction, labels, gc, dc):
        op = core.CreateOperator(
            "Accuracy",
            ["prediction", "labels"],
            ["accuracy"]
        )

        def op_ref(prediction, labels):
            N = prediction.shape[0]
            correct = 0
            max_ids = np.argmax(prediction, axis=1)
            for i in range(0, N):
                if max_ids[i] == labels[i]:
                    correct += 1
            accuracy = correct / N
            return (accuracy,)

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[prediction, labels],
            reference=op_ref)

    @given(target_probabilities=hu.arrays(
        dims=[10], elements=st.floats(allow_nan=False,
                                      allow_infinity=False,
                                      min_value=0,
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
            return (np.array(sids, dtype=int), )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[np.array(lengths, dtype=int)],
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
                return (np.array([], dtype=int),)

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
            return (np.array(lengths, dtype=int),)

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[np.array(segment_ids, dtype=int)],
            reference=ids_to_lengths)

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
        import Queue
        op = core.CreateOperator(
            "CreateBlobsQueue",
            [],
            ["queue"],
            capacity=capacity,
            num_blobs=num_blobs,
            device_option=do)
        workspace.RunOperatorOnce(op)

        xs = [np.random.randn(num_elements, 5).astype(np.float32)
              for _ in range(num_blobs)]
        q = Queue.Queue()
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
                        workspace.FeedBlob(feed_blob, elem, device_option=do)
                    workspace.RunOperatorOnce(op)
                except Queue.Empty:
                    return

        # Create all blobs before racing on multiple threads
        # (blob creation is not threadsafe)
        for t in range(num_threads):
            for i in range(num_blobs):
                workspace.CreateBlob("x_{}_{}".format(i, t))

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
            workspace.RunOperatorOnce(op)
        for thread in threads:
            thread.join()
        op = core.CreateOperator("CloseBlobsQueue", ["queue"], [])
        workspace.RunOperatorOnce(op)
        ys = [np.vstack([workspace.FetchBlob("y_{}_{}".format(i, n))
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

    @given(
        data=hu.tensor(),
        **hu.gcs_cpu_only)
    def test_squeeze_expand_dims(self, data, gc, dc):
            dims = [0]
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

        workspace.FeedBlob("X", X)
        workspace.FeedBlob("b", b)
        workspace.FeedBlob("cores", cores)
        workspace.RunOperatorOnce(op)

        Y = workspace.FetchBlob("Y")
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
        self.assertAlmostEqual(np.linalg.norm(golden - Y), 0, delta=1e-12)

    @given(num_workers=st.integers(1, 10),
           net_type=st.sampled_from(
               ["simple", "dag"] +
               (["async_dag"] if workspace.has_gpu_support else [])),
           do=st.sampled_from(hu.device_options))
    def test_dag_net_forking(self, net_type, num_workers, do):
        from caffe2.python.cnn import CNNModelHelper
        m = CNNModelHelper()
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
                m.FC(
                    bottom_1, mid_1,
                    dim_in=d, dim_out=d,
                    weight_init=m.ConstantInit(np.random.randn()),
                    bias_init=m.ConstantInit(np.random.randn()))
                m.FC(
                    bottom_2, mid_2,
                    dim_in=d, dim_out=d,
                    weight_init=m.ConstantInit(np.random.randn()),
                    bias_init=m.ConstantInit(np.random.randn()))
                m.net.Sum([mid_1, mid_2], top)
        m.net.SquaredL2Distance(["0_0", "label"], "xent")
        m.net.AveragedLoss("xent", "loss")
        input_to_grad = m.AddGradientOperators(["loss"])
        m.Proto().device_option.CopyFrom(do)
        m.param_init_net.Proto().device_option.CopyFrom(do)

        m.Proto().type = net_type
        m.Proto().num_workers = num_workers

        workspace.RunNetOnce(m.param_init_net)

        print(str(m.Proto()))

        def run():
            import numpy as np
            np.random.seed(1701)
            input_blobs = ["{}_{}".format(depth, j) for j in range(2 ** depth)]
            for input_blob in input_blobs:
                workspace.FeedBlob(
                    input_blob,
                    np.random.randn(n, d).astype(np.float32),
                    device_option=do)
                workspace.FeedBlob(
                    "label",
                    np.random.randn(n, d).astype(np.float32),
                    device_option=do)
            workspace.RunNetOnce(m.net)
            gradients = [
                workspace.FetchBlob(str(input_to_grad[input_blob]))
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
           **hu.gcs_cpu_only)
    def test_slice(self, input, slice_dim, a, b, gc, dc):
        slice_dim %= len(input.shape)
        a %= input.shape[slice_dim]
        b %= input.shape[slice_dim] + 1
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
    def test_has_elements(self, data, gc, dc):
        op = core.CreateOperator("HasElements", ["data"], ["has_elements"])
        self.assertReferenceChecks(gc, op, [data], lambda x: (len(x) > 0, ))

    @given(initial_iters=st.integers(0, 100),
           max_iters=st.integers(0, 100))
    def test_criteria_net_with_execution_step(self, initial_iters, max_iters):
        net = core.Net("net")
        net.Iter(["iter"], ["iter"])
        workspace.FeedBlob(
            "iter", np.asarray([initial_iters]).astype(np.int32))
        workspace.FeedBlob(
            "num_iters", np.asarray([max_iters]).astype(np.int32))
        criteria_net = core.Net("criteria")
        criteria_net.LT(["iter", "num_iters"], ["continue"])
        criteria_net.Proto().external_output.extend(["continue"])

        plan = core.Plan('plan')
        plan.AddStep(core.execution_step('step', net, criteria=criteria_net))
        workspace.RunPlan(plan)
        iters = workspace.FetchBlob("iter")
        self.assertEqual(iters.dtype, np.int32)
        self.assertEqual(iters[0], max(initial_iters, max_iters))

    @given(initial_iters=st.integers(0, 100),
           num_iters=st.integers(0, 100))
    def test_iter_count_with_execution_step(self, initial_iters, num_iters):
        net = core.Net("net")
        net.Iter(["iter"], ["iter"])
        workspace.FeedBlob(
            "iter", np.asarray([initial_iters]).astype(np.int32))

        step = core.ExecutionStep("step", [net])
        step.SetIter(num_iters)

        plan = core.Plan("plan")
        plan.AddStep(step)
        workspace.RunPlan(plan)
        iters = workspace.FetchBlob("iter")
        self.assertEqual(iters.dtype, np.int32)
        self.assertEqual(iters[0], initial_iters + num_iters)

    @given(a=hu.tensor(),
           src=st.sampled_from(_NUMPY_TYPE_TO_ENUM.keys()),
           dst=st.sampled_from(_NUMPY_TYPE_TO_ENUM.keys()),
           use_name=st.booleans(),
           **hu.gcs)
    def test_cast(self, a, src, dst, use_name, gc, dc):
        a = a.astype(src)

        def ref(data):
            return [data.astype(dst)]

        to = _NUMPY_TYPE_TO_ENUM[dst]
        if use_name:
            to = TensorProto.DataType.Name(to).lower()
        op = core.CreateOperator('Cast', ["X"], ["Y"], to=to)
        self.assertDeviceChecks(dc, op, [a], [0])
        out, = self.assertReferenceChecks(gc, op, [a], ref)
        self.assertEqual(dst, out.dtype)

    @given(n=st.integers(1, 10),
           d=st.integers(1, 10),
           t=st.integers(1, 10),
           **hu.gcs)
    def test_lstm_unit_recurrent_network(self, n, d, t, dc, gc):
        op = core.CreateOperator(
            "LSTMUnit",
            ["cell_t-1", "gates_t", "seq_lengths", "timestep"],
            ["hidden_t", "cell_t"])
        cell_t_m_1 = np.random.randn(1, n, d).astype(np.float32)
        gates = np.random.randn(1, n, 4 * d).astype(np.float32)
        seq_lengths = np.random.randint(0, t, size=(n,)).astype(np.int32)
        timestep = np.random.randint(0, t, size=(1,)).astype(np.int32)
        inputs = [cell_t_m_1, gates, seq_lengths, timestep]
        input_device_options = {"timestep": hu.cpu_do}
        self.assertDeviceChecks(
            dc, op, inputs, [0],
            input_device_options=input_device_options)
        self.assertReferenceChecks(
            gc, op, inputs, lstm_unit,
            input_device_options=input_device_options)
        for i in range(2):
            self.assertGradientChecks(
                gc, op, inputs, i, [0, 1],
                input_device_options=input_device_options)

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

    @given(X=hu.tensor(), seed=st.integers(min_value=0, max_value=65536),
           null_axes=st.booleans(),
           **hu.gcs)
    def test_transpose(self, X, seed, null_axes, gc, dc):
        if null_axes:
            axes = None
            op = core.CreateOperator("Transpose", "input", "output")
        else:
            np.random.seed(int(seed))
            axes = [int(v) for v in list(np.random.permutation(X.ndim))]
            op = core.CreateOperator(
                "Transpose", "input", "output", axes=axes)

        def transpose_ref(x, axes):
            return (np.transpose(x, axes),)

        self.assertReferenceChecks(gc, op, [X, axes],
                                   transpose_ref)
        self.assertGradientChecks(gc, op, [X], 0, [0])

    @given(n=st.integers(1, 3),
           dim=st.integers(4, 16),
           **hu.gcs_cpu_only)
    def test_distances(self, n, dim, gc, dc):
        X = np.random.uniform(-1, 1, (n, dim)).astype(np.float32)
        Y = np.random.uniform(-1, 1, (n, dim)).astype(np.float32)
        workspace.FeedBlob("X", X)
        workspace.FeedBlob("Y", Y)

        def check_grad(op):
            self.assertGradientChecks(gc, op, [X, Y], 0, [0],
                                      stepsize=1e-2, threshold=1e-2)
            self.assertGradientChecks(gc, op, [X, Y], 1, [0],
                                      stepsize=1e-2, threshold=1e-2)

        l2_op = core.CreateOperator("SquaredL2Distance",
                                    ["X", "Y"], ["l2_dist"])
        workspace.RunOperatorOnce(l2_op)
        np.testing.assert_allclose(workspace.FetchBlob("l2_dist"),
                                   np.square(X - Y).sum(axis=1) * 0.5,
                                   rtol=1e-4, atol=1e-4)
        check_grad(l2_op)

        dot_op = core.CreateOperator("DotProduct", ["X", "Y"], ["dot"])
        workspace.RunOperatorOnce(dot_op)
        np.testing.assert_allclose(workspace.FetchBlob("dot"),
                                   np.multiply(X, Y).sum(axis=1),
                                   rtol=1e-4, atol=1e-4)
        check_grad(dot_op)

        kEps = 1e-12
        cos_op = core.CreateOperator("CosineSimilarity", ["X", "Y"], ["cos"])
        workspace.RunOperatorOnce(cos_op)
        cos = np.divide(np.multiply(X, Y).sum(axis=1),
                        np.multiply(np.linalg.norm(X, axis=1) + kEps,
                                    np.linalg.norm(Y, axis=1) + kEps))
        np.testing.assert_allclose(workspace.FetchBlob("cos"), cos,
                                   rtol=1e-4, atol=1e-4)
        check_grad(cos_op)
