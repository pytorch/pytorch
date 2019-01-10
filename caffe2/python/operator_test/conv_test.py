from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools

import numpy as np
from hypothesis import assume, given
import hypothesis.strategies as st

from caffe2.proto import caffe2_pb2
from caffe2.python import brew, core, workspace
import caffe2.python.hypothesis_test_util as hu
from caffe2.python.model_helper import ModelHelper


def _cudnn_supports(
        dilation=False,
        nhwc=False,
        backward=False,
):
    """Return True if cuDNN supports this configuration."""
    v = workspace.GetCuDNNVersion()
    if backward:
        if nhwc:
            # nhwc isn't supported in backward ops.
            return False
    else:
        # Forward mode.
        if dilation and v < 6000:
            # Dilation not supported until v6
            return False
        if dilation and nhwc:
            # Dilation and NHWC not supported together
            return False
    return True


class TestConvolution(hu.HypothesisTestCase):
    # CUDNN does NOT support different padding values and we skip it
    @given(op_type=st.sampled_from(["Conv", "Conv2D"]),
           stride_h=st.integers(1, 3),
           stride_w=st.integers(1, 3),
           pad_t=st.integers(0, 3),
           pad_l=st.integers(0, 3),
           pad_b=st.integers(0, 3),
           pad_r=st.integers(0, 3),
           kernel=st.integers(3, 5),
           size=st.integers(1, 8),
           input_channels=st.integers(1, 3),
           output_channels=st.integers(1, 3),
           batch_size=st.integers(1, 3),
           order=st.sampled_from(["NCHW", "NHWC"]),
           engine=st.sampled_from(["", "EIGEN"]),
           shared_buffer=st.booleans(),
           use_bias=st.booleans(),
           **hu.gcs)
    def test_convolution_separate_stride_pad_gradients(self, op_type,
                                                       stride_h, stride_w,
                                                       pad_t, pad_l, pad_b,
                                                       pad_r, kernel, size,
                                                       input_channels,
                                                       output_channels,
                                                       batch_size, order,
                                                       engine, shared_buffer,
                                                       use_bias,
                                                       gc, dc):
        op = core.CreateOperator(
            op_type,
            ["X", "w", "b"] if use_bias else ["X", "w"],
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
            shared_buffer=int(shared_buffer),
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

        inputs = [X, w, b] if use_bias else [X, w]

        # Error handling path.
        if size + pad_r + pad_l < kernel or size + pad_t + pad_b < kernel:
            with self.assertRaises(RuntimeError):
                self.assertDeviceChecks(dc, op, inputs, [0])
            return

        self.assertDeviceChecks(dc, op, inputs, [0])
        for i in range(len(inputs)):
            self.assertGradientChecks(gc, op, inputs, i, [0])

    # CUDNN does NOT support different padding values and we skip it
    @given(op_type=st.sampled_from(["Conv", "Conv2D"]),
           stride_h=st.integers(1, 3),
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
           engine=st.sampled_from(["", "EIGEN"]),
           use_bias=st.booleans(),
           **hu.gcs)
    def test_convolution_separate_stride_pad_layout(self, op_type,
                                                    stride_h, stride_w,
                                                    pad_t, pad_l, pad_b, pad_r,
                                                    kernel, size,
                                                    input_channels,
                                                    output_channels, batch_size,
                                                    engine, use_bias, gc, dc):
        X = np.random.rand(
            batch_size, size, size, input_channels).astype(np.float32) - 0.5
        w = np.random.rand(
            output_channels, kernel, kernel, input_channels).astype(np.float32)\
            - 0.5
        b = np.random.rand(output_channels).astype(np.float32) - 0.5
        outputs = {}
        for order in ["NCHW", "NHWC"]:
            op = core.CreateOperator(
                op_type,
                ["X", "w", "b"] if use_bias else ["X", "w"],
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
            self.ws.create_blob("X").feed(X_f, device_option=gc)
            self.ws.create_blob("w").feed(w_f, device_option=gc)
            self.ws.create_blob("b").feed(b, device_option=gc)
            self.ws.run(op)
            outputs[order] = self.ws.blobs["Y"].fetch()
        np.testing.assert_allclose(
            outputs["NCHW"],
            outputs["NHWC"].transpose((0, 3, 1, 2)),
            atol=1e-4,
            rtol=1e-4)

    @given(op_type=st.sampled_from(["Conv", "Conv2D"]),
           stride=st.integers(1, 3),
           pad=st.integers(0, 3),
           kernel=st.integers(1, 5),
           dilation=st.integers(1, 3),
           size=st.integers(7, 10),
           input_channels=st.integers(1, 8),
           output_channels=st.integers(1, 8),
           batch_size=st.integers(1, 3),
           order=st.sampled_from(["NCHW", "NHWC"]),
           engine=st.sampled_from(["", "CUDNN", "MKLDNN"]),
           use_bias=st.booleans(),
           **hu.gcs)
    def test_convolution_gradients(self, op_type, stride, pad, kernel, dilation,
                                   size, input_channels, output_channels,
                                   batch_size, order, engine, use_bias, gc, dc):
        dkernel = dilation * (kernel - 1) + 1

        if engine == 'CUDNN':
            assume(_cudnn_supports(dilation=(dilation > 1),
                                   nhwc=(order == 'NHWC'),
                                   backward=True))

        assume(engine != "MKLDNN" or use_bias is True)

        op = core.CreateOperator(
            op_type,
            ["X", "w", "b"] if use_bias else ["X", "w"],
            ["Y"],
            stride=stride,
            kernel=kernel,
            dilation=dilation,
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

        inputs = [X, w, b] if use_bias else [X, w]
        # Error handling path.
        if size + pad + pad < dkernel or size + pad + pad < dkernel:
            with self.assertRaises(RuntimeError):
                self.assertDeviceChecks(dc, op, inputs, [0])
            return

        self.assertDeviceChecks(dc, op, inputs, [0])
        for i in range(len(inputs)):
            self.assertGradientChecks(gc, op, inputs, i, [0])

    def _nd_convolution_nchw(self, n, input_channels, output_channels,
                            batch_size, stride, size, kernel, dilation, pad,
                            use_bias, gc, dc):
        dkernel = dilation * (kernel - 1) + 1
        for op_type in ["Conv", "Conv" + str(n) + "D"]:
            op = core.CreateOperator(
                op_type,
                ["X", "w", "b"] if use_bias else ["X", "w"],
                ["Y"],
                strides=[stride] * n,
                kernels=[kernel] * n,
                dilations=[dilation] * n,
                pads=[pad] * n * 2,
                order="NCHW",
                engine="",
            )

            input_dims = [batch_size, input_channels]
            input_dims.extend([size] * n)
            filter_dims = [output_channels, input_channels]
            filter_dims.extend([kernel] * n)

            X = np.random.rand(*input_dims).astype(np.float32) - 0.5
            w = np.random.rand(*filter_dims).astype(np.float32) - 0.5
            b = np.random.rand(output_channels).astype(np.float32) - 0.5

            inputs = [X, w, b] if use_bias else [X, w]

            if size + pad + pad < dkernel or size + pad + pad < dkernel:
                with self.assertRaises(RuntimeError):
                    self.assertDeviceChecks(dc, op, inputs, [0])
                return

            self.assertDeviceChecks(dc, op, inputs, [0])
            for i in range(len(inputs)):
                self.assertGradientChecks(gc, op, inputs, i, [0])

    @given(input_channels=st.integers(1, 3),
           output_channels=st.integers(1, 2),
           batch_size=st.integers(1, 3),
           stride=st.integers(1, 3),
           size=st.integers(7, 10),
           kernel=st.integers(1, 2),
           dilation=st.integers(1, 3),
           pad=st.integers(0, 3),
           use_bias=st.booleans(),
           **hu.gcs)
    def test_1d_convolution_nchw(self, input_channels, output_channels,
                                 batch_size, stride, size, kernel, dilation,
                                 pad, use_bias, gc, dc):
        self._nd_convolution_nchw(
            1, input_channels, output_channels, batch_size, stride, size,
            kernel, dilation, pad, use_bias, gc, dc
        )

    @given(input_channels=st.integers(1, 2),
           output_channels=st.integers(1, 2),
           batch_size=st.integers(1, 2),
           stride=st.integers(1, 2),
           size=st.integers(4, 5),
           kernel=st.integers(1, 2),
           dilation=st.integers(1, 2),
           pad=st.integers(0, 2),
           use_bias=st.booleans(),
           **hu.gcs)
    def test_3d_convolution_nchw(self, input_channels, output_channels,
                                 batch_size, stride, size, kernel, dilation,
                                 pad, use_bias, gc, dc):
        self._nd_convolution_nchw(
            3, input_channels, output_channels, batch_size, stride, size,
            kernel, dilation, pad, use_bias, gc, dc
        )

    @given(op_type=st.sampled_from(["Conv", "Conv3D"]),
           batch_size=st.integers(1, 2),
           stride=st.integers(1, 2),
           size=st.integers(3, 5),
           kernel=st.integers(1, 2),
           dilation=st.integers(1, 2),
           pad=st.integers(0, 2),
           use_bias=st.booleans(),
           **hu.gcs)
    def test_3d_convolution_cudnn_nchw(self, op_type, batch_size, stride, size,
                                       kernel, dilation, pad, use_bias, gc, dc):
        input_channels = 1
        output_channels = 1
        n = 3
        dkernel = dilation * (kernel - 1) + 1
        order = "NCHW"

        op = core.CreateOperator(
            op_type,
            ["X", "w", "b"] if use_bias else ["X", "w"],
            ["Y"],
            strides=[stride] * n,
            kernels=[kernel] * n,
            dilations=[dilation] * n,
            pads=[pad] * n * 2,
            order=order,
            engine="CUDNN",
        )

        input_dims = [batch_size, input_channels]
        input_dims.extend([size] * n)
        filter_dims = [output_channels, input_channels]
        filter_dims.extend([kernel] * n)
        X = np.random.rand(*input_dims).astype(np.float32) - 0.5
        w = np.random.rand(*filter_dims).astype(np.float32) - 0.5
        b = np.random.rand(output_channels).astype(np.float32) - 0.5

        inputs = [X, w, b] if use_bias else [X, w]

        if size + pad + pad < dkernel or size + pad + pad < dkernel:
            with self.assertRaises(RuntimeError):
                self.assertDeviceChecks(dc, op, inputs, [0])
            return

        self.assertDeviceChecks(dc, op, inputs, [0])
        for i in range(len(inputs)):
            self.assertGradientChecks(gc, op, inputs, i, [0])

    @given(op_type=st.sampled_from(["Conv", "Conv2D"]),
           stride=st.integers(1, 3),
           pad=st.integers(0, 3),
           kernel=st.integers(1, 5),
           dilation=st.integers(1, 3),
           size=st.integers(7, 10),
           input_channels=st.integers(1, 8),
           output_channels=st.integers(1, 8),
           batch_size=st.integers(1, 3),
           use_bias=st.booleans(),
           **hu.gcs)
    def test_convolution_layout(self, op_type, stride, pad, kernel, dilation,
                                size, input_channels, output_channels,
                                batch_size, use_bias, gc, dc):
        assume(size >= dilation * (kernel - 1) + 1)

        X = np.random.rand(
            batch_size, size, size, input_channels).astype(np.float32) - 0.5
        w = np.random.rand(
            output_channels, kernel, kernel, input_channels).astype(np.float32)\
            - 0.5
        b = np.random.rand(output_channels).astype(np.float32) - 0.5
        Output = collections.namedtuple("Output", ["Y", "engine", "order"])
        outputs = []

        for order in ["NCHW", "NHWC"]:
            engine_list = ['']
            if _cudnn_supports(dilation=(dilation > 1), nhwc=(order == 'NHWC')):
                engine_list.append('CUDNN')

            for engine in engine_list:
                op = core.CreateOperator(
                    op_type,
                    ["X", "w", "b"] if use_bias else ["X", "w"],
                    ["Y"],
                    stride=stride,
                    kernel=kernel,
                    dilation=dilation,
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
                self.assertDeviceChecks(
                    dc,
                    op,
                    [X_f, w_f, b] if use_bias else [X_f, w_f],
                    [0])
                self.ws.create_blob("X").feed(X_f, device_option=gc)
                self.ws.create_blob("w").feed(w_f, device_option=gc)
                self.ws.create_blob("b").feed(b, device_option=gc)
                self.ws.run(op)
                outputs.append(Output(
                    Y=self.ws.blobs["Y"].fetch(), engine=engine, order=order))

        def canonical(o):
            if o.order == "NHWC":
                return o.Y.transpose((0, 3, 1, 2))
            else:
                return o.Y

        for o in outputs:
            np.testing.assert_allclose(
                canonical(outputs[0]),
                canonical(o),
                atol=1e-4,
                rtol=1e-4)

    @given(num_workers=st.integers(1, 4),
           net_type=st.sampled_from(
               ["simple", "dag"] +
               (["async_dag"] if workspace.has_gpu_support else [])),
           do=st.sampled_from(hu.device_options),
           engine=st.sampled_from(["CUDNN", ""]))
    def test_convolution_sync(self, net_type, num_workers, do, engine):
        m = ModelHelper(name="test_model")
        n = 1
        d = 2
        depth = 3
        iters = 5
        h = 5
        w = 5
        workspace.ResetWorkspace()

        use_cudnn = (engine == 'CUDNN')

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
                brew.conv(
                    m, bottom_1, mid_1,
                    dim_in=d, dim_out=d,
                    kernel=3,
                    weight_init=('ConstantFill', dict(value=w1)),
                    bias_init=('ConstantFill', dict(value=b1)),
                    cudnn_state=np.random.randint(0, 3),
                    stride=1,
                    pad=1,
                    deterministic=1,
                    use_cudnn=use_cudnn,
                    engine=engine)
                brew.conv(
                    m, bottom_2, mid_2,
                    dim_in=d, dim_out=d,
                    kernel=3,
                    stride=1,
                    pad=1,
                    weight_init=('ConstantFill', dict(value=w2)),
                    bias_init=('ConstantFill', dict(value=b2)),
                    deterministic=1,
                    cudnn_state=np.random.randint(0, 3),
                    use_cudnn=use_cudnn,
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
        self.ws.run(m.param_init_net)

        def run():
            import numpy as np
            np.random.seed(1701)
            input_blobs = ["{}_{}".format(depth, j) for j in range(2 ** depth)]
            for input_blob in input_blobs:
                self.ws.create_blob(input_blob).feed(
                    np.random.randn(n, d, h, w).astype(np.float32),
                    device_option=do)
                self.ws.create_blob("label").feed(
                    np.random.randn(n, d * h * w).astype(np.float32),
                    device_option=do)
            self.ws.run(m.net)
            gradients = [
                self.ws.blobs[str(input_to_grad[input_blob])].fetch()
                for input_blob in input_blobs]
            return gradients

        outputs = [run() for _ in range(iters)]
        for output in outputs[1:]:
            np.testing.assert_array_equal(outputs[0], output)
            np.testing.assert_allclose(
                np.sum(np.square(output)),
                1763719461732352.0,
                rtol=1e-5)

    def test_use_cudnn_engine_interactions(self):
        """Make sure the use_cudnn and engine kwargs work as expected."""
        for model_default in [None, True, False]:
            arg_scope = {}
            if model_default is not None:
                arg_scope['use_cudnn'] = model_default
            else:
                model_default = True  # the default

            model = ModelHelper(arg_scope=arg_scope)
            self.assertEqual(model.arg_scope['use_cudnn'], model_default)
            f = functools.partial(brew.conv, model,
                                  'conv_in', 'conv_out', 10, 10, 5)

            for op_cudnn in [None, True, False]:
                for op_engine in [None, '', 'CUDNN']:
                    kwargs = {}
                    if op_cudnn is not None:
                        kwargs['use_cudnn'] = op_cudnn
                    else:
                        op_cudnn = False  # the default
                    if op_engine is not None:
                        kwargs['engine'] = op_engine

                    calculated_cudnn = kwargs.get('use_cudnn', model_default)
                    expected_engine = kwargs.get(
                        'engine',
                        'CUDNN' if calculated_cudnn else '')

                    if ((calculated_cudnn is True and op_engine == '') or
                            (calculated_cudnn is False and op_engine == 'CUDNN')):
                        with self.assertRaises(ValueError):
                            f(**kwargs)
                    else:
                        f(**kwargs)
                        self.assertEqual(model.Proto().op[-1].engine,
                                         expected_engine)


if __name__ == "__main__":
    import unittest
    unittest.main()
