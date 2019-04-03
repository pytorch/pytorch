from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import caffe2.python.hypothesis_test_util as hu
from caffe2.python import core, dyndep, utils, workspace
from hypothesis import given
import hypothesis.strategies as st
import unittest


def _test_convolution_gradients(
    self, pad, kernel, size, channels, batch_size, order, engine, use_bias, gc, dc
):
    op = core.CreateOperator(
        "Conv",
        ["X", "w", "b"] if use_bias else ["X", "w"],
        ["Y"],
        kernel=kernel,
        pad=pad,
        group=channels,
        order=order,
        engine=engine,
    )
    X = np.random.rand(
        batch_size, size, size, channels).astype(np.float32) - 0.5
    w = np.random.rand(
        channels, kernel, kernel, 1).astype(np.float32)\
        - 0.5
    b = np.random.rand(channels).astype(np.float32) - 0.5
    if order == "NCHW":
        X = utils.NHWC2NCHW(X)
        w = utils.NHWC2NCHW(w)

    inputs = [X, w, b] if use_bias else [X, w]
    # Error handling path.
    if size + pad + pad < kernel or size + pad + pad < kernel:
        with self.assertRaises(RuntimeError):
            self.assertDeviceChecks(dc, op, inputs, [0])
        return

    self.assertDeviceChecks(dc, op, inputs, [0])
    for i in range(len(inputs)):
        self.assertGradientChecks(gc, op, inputs, i, [0])


def _test_convolution_engine(
    self, pad, kernel, size, channels, batch_size, order, engine, use_bias, gc, dc
):
    """ Compare CUDNN result and result with engine """
    op = core.CreateOperator(
        "Conv",
        ["X", "w", "b"] if use_bias else ["X", "w"],
        ["Y"],
        kernel=kernel,
        pad=pad,
        group=channels,
        order=order,
        engine=engine,
    )
    op_noengine = core.CreateOperator(
        "Conv",
        ["X", "w", "b"] if use_bias else ["X", "w"],
        ["Y_GT"],
        kernel=kernel,
        pad=pad,
        group=channels,
        order=order,
        engine="CUDNN",
    )
    X = np.random.rand(batch_size, size, size, channels).astype(np.float32) - 0.5
    w = np.random.rand(channels, kernel, kernel, 1).astype(np.float32) - 0.5
    b = np.random.rand(channels).astype(np.float32) - 0.5
    if order == "NCHW":
        X = utils.NHWC2NCHW(X)
        w = utils.NHWC2NCHW(w)

    inputs = [X, w, b] if use_bias else [X, w]

    def get_device_option_cuda(gpu_id=0):
        from caffe2.proto import caffe2_pb2

        device_option = caffe2_pb2.DeviceOption()
        device_option.device_type = caffe2_pb2.CUDA
        device_option.device_id = gpu_id
        return device_option

    def _run(op, inputs, out_name):
        import copy

        do = get_device_option_cuda()

        with core.DeviceScope(do):
            for idx, x in enumerate(op.input):
                workspace.FeedBlob(x, inputs[idx])
            op = copy.deepcopy(op)
            op.device_option.CopyFrom(do)
            workspace.RunOperatorOnce(op)
            y = workspace.FetchBlob(out_name)
        return [y]

    # Error handling path.
    if size + pad + pad < kernel or size + pad + pad < kernel:
        with self.assertRaises(RuntimeError):
            self.assertDeviceChecks(dc, op, inputs, [0])
        return

    Y = _run(op, inputs, "Y")
    Y_GT = _run(op_noengine, inputs, "Y_GT")
    np.testing.assert_allclose(Y, Y_GT, rtol=1e-4)


class DepthwiseConvEngineOpsTest(hu.HypothesisTestCase):
    ENGINE_MAPPING = {
        3: "DEPTHWISE_3x3",
        5: "DEPTHWISE_5x5",
        7: "DEPTHWISE_7x7",
    }

    @unittest.skipIf(not workspace.has_gpu_support, "No gpu support.")
    @given(
        pad=st.integers(0, 1),
        kernel=st.sampled_from([3, 5, 7]),
        size=st.integers(4, 8),
        channels=st.integers(2, 4),
        batch_size=st.integers(1, 1),
        order=st.sampled_from(["NCHW"]),
        use_bias=st.booleans(),
        **hu.gcs
    )
    def test_convolution_gradients(
        self, pad, kernel, size, channels, batch_size, order, use_bias, gc, dc
    ):
        _test_convolution_gradients(
            self,
            pad,
            kernel,
            size,
            channels,
            batch_size,
            order,
            self.ENGINE_MAPPING[kernel],
            use_bias,
            gc,
            dc,
        )

    @unittest.skipIf(not workspace.has_gpu_support, "No gpu support.")
    @given(
        pad=st.integers(0, 4),
        kernel=st.sampled_from([3, 5, 7]),
        size=st.integers(4, 8),
        channels=st.integers(2, 4),
        batch_size=st.integers(1, 1),
        order=st.sampled_from(["NCHW"]),
        use_bias=st.booleans(),
        **hu.gcs_gpu_only
    )
    def test_convolution_engine(
        self, pad, kernel, size, channels, batch_size, order, use_bias, gc, dc
    ):
        """ Compare CUDNN result and result with engine """
        _test_convolution_engine(
            self,
            pad,
            kernel,
            size,
            channels,
            batch_size,
            order,
            self.ENGINE_MAPPING[kernel],
            use_bias,
            gc,
            dc,
        )
