from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from hypothesis import assume, given, settings
import hypothesis.strategies as st
import os
import unittest

from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu


class TestPooling(hu.HypothesisTestCase):
    # CUDNN does NOT support different padding values and we skip it
    @given(stride_h=st.integers(1, 3),
           stride_w=st.integers(1, 3),
           pad_t=st.integers(0, 3),
           pad_l=st.integers(0, 3),
           pad_b=st.integers(0, 3),
           pad_r=st.integers(0, 3),
           kernel=st.integers(3, 5),
           size=st.integers(7, 9),
           input_channels=st.integers(1, 3),
           batch_size=st.integers(1, 3),
           order=st.sampled_from(["NCHW", "NHWC"]),
           method=st.sampled_from(["MaxPool", "AveragePool", "LpPool"]),
           **hu.gcs)
    def test_pooling_separate_stride_pad(self, stride_h, stride_w,
                                         pad_t, pad_l, pad_b,
                                         pad_r, kernel, size,
                                         input_channels,
                                         batch_size, order,
                                         method,
                                         gc, dc):
        assume(np.max([pad_t, pad_l, pad_b, pad_r]) < kernel)

        op = core.CreateOperator(
            method,
            ["X"],
            ["Y"],
            stride_h=stride_h,
            stride_w=stride_w,
            pad_t=pad_t,
            pad_l=pad_l,
            pad_b=pad_b,
            pad_r=pad_r,
            kernel=kernel,
            order=order,
        )
        X = np.random.rand(
            batch_size, size, size, input_channels).astype(np.float32)

        if order == "NCHW":
            X = X.transpose((0, 3, 1, 2))
        self.assertDeviceChecks(dc, op, [X], [0])
        if method not in ('MaxPool'):
            self.assertGradientChecks(gc, op, [X], 0, [0])

    # This test is to check if CUDNN works for bigger batch size or not
    @unittest.skipIf(not os.getenv('CAFFE2_DEBUG'),
                     "This is a test that reproduces a cudnn error. If you "
                     "want to run it, set env variable CAFFE2_DEBUG=1.")
    @given(**hu.gcs_gpu_only)
    def test_pooling_big_batch(self, gc, dc):
        op = core.CreateOperator(
            "AveragePool",
            ["X"],
            ["Y"],
            stride=1,
            kernel=7,
            pad=0,
            order="NHWC",
            engine="CUDNN",
        )
        X = np.random.rand(70000, 7, 7, 81).astype(np.float32)

        self.assertDeviceChecks(dc, op, [X], [0])

    @given(stride=st.integers(1, 3),
           pad=st.integers(0, 3),
           kernel=st.integers(1, 5),
           size=st.integers(7, 9),
           input_channels=st.integers(1, 3),
           batch_size=st.integers(1, 3),
           order=st.sampled_from(["NCHW", "NHWC"]),
           method=st.sampled_from(["MaxPool", "AveragePool"]),
           **hu.gcs)
    def test_pooling_1d(self, stride, pad, kernel, size, input_channels,
                        batch_size, order, method, gc, dc):
        assume(pad < kernel)
        op = core.CreateOperator(
            method,
            ["X"],
            ["Y"],
            strides=[stride],
            kernels=[kernel],
            pads=[pad, pad],
            order=order,
            engine="",
        )
        X = np.random.rand(
            batch_size, size, input_channels).astype(np.float32)
        if order == "NCHW":
            X = X.transpose((0, 2, 1))

        self.assertDeviceChecks(dc, op, [X], [0])
        if method not in ('MaxPool'):
            self.assertGradientChecks(gc, op, [X], 0, [0])

    @given(stride=st.integers(1, 3),
           pad=st.integers(0, 3),
           kernel=st.integers(1, 5),
           size=st.integers(7, 9),
           input_channels=st.integers(1, 3),
           batch_size=st.integers(1, 3),
           order=st.sampled_from(["NCHW", "NHWC"]),
           method=st.sampled_from(["MaxPool", "AveragePool"]),
           engine=st.sampled_from(["", "CUDNN"]),
           **hu.gcs)
    def test_pooling_3d(self, stride, pad, kernel, size, input_channels,
                        batch_size, order, method, engine, gc, dc):
        assume(pad < kernel)
        op = core.CreateOperator(
            method,
            ["X"],
            ["Y"],
            strides=[stride] * 3,
            kernels=[kernel] * 3,
            pads=[pad] * 6,
            order=order,
            engine=engine,
        )
        X = np.random.rand(
            batch_size, size, size, size, input_channels).astype(np.float32)
        if order == "NCHW":
            X = X.transpose((0, 4, 1, 2, 3))

        self.assertDeviceChecks(dc, op, [X], [0])
        if method not in ('MaxPool'):
            self.assertGradientChecks(gc, op, [X], 0, [0])

    @unittest.skipIf(not workspace.has_gpu_support, "No GPU support")
    @given(stride=st.integers(1, 3),
           pad=st.integers(0, 3),
           kernel=st.integers(1, 5),
           size=st.integers(7, 9),
           input_channels=st.integers(1, 3),
           batch_size=st.integers(1, 3),
           **hu.gcs_gpu_only)
    def test_pooling_with_index(self, stride, pad, kernel, size,
                                input_channels, batch_size, gc, dc):
        assume(pad < kernel)
        op = core.CreateOperator(
            "MaxPoolWithIndex",
            ["X"],
            ["Y", "Y_index"],
            stride=stride,
            kernel=kernel,
            pad=pad,
            order="NCHW",
        )
        X = np.random.rand(
            batch_size, size, size, input_channels).astype(np.float32)

        # transpose due to order = NCHW
        X = X.transpose((0, 3, 1, 2))

        self.assertDeviceChecks(dc, op, [X], [0])

    @given(sz=st.integers(1, 20),
           batch_size=st.integers(1, 4),
           engine=st.sampled_from(["", "CUDNN"]),
           **hu.gcs)
    @settings(max_examples=3, timeout=10)
    def test_global_avg_pool_nchw(self, sz, batch_size, engine, gc, dc):
        ''' Special test to stress the fast path of NCHW average pool '''
        op = core.CreateOperator(
            "AveragePool",
            ["X"],
            ["Y"],
            stride=1,
            kernel=sz,
            pad=0,
            order="NCHW",
            engine=engine,
        )
        X = np.random.rand(
            batch_size, 3, sz, sz).astype(np.float32)

        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0])

    @given(stride=st.integers(1, 3),
           pad=st.integers(0, 3),
           kernel=st.integers(1, 5),
           size=st.integers(7, 9),
           input_channels=st.integers(1, 3),
           batch_size=st.integers(1, 3),
           order=st.sampled_from(["NCHW", "NHWC"]),
           method=st.sampled_from(["MaxPool", "AveragePool", "LpPool"]),
           engine=st.sampled_from(["", "CUDNN"]),
           **hu.gcs)
    def test_pooling(self, stride, pad, kernel, size,
                     input_channels, batch_size,
                     order, method, engine, gc, dc):
        assume(pad < kernel)
        op = core.CreateOperator(
            method,
            ["X"],
            ["Y"],
            stride=stride,
            kernel=kernel,
            pad=pad,
            order=order,
            engine=engine,
        )
        X = np.random.rand(
            batch_size, size, size, input_channels).astype(np.float32)
        if order == "NCHW":
            X = X.transpose((0, 3, 1, 2))

        self.assertDeviceChecks(dc, op, [X], [0])
        if method not in ('MaxPool'):
            self.assertGradientChecks(gc, op, [X], 0, [0])

    @given(size=st.integers(7, 9),
           input_channels=st.integers(1, 3),
           batch_size=st.integers(1, 3),
           order=st.sampled_from(["NCHW", "NHWC"]),
           method=st.sampled_from(["MaxPool", "AveragePool", "LpPool"]),
           engine=st.sampled_from(["", "CUDNN"]),
           **hu.gcs)
    def test_global_pooling(self, size, input_channels, batch_size,
                            order, method, engine, gc, dc):
        op = core.CreateOperator(
            method,
            ["X"],
            ["Y"],
            order=order,
            engine=engine,
            global_pooling=True,
        )
        X = np.random.rand(
            batch_size, size, size, input_channels).astype(np.float32)
        if order == "NCHW":
            X = X.transpose((0, 3, 1, 2))

        self.assertDeviceChecks(dc, op, [X], [0])
        if method not in ('MaxPool'):
            self.assertGradientChecks(gc, op, [X], 0, [0])


if __name__ == "__main__":
    import unittest
    unittest.main()
