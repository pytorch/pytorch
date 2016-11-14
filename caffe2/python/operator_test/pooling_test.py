from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from hypothesis import assume, given, settings
import hypothesis.strategies as st
import collections

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
