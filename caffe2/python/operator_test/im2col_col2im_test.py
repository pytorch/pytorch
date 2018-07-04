from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import assume, given

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


class TestReduceFrontSum(hu.HypothesisTestCase):
    @given(batch_size=st.integers(1, 3),
           stride=st.integers(1, 3),
           pad=st.integers(0, 3),
           kernel=st.integers(1, 5),
           dilation=st.integers(1, 3),
           size=st.integers(7, 10),
           channels=st.integers(1, 8),
           **hu.gcs)
    def test_im2col_layout(self, batch_size, stride, pad, kernel, dilation,
                           size, channels, gc, dc):

        dkernel = (dilation * (kernel - 1) + 1)
        assume(size >= dkernel)

        NCHW_TO_NHWC = (0, 2, 3, 1)
        NHWC_TO_NCHW = (0, 3, 1, 2)
        COL_NHWC_TO_NCHW = (4, 2, 3, 0, 1)

        N = batch_size
        C = channels
        H = size
        W = size

        out_h = int((H + (2 * pad) - dkernel) / stride + 1)
        out_w = int((W + (2 * pad) - dkernel) / stride + 1)

        im_nchw = np.random.rand(N, C, H, W).astype(np.float32) - 0.5
        im_nhwc = im_nchw.transpose(NCHW_TO_NHWC)

        op_im2col_nchw = core.CreateOperator(
            "Im2Col",
            ["im_nchw"], ["col_nchw"],
            stride=stride,
            kernel=kernel,
            dilation=dilation,
            pad=pad,
            order="NCHW",
            device_option=gc)

        op_im2col_nhwc = core.CreateOperator(
            "Im2Col",
            ["im_nhwc"], ["col_nhwc"],
            stride=stride,
            kernel=kernel,
            dilation=dilation,
            pad=pad,
            order="NHWC",
            device_option=gc)

        self.ws.create_blob("im_nchw").feed(im_nchw, device_option=gc)
        self.ws.create_blob("im_nhwc").feed(im_nhwc, device_option=gc)
        self.ws.run(op_im2col_nchw)
        self.ws.run(op_im2col_nhwc)

        # there is probably a clever way to spell this in np
        col_nchw = self.ws.blobs["col_nchw"].fetch()
        col_nhwc = self.ws.blobs["col_nhwc"].fetch()
        col_nchw_ = col_nchw.reshape(N, C, kernel, kernel, out_h, out_w)
        col_nhwc_ = col_nhwc.reshape(N, out_h, out_w, kernel, kernel, C)
        for i in range(0, N):
            np.testing.assert_allclose(
                col_nchw_[i],
                col_nhwc_[i].transpose(COL_NHWC_TO_NCHW),
                atol=1e-4,
                rtol=1e-4)

        op_col2im_nchw = core.CreateOperator(
            "Col2Im",
            ["col_nchw", "im_nchw"],
            ["out_nchw"],
            stride=stride,
            kernel=kernel,
            dilation=dilation,
            pad=pad,
            order="NCHW",
            device_option=gc)

        op_col2im_nhwc = core.CreateOperator(
            "Col2Im",
            ["col_nhwc", "im_nhwc"],
            ["out_nhwc"],
            stride=stride,
            kernel=kernel,
            dilation=dilation,
            pad=pad,
            order="NHWC",
            device_option=gc)

        self.ws.run(op_col2im_nchw)
        self.ws.run(op_col2im_nhwc)

        out_nchw = self.ws.blobs["out_nchw"].fetch()
        out_nhwc = self.ws.blobs["out_nhwc"].fetch()
        np.testing.assert_allclose(
            out_nchw,
            out_nhwc.transpose(NHWC_TO_NCHW),
            atol=1e-4,
            rtol=1e-4)

    @given(batch_size=st.integers(1, 3),
           stride=st.integers(1, 3),
           pad=st.integers(0, 3),
           kernel=st.integers(1, 5),
           dilation=st.integers(1, 3),
           size=st.integers(7, 10),
           channels=st.integers(1, 8),
           order=st.sampled_from(["NCHW"]),
           **hu.gcs)
    def test_col2im_gradients(self, batch_size, stride, pad, kernel,
                              dilation, size, channels, order, gc, dc):
        assume(size >= dilation * (kernel - 1) + 1)
        op = core.CreateOperator(
            "Im2Col",
            ["X"], ["Y"],
            stride=stride,
            kernel=kernel,
            dilation=dilation,
            pad=pad,
            order=order,
            device_option=gc)
        X = np.random.rand(batch_size, channels, size, size).astype(np.float32)
        self.assertGradientChecks(gc, op, [X], 0, [0])
        return
