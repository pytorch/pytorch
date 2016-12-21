from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from hypothesis import assume, given, settings
import hypothesis.strategies as st

from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu

import unittest


class TestGroupConvolution(hu.HypothesisTestCase):

    @given(stride=st.integers(1, 3),
           pad=st.integers(0, 3),
           kernel=st.integers(1, 5),
           size=st.integers(7, 10),
           group=st.integers(1, 4),
           input_channels_per_group=st.integers(1, 8),
           output_channels_per_group=st.integers(1, 8),
           batch_size=st.integers(1, 3),
           # TODO(jiayq): if needed, add NHWC support.
           order=st.sampled_from(["NCHW"]),
           # TODO(jiayq): enable other engines and add reference check.
           engine=st.sampled_from(["", "CUDNN"]),
           **hu.gcs)
    @settings(max_examples=2, timeout=100)
    def test_group_convolution(
            self, stride, pad, kernel, size, group,
            input_channels_per_group, output_channels_per_group, batch_size,
            order, engine, gc, dc):
        assume(size >= kernel)
        input_channels = input_channels_per_group * group
        output_channels = output_channels_per_group * group

        op = core.CreateOperator(
            "Conv",
            ["X", "w", "b"],
            ["Y"],
            stride=stride,
            kernel=kernel,
            pad=pad,
            order=order,
            engine=engine,
            group=group,
        )
        X = np.random.rand(
            batch_size, size, size, input_channels).astype(np.float32) - 0.5
        w = np.random.rand(
            output_channels, kernel, kernel,
            input_channels_per_group).astype(np.float32)\
            - 0.5
        b = np.random.rand(output_channels).astype(np.float32) - 0.5
        if order == "NCHW":
            X = X.transpose((0, 3, 1, 2))
            w = w.transpose((0, 3, 1, 2))

        self.assertDeviceChecks(dc, op, [X, w, b], [0])
        for i in range(3):
            self.assertGradientChecks(gc, op, [X, w, b], i, [0])


if __name__ == "__main__":
    unittest.main()
