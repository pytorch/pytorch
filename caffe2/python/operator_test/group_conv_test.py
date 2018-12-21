from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from hypothesis import assume, given, settings
import hypothesis.strategies as st

from caffe2.proto import caffe2_pb2
from caffe2.python import core
import caffe2.python.hip_test_util as hiputl
import caffe2.python.hypothesis_test_util as hu

import unittest
import os

class TestGroupConvolution(hu.HypothesisTestCase):

    @given(stride=st.integers(1, 3),
           pad=st.integers(0, 3),
           kernel=st.integers(1, 5),
           size=st.integers(7, 10),
           group=st.integers(1, 4),
           input_channels_per_group=st.integers(1, 8),
           output_channels_per_group=st.integers(1, 8),
           batch_size=st.integers(1, 3),
           order=st.sampled_from(["NCHW", "NHWC"]),
           # Note: Eigen does not support group convolution, but it should
           # fall back to the default engine without failing.
           engine=st.sampled_from(["", "CUDNN", "EIGEN"]),
           use_bias=st.booleans(),
           **hu.gcs)
    @settings(max_examples=2, timeout=100)
    def test_group_convolution(
            self, stride, pad, kernel, size, group,
            input_channels_per_group, output_channels_per_group, batch_size,
            order, engine, use_bias, gc, dc):
        assume(size >= kernel)

        if hiputl.run_in_hip(gc, dc):
            if order == "NHWC":
                assume(group == 1 and engine != "CUDNN")
        else:
            # TODO: Group conv in NHWC not implemented for GPU yet.
            assume(group == 1 or order == "NCHW" or gc.device_type != caffe2_pb2.CUDA)
        input_channels = input_channels_per_group * group
        output_channels = output_channels_per_group * group

        op = core.CreateOperator(
            "Conv",
            ["X", "w", "b"] if use_bias else ["X", "w"],
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

        inputs = [X, w, b] if use_bias else [X, w]

        self.assertDeviceChecks(dc, op, inputs, [0])
        for i in range(len(inputs)):
            self.assertGradientChecks(gc, op, inputs, i, [0])


if __name__ == "__main__":
    unittest.main()
