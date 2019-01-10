from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import caffe2.python.hypothesis_test_util as hu
from caffe2.python import core, dyndep, workspace
from hypothesis import given
import hypothesis.strategies as st


class Depthwise3x3ConvOpsTest(hu.HypothesisTestCase):
    @given(pad=st.integers(0, 1),
           kernel=st.integers(3, 3),
           size=st.integers(4, 8),
           channels=st.integers(2, 4),
           batch_size=st.integers(1, 1),
           order=st.sampled_from(["NCHW"]),
           engine=st.sampled_from(["DEPTHWISE_3x3"]),
           use_bias=st.booleans(),
           **hu.gcs)
    def test_convolution_gradients(self, pad, kernel, size,
                                   channels, batch_size,
                                   order, engine, use_bias, gc, dc):
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
            X = X.transpose((0, 3, 1, 2))
            w = w.transpose((0, 3, 1, 2))

        inputs = [X, w, b] if use_bias else [X, w]
        # Error handling path.
        if size + pad + pad < kernel or size + pad + pad < kernel:
            with self.assertRaises(RuntimeError):
                self.assertDeviceChecks(dc, op, inputs, [0])
            return

        self.assertDeviceChecks(dc, op, inputs, [0])
        for i in range(len(inputs)):
            self.assertGradientChecks(gc, op, inputs, i, [0])
