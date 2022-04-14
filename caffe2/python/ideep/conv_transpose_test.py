



import unittest
import numpy as np
from hypothesis import assume, given, settings
import hypothesis.strategies as st

from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.ideep_test_util as mu


@unittest.skipIf(not workspace.C.use_mkldnn, "No MKLDNN support.")
class ConvTransposeTest(hu.HypothesisTestCase):
    @given(stride=st.integers(1, 2),
           pad=st.integers(0, 3),
           kernel=st.integers(1, 5),
           adj=st.integers(0, 2),
           size=st.integers(7, 10),
           input_channels=st.integers(1, 8),
           output_channels=st.integers(1, 8),
           batch_size=st.integers(1, 3),
           use_bias=st.booleans(),
           training_mode=st.booleans(),
           compute_dX=st.booleans(),
           **mu.gcs)
    @settings(max_examples=2, timeout=100)
    def test_convolution_transpose_gradients(self, stride, pad, kernel, adj,
                                             size, input_channels,
                                             output_channels, batch_size,
                                             use_bias, training_mode,
                                             compute_dX, gc, dc):
        training = 1 if training_mode else 0
        assume(adj < stride)
        X = np.random.rand(
            batch_size, input_channels, size, size).astype(np.float32) - 0.5
        w = np.random.rand(
            input_channels, output_channels, kernel, kernel)\
            .astype(np.float32) - 0.5
        b = np.random.rand(output_channels).astype(np.float32) - 0.5
        op = core.CreateOperator(
            "ConvTranspose",
            ["X", "w", "b"] if use_bias else ["X", "w"],
            ["Y"],
            stride=stride,
            kernel=kernel,
            pad=pad,
            adj=adj,
            training_mode=training,
            no_gradient_to_input=not compute_dX,
        )

        inputs = [X, w, b] if use_bias else [X, w]
        self.assertDeviceChecks(dc, op, inputs, [0], threshold=0.001)

        if training_mode:
            if use_bias and compute_dX:
                # w, b, X
                outputs_to_check = [1, 2, 0]
            elif use_bias:
                # w, b
                outputs_to_check = [1, 2]
            elif compute_dX:
                # w, X
                outputs_to_check = [1, 0]
            else:
                # w
                outputs_to_check = [1]
            for i in outputs_to_check:
                self.assertGradientChecks(gc, op, inputs, i, [0])


if __name__ == "__main__":
    unittest.main()
