from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from hypothesis import given
import hypothesis.strategies as st
import numpy as np

from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial


class TestCosineEmbeddingCriterion(serial.SerializedTestCase):
    @serial.given(N=st.integers(min_value=10, max_value=20),
           seed=st.integers(min_value=0, max_value=65535),
           margin=st.floats(min_value=-0.5, max_value=0.5),
           **hu.gcs)
    def test_cosine_embedding_criterion(self, N, seed, margin, gc, dc):
        np.random.seed(seed)
        S = np.random.randn(N).astype(np.float32)
        Y = np.random.choice([-1, 1], size=N).astype(np.int32)
        op = core.CreateOperator(
            "CosineEmbeddingCriterion", ["S", "Y"], ["output"],
            margin=margin)

        def ref_cec(S, Y):
            result = (1 - S) * (Y == 1) + np.maximum(S - margin, 0) * (Y == -1)
            return (result, )

        # This checks the op implementation against a reference function in
        # python.
        self.assertReferenceChecks(gc, op, [S, Y], ref_cec)
        # This checks the op implementation over multiple device options (e.g.
        # CPU and CUDA). [0] means that the 0-th output is checked.
        self.assertDeviceChecks(dc, op, [S, Y], [0])

        # Now, since this operator's output has a "kink" around the margin
        # value, we move the S vector away from the margin a little bit. This
        # is a standard trick to avoid gradient check to fail on subgradient
        # points.
        S[np.abs(S - margin) < 0.1] += 0.2
        # This checks the operator's gradient. the first 0 means that we are
        # checking the gradient of the first input (S), and the second [0] means
        # that the gradient check should initiate from the 0-th output.
        self.assertGradientChecks(gc, op, [S, Y], 0, [0])

if __name__ == "__main__":
    import unittest
    unittest.main()
