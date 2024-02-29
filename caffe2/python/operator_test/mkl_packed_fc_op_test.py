




import unittest
import hypothesis.strategies as st
from hypothesis import given
import numpy as np
from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu


@unittest.skipIf(not core.IsOperator("PackedFC"),
                 "PackedFC is not supported in this caffe2 build.")
class PackedFCTest(hu.HypothesisTestCase):
    @given(seed=st.integers(0, 65536),
           M=st.integers(16, 32),
           K=st.integers(128, 1024),
           N=st.integers(128, 1024),
           **hu.gcs_cpu_only)
    @unittest.skipIf(not core.C.builtin_cpu_supports_avx2(),
                     "Intel MKL sgemm_pack has a known numerical issue with "
                     "non-avx2 machines that will be fixed in a later build.")
    def test_packed_fc(self, seed, M, K, N, gc, dc):
        np.random.seed(seed)
        X = np.random.rand(M, K).astype(np.float32) - 0.5
        W = np.random.rand(N, K).astype(np.float32) - 0.5
        b = np.random.rand(N).astype(np.float32) - 0.5

        # If you are debugging, the following hard-coded ones might help.
        # X = np.ones((24, 256)).astype(np.float32)
        # W = np.ones((128, 256)).astype(np.float32)
        # b = np.zeros(128).astype(np.float32)

        def ref(X, W, b):
            return (np.dot(X, W.T) + b,)

        for name in ["FC", "PackedFC"]:
            op = core.CreateOperator(
                name,
                ["X", "W", "b"],
                ["Y"],
            )
            self.assertReferenceChecks(gc, op, [X, W, b], ref)

    @unittest.skipIf(not core.C.builtin_cpu_supports_avx2(),
                     "Intel MKL sgemm_pack has a known numerical issue with "
                     "non-avx2 machines that will be fixed in a later build.")
    @given(axis=st.integers(min_value=1, max_value=4),
           num_output=st.integers(min_value=4, max_value=8),
           **hu.gcs_cpu_only)
    def test_packed_fc_axis(self, axis, num_output, gc, dc):
        np.random.seed(1701)
        X = np.random.randn(1, 2, 3, 2, 1).astype(np.float32)
        K = np.prod(X.shape[axis:])
        N = num_output
        W = np.random.randn(N, K).astype(np.float32)
        b = np.random.randn(N).astype(np.float32)

        op = core.CreateOperator(
            "PackedFC",
            ["X", "W", "b"],
            ["Y"],
            axis=axis)

        def ref(X, W, b):
            output_axes = list(X.shape[:axis]) + [N]
            return (
                np.dot(X.reshape(int(X.size / K), K), W.T).reshape(output_axes) + b,)

        self.assertReferenceChecks(gc, op, [X, W, b], ref)

if __name__ == "__main__":
    import unittest
    unittest.main()
