

from caffe2.python import core
from hypothesis import given, settings

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np

import unittest


class TestRMSNormOp(hu.HypothesisTestCase):
    @given(
        M=st.integers(0, 8),
        N=st.integers(1, 16),
        eps=st.floats(0, 1e-3),
        dtype=st.sampled_from([np.float32, np.float64]),
        **hu.gcs,
    )
    @settings(deadline=None)
    def test_rms_norm(self, M, N, eps, dtype, gc, dc):
        X = (np.random.randn(M, N) * 2.0 + 1.0).astype(dtype)
        gamma = np.random.randn(N).astype(dtype)
        beta = np.random.randn(N).astype(dtype)

        op = core.CreateOperator(
            "RMSNorm",
            ["X", "gamma", "beta"],
            ["Y", "rrms"],
            eps=eps,
        )

        def rms_norm_ref(X, gamma, beta):
            rrms = 1.0 / np.sqrt(np.mean(np.square(X), axis=1) + eps)
            Y = X * np.expand_dims(rrms, axis=1) * gamma + beta
            return Y, rrms

        inputs = [X, gamma, beta]
        self.assertReferenceChecks(gc, op, inputs, rms_norm_ref)
        self.assertDeviceChecks(dc, op, inputs, [0, 1])
        for i in range(len(inputs)):
            self.assertGradientChecks(gc, op, inputs, i, [0])


if __name__ == "__main__":
    unittest.main()
