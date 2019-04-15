from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace
from hypothesis import given, assume
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np

import unittest
import os


class TestAdjustBatchOp(hu.HypothesisTestCase):
    @given(d=st.integers(1, 4), n=st.integers(1, 20),
           seed=st.integers(0, 1000), **hu.gcs_cpu_only)
    def test_pad(self, d, n, gc, dc, seed):
        for dtype in [np.float32, np.int8, np.int64]:
            np.random.seed(seed)
            dims = [n] * d
            X = np.random.rand(*dims).astype(dtype)
            max_batch_size = n + 8

            def ref_op(X):
                shape = list(X.shape)
                out = np.zeros((1), dtype=np.int64)
                out[0] = shape[0]
                shape[0] = max_batch_size
                Y = np.zeros(shape, dtype=dtype)
                Y[:n] = X
                return [Y, out]

            op = core.CreateOperator(
                "AdjustBatch",
                ["X"],
                ["Y", "RealBatch"],
                max_batch_size=max_batch_size,
            )

            self.assertReferenceChecks(
                device_option=gc,
                op=op,
                inputs=[X],
                reference=ref_op,
            )

    @given(d=st.integers(1, 4), n=st.integers(8, 20),
           seed=st.integers(0, 1000), **hu.gcs_cpu_only)
    def test_truncate(self, d, n, gc, dc, seed):
        for dtype in [np.float32, np.int8, np.int64]:
            np.random.seed(seed)
            dims = [n] * d
            X = np.random.rand(*dims).astype(dtype)
            real_batch_size = n - 8
            R = np.zeros((1), dtype=np.int64)
            R[0] = real_batch_size

            def ref_op(X, R):
                r = R[0]
                return [X[:r]]

            op = core.CreateOperator(
                "AdjustBatch",
                ["X", "RealBatch"],
                ["Y"],
            )

            self.assertReferenceChecks(
                device_option=gc,
                op=op,
                inputs=[X, R],
                reference=ref_op,
            )
