from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from scipy.sparse import coo_matrix

from hypothesis import given
import hypothesis.strategies as st

from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu


class TestSparseGradient(hu.HypothesisTestCase):
    @given(M=st.integers(min_value=5, max_value=20),
           N=st.integers(min_value=5, max_value=20),
           K=st.integers(min_value=5, max_value=15),
           sparsity=st.floats(min_value=0.1, max_value=1.0),
           **hu.gcs_cpu_only)
    def test_sparse_gradient(self, M, N, K, sparsity, gc, dc):
        X = np.random.randn(M, K).astype(np.float32)
        X[X > sparsity] = 0
        X_coo = coo_matrix(X)
        val, key, seg = X_coo.data, X_coo.col, X_coo.row

        val = val.astype(np.float32)
        key = key.astype(np.int64)
        seg = seg.astype(np.int32)

        Y = np.random.randn(K, N).astype(np.float32)

        op = core.CreateOperator(
            'SparseUnsortedSegmentWeightedSum',
            ['Y', 'val', 'key', 'seg'],
            ['out'],
            num_segments=M)

        # Gradient check wrt Y
        self.assertGradientChecks(
            gc, op, [Y, val, key, seg], 0, [0])

if __name__ == "__main__":
    import unittest
    unittest.main()
