from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


class TestLengthsMatMulOps(hu.HypothesisTestCase):
    @given(N=st.integers(min_value=1, max_value=10),
           **hu.gcs_cpu_only)
    def test_lengths_matmul_op(self, N, gc, dc):
        X_LEN = np.random.randint(low=1, high=5, size=10).astype(np.int32)
        Y_LEN = np.random.randint(low=1, high=5, size=10).astype(np.int32)
        X = []
        Y = []
        for i in X_LEN:
            X.extend(np.random.rand(i, N))
        for i in Y_LEN:
            Y.extend(np.random.rand(i, N))
        X = np.array(X, dtype=np.float32).reshape(-1, N)
        Y = np.array(Y, dtype=np.float32).reshape(-1, N)
        op = core.CreateOperator("LengthsMatMul",
                                 ["X", "X_LEN", "Y", "Y_LEN"],
                                 ["values", "lengths"])

        def lengths_matmul(X, X_lens, Y, Y_lens):
            N = X_lens.shape[0]
            values, lengths = [], []
            x_dist = 0
            y_dist = 0
            for i in range(N):
                cur_lengths = X_lens[i] * Y_lens[i]
                cur_values = np.matmul(X[x_dist:x_dist + X_lens[i], :],
                                       Y[y_dist:y_dist + Y_lens[i], :].T)
                values.extend(cur_values.flatten())
                lengths.append(cur_lengths)
                x_dist += X_lens[i]
                y_dist += Y_lens[i]
            return (np.array(values, dtype=np.float32).flatten(),
                    np.array(lengths, dtype=np.int32))

        self.assertDeviceChecks(dc, op, [X, X_LEN, Y, Y_LEN], [0, 1])
        self.assertReferenceChecks(gc, op, [X, X_LEN, Y, Y_LEN], lengths_matmul)
        self.assertGradientChecks(gc, op, [X, X_LEN, Y, Y_LEN], 0, [0])
