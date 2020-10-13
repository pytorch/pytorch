




from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np


class TestLengthsTopKOps(serial.SerializedTestCase):
    @serial.given(N=st.integers(min_value=0, max_value=10),
           K=st.integers(min_value=1, max_value=10),
           **hu.gcs_cpu_only)
    def test_lengths_top_k_op(self, N, K, gc, dc):
        lens = np.random.randint(low=1, high=2 * K + 1, size=N).astype(np.int32)
        X = []
        for i in lens:
            X.extend(map(lambda x: x / 100.0, range(0, 6 * i, 6)))
        X = np.array(X, dtype=np.float32)
        op = core.CreateOperator("LengthsTopK", ["X", "Y"], ["values", "indices"], k=K)

        def lengths_top_k(X, lens):
            N, si = lens.shape[0], 0
            values, indices = [], []
            for i in range(N):
                cur_indices = X[si:si + lens[i]].argsort()[-K:][::-1]
                cur_values = X[si:si + lens[i]][cur_indices]
                values.extend(cur_values)
                indices.extend(cur_indices)
                si += lens[i]
                if lens[i] < K:
                    values.extend([0] * (K - lens[i]))
                    indices.extend([-1] * (K - lens[i]))

            return (np.array(values, dtype=np.float32).reshape(-1, K),
                    np.array(indices, dtype=np.int32).reshape(-1, K))

        self.assertDeviceChecks(dc, op, [X, lens], [0, 1])
        self.assertReferenceChecks(gc, op, [X, lens], lengths_top_k)
        self.assertGradientChecks(gc, op, [X, lens], 0, [0])

    @given(N=st.integers(min_value=0, max_value=10),
           K=st.integers(min_value=1, max_value=10),
           **hu.gcs_cpu_only)
    def test_lengths_top_k_empty_op(self, N, K, gc, dc):
        lens = np.zeros((N, ), dtype=np.int32)
        X = np.array([], dtype=np.float32)
        op = core.CreateOperator("LengthsTopK", ["X", "Y"], ["values", "indices"], k=K)

        def lengths_top_k(X, lens):
            return (np.zeros((N, K), dtype=np.float32),
                    -1 * np.ones((N, K), dtype=np.int32))

        self.assertDeviceChecks(dc, op, [X, lens], [0, 1])
        self.assertReferenceChecks(gc, op, [X, lens], lengths_top_k)
        self.assertGradientChecks(gc, op, [X, lens], 0, [0])
