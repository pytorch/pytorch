




from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
from hypothesis import given, settings
import hypothesis.strategies as st
import numpy as np


class TestBatchSparseToDense(serial.SerializedTestCase):

    @given(
        batch_size=st.integers(5, 10),
        dense_last_dim=st.integers(5, 10),
        default_value=st.floats(min_value=2.0, max_value=3.0),
        **hu.gcs
    )
    @settings(deadline=None)
    def test_batch_sparse_to_dense(
        self, batch_size, dense_last_dim, default_value, gc, dc
    ):
        L = np.random.randint(1, dense_last_dim + 1, size=(batch_size))
        num_data = L.sum()
        # The following logic ensure that indices in each batch will not be duplicated
        I = np.array([]).astype(np.int32)
        for l in L:
            I_l = np.random.choice(dense_last_dim, l, replace=False)
            I = np.concatenate((I, I_l))
        V = np.random.rand(num_data).astype(np.float32)

        op = core.CreateOperator(
            'BatchSparseToDense',
            ['L', 'I', 'V'],
            ['O'],
            dense_last_dim=dense_last_dim,
            default_value=default_value,
        )

        S = np.random.rand(batch_size, dense_last_dim).astype(np.float32)
        op2 = core.CreateOperator(
            'BatchSparseToDense',
            ['L', 'I', 'V', 'S'],
            ['O'],
            default_value=default_value,
        )

        def batch_sparse_to_dense_ref(L, I, V, S=None):
            if S is None:
                ret = np.zeros((batch_size, dense_last_dim))
            else:
                ret = np.zeros(S.shape)
            ret.fill(default_value)
            batch = 0
            v_idx = 0
            for length in L:
                for _ in range(length):
                    ret[batch][I[v_idx]] = V[v_idx]
                    v_idx += 1
                batch += 1
            return [ret]

        self.assertDeviceChecks(dc, op, [L, I, V], [0])
        self.assertReferenceChecks(gc, op, [L, I, V], batch_sparse_to_dense_ref)
        self.assertGradientChecks(gc, op, [L, I, V], 2, [0])
        self.assertDeviceChecks(dc, op2, [L, I, V, S], [0])
        self.assertReferenceChecks(gc, op2, [L, I, V, S], batch_sparse_to_dense_ref)
        self.assertGradientChecks(gc, op2, [L, I, V, S], 2, [0])
        self.assertDeviceChecks(dc, op, [L.astype(np.int32), I, V], [0])
        self.assertReferenceChecks(gc, op, [L.astype(np.int32), I, V], batch_sparse_to_dense_ref)
        self.assertGradientChecks(gc, op, [L.astype(np.int32), I, V], 2, [0])

    @given(
        batch_size=st.integers(5, 10),
        dense_last_dim=st.integers(5, 10),
        **hu.gcs
    )
    @settings(deadline=None)
    def test_batch_dense_to_sparse(self, batch_size, dense_last_dim, gc, dc):
        L = np.random.randint(1, dense_last_dim + 1, size=(batch_size))
        # The following logic ensure that indices in each batch will not be duplicated
        I = np.array([]).astype(np.int32)
        for l in L:
            I_l = np.random.choice(dense_last_dim, l, replace=False)
            I = np.concatenate((I, I_l))
        D = np.random.rand(batch_size, dense_last_dim).astype(np.float32)

        op = core.CreateOperator(
            'BatchDenseToSparse',
            ['L', 'I', 'D'],
            ['V'],
        )

        def batch_dense_to_sparse_ref(L, I, D):
            ret = np.zeros(I.shape)
            batch = 0
            i_idx = 0
            for length in L:
                for _ in range(length):
                    ret[i_idx] = D[batch][I[i_idx]]
                    i_idx += 1
                batch += 1
            return [ret]
        print(L, I, D)

        self.assertDeviceChecks(dc, op, [L, I, D], [0])
        self.assertReferenceChecks(gc, op, [L, I, D], batch_dense_to_sparse_ref)
        self.assertGradientChecks(gc, op, [L, I, D], 2, [0])
        self.assertDeviceChecks(dc, op, [L.astype(np.int32), I, D], [0])
        self.assertReferenceChecks(gc, op, [L.astype(np.int32), I, D], batch_dense_to_sparse_ref)
        self.assertGradientChecks(gc, op, [L.astype(np.int32), I, D], 2, [0])
