from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from hypothesis import given
import hypothesis.strategies as st

from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu


class TestMatMul(hu.HypothesisTestCase):
    @given(M=st.integers(min_value=1, max_value=10),
           K=st.integers(min_value=1, max_value=10),
           N=st.integers(min_value=1, max_value=10),
           trans_a=st.booleans(),
           trans_b=st.booleans(),
           **hu.gcs)
    def test_matmul(self, M, K, N, trans_a, trans_b, gc, dc):
        X = np.random.rand(M, K).astype(np.float32) - 0.5
        if trans_a:
            X = X.transpose()

        Y = np.random.rand(K, N).astype(np.float32) - 0.5
        if trans_b:
            Y = Y.transpose()

        op = core.CreateOperator(
            'MatMul', ['X', 'Y'], 'out',
            trans_a=trans_a, trans_b=trans_b)

        def matmul_ref(X, Y, trans_a, trans_b):
            XX = X.transpose() if trans_a else X
            YY = Y.transpose() if trans_b else Y
            return (XX.dot(YY),)

        # Check against numpy reference
        self.assertReferenceChecks(gc, op, [X, Y, trans_a, trans_b],
                                   matmul_ref)
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [X, Y], [0])
        # Gradient check wrt X
        self.assertGradientChecks(gc, op, [X, Y], 0, [0])
        # Gradient check wrt Y
        self.assertGradientChecks(gc, op, [X, Y], 1, [0])


class TestBatchMatMul(hu.HypothesisTestCase):
    @given(C=st.integers(min_value=1, max_value=10),
           M=st.integers(min_value=1, max_value=10),
           K=st.integers(min_value=1, max_value=10),
           N=st.integers(min_value=1, max_value=10),
           trans_a=st.booleans(),
           trans_b=st.booleans(),
           **hu.gcs)
    def test_batch_matmul(self, C, M, K, N, trans_a, trans_b, gc, dc):
        X = np.random.rand(C, M, K).astype(np.float32) - 0.5
        if trans_a:
            X = X.swapaxes(1, 2)

        Y = np.random.rand(C, K, N).astype(np.float32) - 0.5
        if trans_b:
            Y = Y.swapaxes(1, 2)

        op = core.CreateOperator(
            'BatchMatMul', ['X', 'Y'], 'out',
            trans_a=trans_a, trans_b=trans_b)

        def matmul_ref(X, Y, trans_a, trans_b):
            XX = X.swapaxes(1, 2) if trans_a else X
            YY = Y.swapaxes(1, 2) if trans_b else Y
            output = np.zeros((C, M, N)).astype(XX.dtype)
            for i in range(C):
                output[i] = XX[i].dot(YY[i])
            return (output,)

        # Check against numpy reference
        self.assertReferenceChecks(gc, op, [X, Y, trans_a, trans_b],
                                   matmul_ref)
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [X, Y], [0])
        # Gradient check wrt X
        self.assertGradientChecks(gc, op, [X, Y], 0, [0])
        # Gradient check wrt Y
        self.assertGradientChecks(gc, op, [X, Y], 1, [0])

if __name__ == "__main__":
    import unittest
    unittest.main()
