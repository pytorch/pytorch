from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect

import numpy as np

from hypothesis import assume, given, settings
import hypothesis.strategies as st

from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from functools import reduce
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.ideep_test_util as mu


class TestMatMul(hu.HypothesisTestCase):
    @given(
        M=st.integers(min_value=1, max_value=10),
        K=st.integers(min_value=1, max_value=10),
        N=st.integers(min_value=1, max_value=10),
        trans_a=st.booleans(),
        trans_b=st.booleans(),
        use_bias=st.booleans(),
        **mu.gcs_ideep_only
    )
    def test_matmulwithbias_2Dx2D(self, M, K, N, trans_a, trans_b, use_bias, gc, dc):
        trans_b = True
        trans_b_ = 1 if trans_b else 0
        trans_a_ = 1 if trans_a else 0
        X = np.random.rand(M, K).astype(np.float32) - 0.5
        if trans_a:
            X = X.transpose()

        Y = np.random.rand(K, N).astype(np.float32) - 0.5
        if trans_b:
            Y = Y.transpose()
        if use_bias:
            Bias = np.random.rand(N).astype(np.float32) - 0.5

        op = core.CreateOperator(
            'MatMulWithBias',
            ['A', 'B', 'Bias'] if use_bias else ['A', 'B'],
            'out',
            trans_a=trans_a_,
            trans_b=trans_b_,
            device_option=dc[0]
        )

        def matmulWithBias_ref(X, Y, Bias, trans_a, trans_b):
            XX = X.transpose() if trans_a else X
            YY = Y.transpose() if trans_b else Y
            output = XX.dot(YY)
            output = output + Bias
            return (output, )

        def matmul_ref(X, Y, trans_a, trans_b):
            XX = X.transpose() if trans_a else X
            YY = Y.transpose() if trans_b else Y
            return (XX.dot(YY), )

        # Check against numpy reference
        self.assertReferenceChecks(gc, op, [X, Y, Bias, trans_a, trans_b] if use_bias else [X, Y, trans_a, trans_b],
                                   matmulWithBias_ref if use_bias else matmul_ref)
        # Gradient check wrt X
        self.assertGradientChecks(gc, op, [X, Y, Bias] if use_bias else [X, Y], 0, [0])
        # Gradient check wrt Y
        self.assertGradientChecks(gc, op, [X, Y, Bias] if use_bias else [X, Y], 1, [0])
        if use_bias:
            self.assertGradientChecks(gc, op, [X, Y, Bias], 2, [0])

    @given(
        B=st.integers(min_value=1, max_value=10),
        M=st.integers(min_value=1, max_value=10),
        K=st.integers(min_value=1, max_value=10),
        N=st.integers(min_value=1, max_value=10),
        trans_a=st.booleans(),
        trans_b=st.booleans(),
        use_bias=st.booleans(),
        **mu.gcs_ideep_only
    )
    def test_matmulwithbias_3Dx2D(self, B, M, K, N, trans_a, trans_b, use_bias, gc, dc):
        trans_b_ = 1 if trans_b else 0
        trans_a_ = 1 if trans_a else 0
        X = np.random.rand(B, M, K).astype(np.float32) - 0.5
        if trans_a:
            X = X.transpose(2, 0, 1)

        Y = np.random.rand(K, N).astype(np.float32) - 0.5
        if trans_b:
            Y = Y.transpose()
        Bias = np.random.rand(N).astype(np.float32) - 0.5

        op = core.CreateOperator(
            'MatMulWithBias',
            ['X', 'Y', 'B'] if use_bias else ['X', 'Y'],
            'out',
            axis_A=1 if trans_a else 2,
            trans_a=trans_a_,
            trans_b=trans_b_,
            device_option=dc[0]
        )

        def matmulWithBias_ref(X, Y, Bias, trans_a, trans_b):
            XX = X.transpose(1, 2, 0) if trans_a else X
            XX = XX.reshape(-1, K)
            YY = Y.transpose() if trans_b else Y
            output = XX.dot(YY)
            output = output.reshape(B, M, -1)
            output = output + Bias
            return (output, )

        def matmul_ref(X, Y, trans_a, trans_b):
            XX = X.transpose(1, 2, 0) if trans_a else X
            XX = XX.reshape(-1, K)
            YY = Y.transpose() if trans_b else Y
            output = XX.dot(YY)
            output = output.reshape(B, M, -1)
            return (output, )

        # Check against numpy reference
        self.assertReferenceChecks(gc, op, [X, Y, Bias, trans_a, trans_b] if use_bias else [X, Y, trans_a, trans_b],
                                   matmulWithBias_ref if use_bias else matmul_ref)
        # Gradient check wrt X
        self.assertGradientChecks(gc, op, [X, Y, Bias] if use_bias else [X, Y], 0, [0])
        # Gradient check wrt Y
        self.assertGradientChecks(gc, op, [X, Y, Bias] if use_bias else [X, Y], 1, [0])
        if use_bias:
            self.assertGradientChecks(gc, op, [X, Y, Bias], 2, [0])

if __name__ == "__main__":
    import unittest
    unittest.main()
