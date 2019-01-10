from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from hypothesis import given
import hypothesis.strategies as st

from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu


class TestTTContraction(hu.HypothesisTestCase):
    @given(D=st.integers(min_value=5, max_value=20),
           K=st.integers(min_value=5, max_value=20),
           M=st.integers(min_value=5, max_value=20),
           N=st.integers(min_value=5, max_value=20),
           **hu.gcs)
    def test_tt_contraction(self, D, K, M, N, gc, dc):
        A = np.random.rand(K, M).astype(np.float32)
        B = np.random.rand(D, K, N).astype(np.float32)

        workspace.FeedBlob('A', A)
        workspace.FeedBlob('B', B)

        op = core.CreateOperator(
            'TTContraction',
            ['A', 'B'],
            ['C'],
            K=K,
            M=M,
            N=N)
        workspace.RunOperatorOnce(op)

        def tt_contraction_ref(A_, B_):
            return ((A_[:, :, np.newaxis] * B_[:, :, np.newaxis, :])
                    .sum(axis=1).flatten()),

        # Check against numpy reference
        self.assertReferenceChecks(gc, op, [A, B], tt_contraction_ref)
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [A, B], [0])
        # Gradient check wrt A
        self.assertGradientChecks(gc, op, [A, B], 0, [0])
        # Gradient check wrt B
        self.assertGradientChecks(gc, op, [A, B], 1, [0])
