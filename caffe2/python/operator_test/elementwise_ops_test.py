from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


class TestElementwiseOps(hu.HypothesisTestCase):

    @given(n=st.integers(2, 10), m=st.integers(4, 6),
           d=st.integers(2, 3), **hu.gcs)
    def test_div(self, n, m, d, gc, dc):
        X = np.random.rand(n, m, d).astype(np.float32)
        Y = np.random.rand(n, m, d).astype(np.float32) + 5.0

        def div_op(X, Y):
            return [np.divide(X, Y)]

        op = core.CreateOperator(
            "Div",
            ["X", "Y"],
            ["Z"]
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X, Y],
            reference=div_op,
        )

        self.assertGradientChecks(
            gc, op, [X, Y], 0, [0], stepsize=1e-4, threshold=1e-2)
