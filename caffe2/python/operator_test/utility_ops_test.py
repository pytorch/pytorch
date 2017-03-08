from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


class TestUtilityOps(hu.HypothesisTestCase):

    @given(n=st.integers(4, 5), m=st.integers(6, 7),
           d=st.integers(2, 3), **hu.gcs)
    def test_elementwise_max(self, n, m, d, gc, dc):
        X = np.random.rand(n, m, d).astype(np.float32)
        Y = np.random.rand(n, m, d).astype(np.float32)
        Z = np.random.rand(n, m, d).astype(np.float32)

        def max_op(X, Y, Z):
            return [np.maximum(np.maximum(X, Y), Z)]

        op = core.CreateOperator(
            "Max",
            ["X", "Y", "Z"],
            ["mx"]
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X, Y, Z],
            reference=max_op,
        )

    @given(n=st.integers(5, 8), **hu.gcs)
    def test_elementwise_sum(self, n, gc, dc):
        X = np.random.rand(n).astype(np.float32)

        def sum_op(X):
            return [np.sum(X)]

        op = core.CreateOperator(
            "SumElements",
            ["X"],
            ["y"]
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=sum_op,
        )

        self.assertGradientChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            outputs_to_check=0,
            outputs_with_grads=[0],
        )

    @given(n=st.integers(5, 8), **hu.gcs)
    def test_elementwise_avg(self, n, gc, dc):
        X = np.random.rand(n).astype(np.float32)

        def sum_op(X):
            return [np.mean(X)]

        op = core.CreateOperator(
            "SumElements",
            ["X"],
            ["y"],
            average=1
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=sum_op,
        )
