from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


class TestReductionOps(hu.HypothesisTestCase):

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

    @given(n=st.integers(1, 65536), **hu.gcs)
    def test_elementwise_sqrsum(self, n, gc, dc):
        X = np.random.rand(n).astype(np.float32)

        def sumsqr_op(X):
            return [np.sum(X * X)]

        op = core.CreateOperator(
            "SumSqrElements",
            ["X"],
            ["y"]
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=sumsqr_op,
        )

    @given(n=st.integers(5, 8), **hu.gcs)
    def test_elementwise_avg(self, n, gc, dc):
        X = np.random.rand(n).astype(np.float32)

        def avg_op(X):
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
            reference=avg_op,
        )

        self.assertGradientChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            outputs_to_check=0,
            outputs_with_grads=[0],
        )

    @given(batch_size=st.integers(1, 3),
           m=st.integers(1, 3),
           n=st.integers(1, 4),
           **hu.gcs)
    def test_rowwise_max(self, batch_size, m, n, gc, dc):
        X = np.random.rand(batch_size, m, n).astype(np.float32)

        def rowwise_max(X):
            return [np.max(X, axis=2)]

        op = core.CreateOperator(
            "RowwiseMax",
            ["x"],
            ["y"]
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=rowwise_max,
        )

    @given(batch_size=st.integers(1, 3),
           m=st.integers(1, 3),
           n=st.integers(1, 4),
           **hu.gcs)
    def test_columnwise_max(self, batch_size, m, n, gc, dc):
        X = np.random.rand(batch_size, m, n).astype(np.float32)

        def columnwise_max(X):
            return [np.max(X, axis=1)]

        op = core.CreateOperator(
            "ColwiseMax",
            ["x"],
            ["y"]
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=columnwise_max,
        )
