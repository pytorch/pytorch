from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


class TestReduceFrontSum(hu.HypothesisTestCase):

    def reduce_op_test(self, op_name, op_ref, in_data, num_reduce_dims, device):
        op = core.CreateOperator(
            op_name,
            ["inputs"],
            ["outputs"],
            num_reduce_dim=num_reduce_dims
        )

        self.assertReferenceChecks(
            device_option=device,
            op=op,
            inputs=[in_data],
            reference=op_ref
        )

        self.assertGradientChecks(
            device, op, [in_data], 0, [0], stepsize=1e-2, threshold=1e-2)

    @given(num_reduce_dim=st.integers(1, 3), **hu.gcs)
    def test_reduce_from_sum(self, num_reduce_dim, gc, dc):
        X = np.random.rand(7, 4, 3, 5).astype(np.float32)

        def ref_sum(X):
            return [np.sum(X, axis=(tuple(range(num_reduce_dim))))]

        self.reduce_op_test("ReduceFrontSum", ref_sum, X, num_reduce_dim, gc)

    @given(num_reduce_dim=st.integers(1, 3), **hu.gcs)
    def test_reduce_from_mean(self, num_reduce_dim, gc, dc):
        X = np.random.rand(6, 7, 8, 2).astype(np.float32)

        def ref_mean(X):
            return [np.mean(X, axis=(tuple(range(num_reduce_dim))))]

        self.reduce_op_test("ReduceFrontMean", ref_mean, X, num_reduce_dim, gc)
