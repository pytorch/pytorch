from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np

import unittest

class TestUtilityOps(hu.HypothesisTestCase):

    @unittest.skipIf(not workspace.has_gpu_support, "No gpu support.")
    @given(dtype=st.sampled_from([np.float32, np.int32, np.int64]),
           ndims=st.integers(min_value=1, max_value=5),
           seed=st.integers(min_value=0, max_value=65536),
           null_axes=st.booleans(),
           engine=st.sampled_from(['CUDNN', None]),
           **hu.gcs_gpu_only)
    def test_transpose(self, dtype, ndims, seed, null_axes, engine, gc, dc):
        dims = (np.random.rand(ndims) * 16 + 1).astype(np.int32)
        X = (np.random.rand(*dims) * 16).astype(dtype)

        if null_axes:
            axes = None
            op = core.CreateOperator(
                "Transpose",
                ["input"], ["output"],
                engine=engine)
        else:
            np.random.seed(int(seed))
            axes = [int(v) for v in list(np.random.permutation(X.ndim))]
            op = core.CreateOperator(
                "Transpose",
                ["input"], ["output"],
                axes=axes,
                engine=engine)

        def transpose_ref(x, axes):
            return (np.transpose(x, axes),)

        self.assertReferenceChecks(gc, op, [X, axes],
                                   transpose_ref)

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
