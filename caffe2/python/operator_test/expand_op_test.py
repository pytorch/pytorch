from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace
from hypothesis import given

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
import itertools as it


class TestExpandOp(hu.HypothesisTestCase):
    def run_expand_op_test(
        self, op_name, X, random_flag, gc, dc):
        if (random_flag):
            shape_length = np.random.randint(5)
        else:
            shape_length = 4
        shape_list = []
        j = shape_length - 1
        i = X.ndim - 1
        while i >= 0 or j >= 0:
            k = np.random.randint(5) + 1
            if i >= 0 and X.shape[i] != 1:
                if np.random.randint(2) == 0:
                    k = 1
                else:
                    k = X.shape[i]
            shape_list.insert(0, k)
            i -= 1
            j -= 1
        shape = np.array(shape_list, dtype=np.int64)

        op = core.CreateOperator(
            op_name,
            ["X", "shape"],
            ["Y"],
        )
        def ref(X, shape):
            return (X * np.ones(shape),)

        self.assertReferenceChecks(gc, op, [X, shape], ref)
        self.assertDeviceChecks(dc, op, [X, shape], [0])
        self.assertGradientChecks(gc, op, [X, shape], 0, [0])

    @given(X=hu.tensor(max_dim=5, dtype=np.float32),
           **hu.gcs)
    def test_expand_rand_shape(self, X, gc, dc):
        self.run_expand_op_test(
            "Expand", X, True, gc, dc)

    @given(X=hu.tensor(max_dim=5, dtype=np.float32),
           **hu.gcs)
    def test_expand_nonrand_shape(self, X, gc, dc):
        self.run_expand_op_test(
            "Expand", X, False, gc, dc)
