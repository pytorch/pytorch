from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, dyndep
from hypothesis import given

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


dyndep.InitOpsLibrary('@/caffe2/caffe2/contrib/aten:aten_op')


class TestATen(hu.HypothesisTestCase):

    @given(inputs=hu.tensors(n=2), **hu.gcs)
    def test_add(self, inputs, gc, dc):
        op = core.CreateOperator(
             "ATen",
             ["X", "Y"],
             ["Z"],
             operator="add")

        def ref(X, Y):
            return [X + Y]
        self.assertReferenceChecks(gc, op, inputs, ref)

    @given(inputs=hu.tensors(n=1), **hu.gcs)
    def test_pow(self, inputs, gc, dc):
        op = core.CreateOperator(
            "ATen",
            ["S"],
            ["Z"],
            operator="pow", exponent=2.0)

        def ref(X):
            return [np.square(X)]

        self.assertReferenceChecks(gc, op, inputs, ref)

    @given(x=st.integers(min_value=2, max_value=8), **hu.gcs)
    def test_sort(self, x, gc, dc):
        inputs = [np.random.permutation(x)]
        op = core.CreateOperator(
            "ATen",
            ["S"],
            ["Z", "I"],
            operator="sort")

        def ref(X):
            return [np.sort(X), np.argsort(X)]
        self.assertReferenceChecks(gc, op, inputs, ref)

    @given(inputs=hu.tensors(n=1), **hu.gcs)
    def test_sum(self, inputs, gc, dc):
        op = core.CreateOperator(
            "ATen",
            ["S"],
            ["Z"],
            operator="sum")

        def ref(X):
            return [np.sum(X)]

        self.assertReferenceChecks(gc, op, inputs, ref)

    @given(**hu.gcs)
    def test_ones(self, gc, dc):
        op = core.CreateOperator(
            "ATen",
            [],
            ["Z"],
            operator="ones", type="float", size={2, 4})

        def ref():
            return [np.ones([2, 4])]

        self.assertReferenceChecks(gc, op, [], ref)


if __name__ == "__main__":
    import unittest
    unittest.main()
