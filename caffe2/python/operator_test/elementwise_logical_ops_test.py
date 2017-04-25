from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given
import hypothesis.strategies as st
import caffe2.python.hypothesis_test_util as hu

import numpy as np
import unittest


def mux(select, left, right):
    return [np.vectorize(lambda c, x, y: x if c else y)(select, left, right)]


class TestWhere(hu.HypothesisTestCase):

    def test_reference(self):
        self.assertTrue((
            np.array([1, 4]) == mux([True, False],
                                    [1, 2],
                                    [3, 4])[0]
        ).all())
        self.assertTrue((
            np.array([[1], [4]]) == mux([[True], [False]],
                                        [[1], [2]],
                                        [[3], [4]])[0]
        ).all())

    @given(N=st.integers(min_value=1, max_value=10),
           engine=st.sampled_from(["", "CUDNN"]),
           **hu.gcs)
    def test_where(self, N, gc, dc, engine):
        C = np.random.rand(N).astype(bool)
        X = np.random.rand(N).astype(np.float32)
        Y = np.random.rand(N).astype(np.float32)
        op = core.CreateOperator("Where", ["C", "X", "Y"], ["Z"], engine=engine)
        self.assertDeviceChecks(dc, op, [C, X, Y], [0])
        self.assertReferenceChecks(gc, op, [C, X, Y], mux)

    @given(N=st.integers(min_value=1, max_value=10),
           engine=st.sampled_from(["", "CUDNN"]),
           **hu.gcs)
    def test_where_dim2(self, N, gc, dc, engine):
        C = np.random.rand(N, N).astype(bool)
        X = np.random.rand(N, N).astype(np.float32)
        Y = np.random.rand(N, N).astype(np.float32)
        op = core.CreateOperator("Where", ["C", "X", "Y"], ["Z"], engine=engine)
        self.assertDeviceChecks(dc, op, [C, X, Y], [0])
        self.assertReferenceChecks(gc, op, [C, X, Y], mux)


if __name__ == "__main__":
    unittest.main()
