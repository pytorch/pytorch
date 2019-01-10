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


def rowmux(select_vec, left, right):
    select = [[s] * len(left) for s in select_vec]
    return mux(select, left, right)


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
           **hu.gcs_cpu_only)
    def test_where(self, N, gc, dc, engine):
        C = np.random.rand(N).astype(bool)
        X = np.random.rand(N).astype(np.float32)
        Y = np.random.rand(N).astype(np.float32)
        op = core.CreateOperator("Where", ["C", "X", "Y"], ["Z"], engine=engine)
        self.assertDeviceChecks(dc, op, [C, X, Y], [0])
        self.assertReferenceChecks(gc, op, [C, X, Y], mux)

    @given(N=st.integers(min_value=1, max_value=10),
           engine=st.sampled_from(["", "CUDNN"]),
           **hu.gcs_cpu_only)
    def test_where_dim2(self, N, gc, dc, engine):
        C = np.random.rand(N, N).astype(bool)
        X = np.random.rand(N, N).astype(np.float32)
        Y = np.random.rand(N, N).astype(np.float32)
        op = core.CreateOperator("Where", ["C", "X", "Y"], ["Z"], engine=engine)
        self.assertDeviceChecks(dc, op, [C, X, Y], [0])
        self.assertReferenceChecks(gc, op, [C, X, Y], mux)


class TestRowWhere(hu.HypothesisTestCase):

    def test_reference(self):
        self.assertTrue((
            np.array([1, 2]) == rowmux([True],
                                       [1, 2],
                                       [3, 4])[0]
        ).all())
        self.assertTrue((
            np.array([[1, 2], [7, 8]]) == rowmux([True, False],
                                                 [[1, 2], [3, 4]],
                                                 [[5, 6], [7, 8]])[0]
        ).all())

    @given(N=st.integers(min_value=1, max_value=10),
           engine=st.sampled_from(["", "CUDNN"]),
           **hu.gcs_cpu_only)
    def test_rowwhere(self, N, gc, dc, engine):
        C = np.random.rand(N).astype(bool)
        X = np.random.rand(N).astype(np.float32)
        Y = np.random.rand(N).astype(np.float32)
        op = core.CreateOperator(
            "Where",
            ["C", "X", "Y"],
            ["Z"],
            broadcast_on_rows=True,
            engine=engine,
        )
        self.assertDeviceChecks(dc, op, [C, X, Y], [0])
        self.assertReferenceChecks(gc, op, [C, X, Y], mux)

    @given(N=st.integers(min_value=1, max_value=10),
           engine=st.sampled_from(["", "CUDNN"]),
           **hu.gcs_cpu_only)
    def test_rowwhere_dim2(self, N, gc, dc, engine):
        C = np.random.rand(N).astype(bool)
        X = np.random.rand(N, N).astype(np.float32)
        Y = np.random.rand(N, N).astype(np.float32)
        op = core.CreateOperator(
            "Where",
            ["C", "X", "Y"],
            ["Z"],
            broadcast_on_rows=True,
            engine=engine,
        )
        self.assertDeviceChecks(dc, op, [C, X, Y], [0])
        self.assertReferenceChecks(gc, op, [C, X, Y], rowmux)


class TestIsMemberOf(hu.HypothesisTestCase):

    @given(N=st.integers(min_value=1, max_value=10),
           engine=st.sampled_from(["", "CUDNN"]),
           **hu.gcs_cpu_only)
    def test_is_member_of(self, N, gc, dc, engine):
        X = np.random.randint(10, size=N).astype(np.int64)
        values = [0, 3, 4, 6, 8]
        op = core.CreateOperator(
            "IsMemberOf",
            ["X"],
            ["Y"],
            value=np.array(values),
            engine=engine,
        )
        self.assertDeviceChecks(dc, op, [X], [0])
        values = set(values)

        def test(x):
            return [np.vectorize(lambda x: x in values)(x)]
        self.assertReferenceChecks(gc, op, [X], test)


if __name__ == "__main__":
    unittest.main()
