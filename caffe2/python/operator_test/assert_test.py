from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from hypothesis import given
import hypothesis.strategies as st
from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu


class TestAssert(hu.HypothesisTestCase):
    @given(
        dtype=st.sampled_from(['bool_', 'int32', 'int64']),
        shape=st.lists(elements=st.integers(1, 10), min_size=1, max_size=4),
        **hu.gcs)
    def test_assert(self, dtype, shape, gc, dc):
        test_tensor = np.random.rand(*shape).astype(np.dtype(dtype))

        op = core.CreateOperator('Assert', ['X'], [])

        def assert_ref(X):
            return []

        try:
            self.assertReferenceChecks(gc, op, [test_tensor], assert_ref)
        except Exception:
            assert(not np.all(test_tensor))
