from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core, dyndep
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


class TestBucketizeOp(hu.HypothesisTestCase):
    @given(
        x=hu.tensor(
            min_dim=1, max_dim=2, dtype=np.float32,
            elements=st.floats(min_value=-5, max_value=5)),
        **hu.gcs)
    def test_bucketize_op(self, x, gc, dc):
        length = np.random.randint(low=1, high=5)
        boundaries = np.random.randn(length) * 5
        boundaries.sort()

        def ref(x, boundaries):
            bucket_idx = np.digitize(x, boundaries, right=True)
            return [bucket_idx]

        op = core.CreateOperator('Bucketize',
                                 ["X"], ["INDICES"],
                                 boundaries=boundaries)
        self.assertReferenceChecks(gc, op, [x, boundaries], ref)


if __name__ == "__main__":
    import unittest
    unittest.main()
