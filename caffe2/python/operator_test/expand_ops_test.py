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

class TestExpandOps(hu.HypothesisTestCase):
    def run_expand_op_test(
        self, op_name, X, new_shape, gc, dc):
        op = core.CreateOperator(
            op_name,
            ["X"],
            ["new_shape"],
        )
        def ref(X):
            return np.array(X) * ones(new_shape)

        self.assertReferenceChecks(gc, op, [X], ref)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0])

    @given(X=hu.tensor(max_dim=3, dtype=np.float32),
           new_shape=hu.tensor1d(max_len=3, dtype=np.int32),
           **hu.gcs)
    def test_expand_normal(self, X, new_shape, gc, dc):
        self.run_expand_op_test(
            "ExpandNormal", X, new_shape, gc, dc)

if __name__ == "__main__":
    unittest.main()
