from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import hypothesis.strategies as st
from hypothesis import given
import numpy as np
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.ideep_test_util as mu

@unittest.skipIf(not workspace.C.use_mkldnn, "No MKLDNN support.")
class ReluTest(hu.HypothesisTestCase):
    @given(X=hu.tensor(),
           inplace=st.booleans(),
           **mu.gcs)
    def test_relu(self, X, inplace, gc, dc):
        op = core.CreateOperator(
            "Relu",
            ["X"],
            ["Y"] if not inplace else ["X"],
        )
        # go away from the origin point to avoid kink problems
        X += 0.02 * np.sign(X)
        X[X == 0.0] += 0.02

        self.assertDeviceChecks(dc, op, [X], [0])

        self.assertGradientChecks(gc, op, [X], 0, [0])


if __name__ == "__main__":
    unittest.main()
