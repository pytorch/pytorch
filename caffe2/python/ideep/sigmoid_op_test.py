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


@unittest.skipIf(not workspace.C.use_mkldnn, "No MKLDNN support.")
class SigmoidTest(hu.HypothesisTestCase):
    @given(X=hu.tensor(dtype=np.float32),
           inplace=st.booleans(),
           **hu.gcs)
    def test_sigmoid(self, X, inplace, gc, dc):
        op = core.CreateOperator(
            "Sigmoid",
            ["X"],
            ["Y"] if not inplace else ["X"],
        )

        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0])


if __name__ == "__main__":
    unittest.main()
