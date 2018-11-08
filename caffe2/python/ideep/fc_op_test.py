from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import hypothesis.strategies as st
from hypothesis import given, settings
import numpy as np
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.ideep_test_util as mu

@unittest.skipIf(not workspace.C.use_mkldnn, "No MKLDNN support.")
class FcTest(hu.HypothesisTestCase):
    @given(n=st.integers(1, 5), m=st.integers(1, 5),
           k=st.integers(1, 5), **mu.gcs)
    def test_fc(self,n, m, k, gc, dc):
        X = np.random.rand(m, k).astype(np.float32) - 0.5
        W = np.random.rand(n, k).astype(np.float32) - 0.5
        b = np.random.rand(n).astype(np.float32) - 0.5

        op = core.CreateOperator(
            'FC',
            ['X', 'W', 'b'],
            ["Y"]
            )

        self.assertDeviceChecks(dc, op, [X, W, b], [0])

        for i in range(3):
            self.assertGradientChecks(gc, op, [X, W, b], i, [0])


if __name__ == "__main__":
    unittest.main()
