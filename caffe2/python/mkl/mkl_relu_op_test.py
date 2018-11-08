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
import caffe2.python.mkl_test_util as mu


@unittest.skipIf(not workspace.C.has_mkldnn,
                 "Skipping as we do not have mkldnn.")
class MKLReluTest(hu.HypothesisTestCase):
    @given(size=st.integers(8, 20),
           input_channels=st.integers(1, 3),
           batch_size=st.integers(1, 3),
           inplace=st.booleans(),
           **mu.gcs)
    def test_mkl_relu(self, size, input_channels, batch_size, inplace, gc, dc):
        op = core.CreateOperator(
            "Relu",
            ["X"],
            ["Y"] if not inplace else ["X"],
        )
        X = np.random.rand(
            batch_size, input_channels, size, size).astype(np.float32) - 0.5
        self.assertDeviceChecks(dc, op, [X], [0])


if __name__ == "__main__":
    import unittest
    unittest.main()
