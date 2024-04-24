




import unittest
import hypothesis.strategies as st
from hypothesis import given
import numpy as np
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.mkl_test_util as mu


@unittest.skipIf(not workspace.C.has_mkldnn,
                 "Skipping as we do not have mkldnn.")
class MKLSigmoidTest(hu.HypothesisTestCase):
    @given(n=st.integers(1, 5), m=st.integers(1, 5), inplace=st.booleans(),
           **mu.gcs)
    def test_mkl_sigmoid(self, n, m, inplace, gc, dc):
        X = np.random.rand(m, n).astype(np.float32)
        op = core.CreateOperator(
            "Sigmoid",
            ["X"],
            ["Y" if not inplace else "X"]
        )
        self.assertDeviceChecks(dc, op, [X], [0])


if __name__ == "__main__":
    import unittest
    unittest.main()
