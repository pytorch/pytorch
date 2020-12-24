




import unittest
import hypothesis.strategies as st
from hypothesis import given, settings
import numpy as np
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.mkl_test_util as mu


@unittest.skipIf(not workspace.C.has_mkldnn,
                 "Skipping as we do not have mkldnn.")
class MKLFcTest(hu.HypothesisTestCase):
    @given(n=st.integers(1, 5), m=st.integers(1, 5),
           k=st.integers(1, 5), **mu.gcs)

    def test_mkl_fc(self,n, m, k, gc, dc):
        X = np.random.rand(m, k).astype(np.float32) - 0.5
        W = np.random.rand(n, k).astype(np.float32) - 0.5
        b = np.random.rand(n).astype(np.float32) - 0.5

        op = core.CreateOperator(
            'FC',
            ['X', 'W', 'b'],
            ["Y"]
            )

        self.assertDeviceChecks(dc, op, [X, W, b], [0])


if __name__ == "__main__":
    import unittest
    unittest.main()
