




import unittest
import hypothesis.strategies as st
from hypothesis import given, settings
import numpy as np
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.mkl_test_util as mu


@unittest.skipIf(not workspace.C.has_mkldnn,
                 "Skipping as we do not have mkldnn.")
class MKLFillTest(hu.HypothesisTestCase):
    @given(n=st.integers(1, 4), c=st.integers(1, 4),
           h=st.integers(1, 4), w=st.integers(1, 4),
           filler=st.sampled_from(
               ["XavierFill", "ConstantFill", "GaussianFill", "MSRAFill"]
           ),
           seed=st.integers(5, 10),
           **mu.gcs_cpu_mkl)
    def test_mkl_fill(self, n, c, h, w, filler, seed, gc, dc):
        op = core.CreateOperator(
            filler,
            [],
            ["Y"],
            shape=[n, c, h, w],
        )
        for d in dc:
            d.random_seed = seed
        self.assertDeviceChecks(dc, op, [], [0])

if __name__ == "__main__":
    import unittest
    unittest.main()
