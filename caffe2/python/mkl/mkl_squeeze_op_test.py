




import unittest
import hypothesis.strategies as st
from hypothesis import given
import numpy as np
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.mkl_test_util as mu


@unittest.skipIf(
    not workspace.C.has_mkldnn, "Skipping as we do not have mkldnn."
)
class MKLSqueezeTest(hu.HypothesisTestCase):
    @given(
        squeeze_dims=st.lists(st.integers(0, 3), min_size=1, max_size=3),
        inplace=st.booleans(),
        **mu.gcs
    )
    def test_mkl_squeeze(self, squeeze_dims, inplace, gc, dc):
        shape = [
            1 if dim in squeeze_dims else np.random.randint(1, 5)
            for dim in range(4)
        ]
        X = np.random.rand(*shape).astype(np.float32)
        op = core.CreateOperator(
            "Squeeze", "X", "X" if inplace else "Y", dims=squeeze_dims
        )
        self.assertDeviceChecks(dc, op, [X], [0])


if __name__ == "__main__":
    unittest.main()
