
import unittest

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, dyndep
from hypothesis import given, settings


dyndep.InitOpsLibrary("@/caffe2/modules/detectron:detectron_ops")


class TestUpsampleNearestOp(hu.HypothesisTestCase):
    @given(
        N=st.integers(1, 3),
        H=st.integers(10, 300),
        W=st.integers(10, 300),
        scale=st.integers(1, 3),
        **hu.gcs
    )
    @settings(deadline=None, max_examples=20)
    def test_upsample_nearest_op(self, N, H, W, scale, gc, dc):
        C = 32
        X = np.random.randn(N, C, H, W).astype(np.float32)
        op = core.CreateOperator("UpsampleNearest", ["X"], ["Y"], scale=scale)

        def ref(X):
            outH = H * scale
            outW = W * scale
            outH_idxs, outW_idxs = np.meshgrid(
                np.arange(outH), np.arange(outW), indexing="ij"
            )
            inH_idxs = (outH_idxs / scale).astype(np.int32)
            inW_idxs = (outW_idxs / scale).astype(np.int32)
            Y = X[:, :, inH_idxs, inW_idxs]
            return [Y]

        self.assertReferenceChecks(device_option=gc, op=op, inputs=[X], reference=ref)


if __name__ == "__main__":
    unittest.main()
