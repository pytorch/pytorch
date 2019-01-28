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


@unittest.skipIf(
    not workspace.C.has_mkldnn, "Skipping as we do not have mkldnn."
)
class MKLConcatTest(hu.HypothesisTestCase):
    @given(
        batch_size=st.integers(1, 10),
        channel_splits=st.lists(st.integers(1, 10), min_size=1, max_size=3),
        height=st.integers(1, 10),
        width=st.integers(1, 10),
        **mu.gcs
    )
    def test_mkl_concat(
        self, batch_size, channel_splits, height, width, gc, dc
    ):
        Xs = [
            np.random.rand(batch_size, channel,
                           height, width).astype(np.float32)
            for channel in channel_splits
        ]
        op = core.CreateOperator(
            "Concat",
            ["X_{}".format(i) for i in range(len(Xs))],
            ["concat_result", "split_info"],
            order="NCHW",
        )
        self.assertDeviceChecks(dc, op, Xs, [0])


if __name__ == "__main__":
    import unittest
    unittest.main()
