from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


class TestLars(hu.HypothesisTestCase):

    @given(offset=st.floats(min_value=0, max_value=100), **hu.gcs)
    def test_lars(self, offset, dc, gc):
        X = np.random.rand(6, 7, 8, 9).astype(np.float32)
        dX = np.random.rand(6, 7, 8, 9).astype(np.float32)

        def ref_lars(X, dX):
            return [1. / (np.linalg.norm(dX) / np.linalg.norm(X) + offset)]

        op = core.CreateOperator(
            "Lars",
            ["X", "dX"],
            ["rescale_factor"],
            offset=offset
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X, dX],
            reference=ref_lars
        )
