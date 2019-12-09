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

    @given(offset=st.floats(min_value=0, max_value=100),
    lr_min=st.floats(min_value=1e-8, max_value=1e-6),
    **hu.gcs)
    def test_lars(self, offset, lr_min, dc, gc):
        X = np.random.rand(6, 7, 8, 9).astype(np.float32)
        dX = np.random.rand(6, 7, 8, 9).astype(np.float32)
        wd = np.array([1e-4]).astype(np.float32)
        trust = np.random.rand(1).astype(np.float32)
        lr_max = np.random.rand(1).astype(np.float32)

        def ref_lars(X, dX, wd, trust, lr_max):
            rescale_factor = \
                trust / (np.linalg.norm(dX) / np.linalg.norm(X) + wd + offset)
            rescale_factor = np.minimum(rescale_factor, lr_max)
            rescale_factor = np.maximum(rescale_factor, lr_min)
            return [rescale_factor]

        op = core.CreateOperator(
            "Lars",
            ["X", "dX", "wd", "trust", "lr_max"],
            ["rescale_factor"],
            offset=offset,
            lr_min=lr_min,
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X, dX, wd, trust, lr_max],
            reference=ref_lars
        )
