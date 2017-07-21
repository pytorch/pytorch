from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from hypothesis import given
import hypothesis.strategies as st
from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu


class TestBooleanMaskOp(hu.HypothesisTestCase):

    @given(x=hu.tensor(min_dim=1,
                       max_dim=5,
                       elements=st.floats(min_value=0.5, max_value=1.0)),
           **hu.gcs)
    def test_boolean_mask(self, x, gc, dc):
        op = core.CreateOperator("BooleanMask",
                                 ["data", "mask"],
                                 "masked_data")
        mask = np.random.choice(a=[True, False], size=x.shape[0])

        def ref(x, mask):
            return (x[mask],)

        self.assertReferenceChecks(gc, op, [x, mask], ref)
        self.assertDeviceChecks(dc, op, [x, mask], [0])

    @given(x=hu.tensor(min_dim=1,
                       max_dim=5,
                       elements=st.floats(min_value=0.5, max_value=1.0)),
           **hu.gcs)
    def test_boolean_mask_indices(self, x, gc, dc):
        op = core.CreateOperator("BooleanMask",
                                 ["data", "mask"],
                                 ["masked_data", "masked_indices"])
        mask = np.random.choice(a=[True, False], size=x.shape[0])

        def ref(x, mask):
            return (x[mask], np.where(mask)[0])

        self.assertReferenceChecks(gc, op, [x, mask], ref)
        self.assertDeviceChecks(dc, op, [x, mask], [0])
