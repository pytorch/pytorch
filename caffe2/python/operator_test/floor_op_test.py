from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given
import hypothesis.strategies as st
import caffe2.python.hypothesis_test_util as hu
import numpy as np

import unittest


class TestFloor(hu.HypothesisTestCase):

    @given(X=hu.tensor(),
           engine=st.sampled_from(["", "CUDNN"]),
           **hu.gcs)
    def test_floor(self, X, gc, dc, engine):
        op = core.CreateOperator("Floor", ["X"], ["Y"], engine=engine)

        def floor_ref(X):
            return (np.floor(X),)

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=floor_ref)

        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [X], [0])


if __name__ == "__main__":
    unittest.main()
