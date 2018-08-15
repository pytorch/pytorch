from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import caffe2.python.hypothesis_test_util as hu
from caffe2.python import core
from hypothesis import given
import hypothesis.strategies as st


class OrderSwitchOpsTest(hu.HypothesisTestCase):
    @given(
        n=st.integers(1, 5),
        c=st.integers(1, 5),
        h=st.integers(1, 5),
        w=st.integers(1, 5),
        **hu.gcs)
    def test_nchw2nhwc(self, n, c, h, w, gc, dc):
        X = np.random.randn(n, c, h, w).astype(np.float32)

        op = core.CreateOperator("NCHW2NHWC", ["X"], ["Y"],
                                 device_option=gc)

        def nchw2nhwc_ref(X):
            X_reshaped = X.transpose((0, 2, 3, 1))
            return (X_reshaped,)

        self.assertReferenceChecks(gc, op, [X], nchw2nhwc_ref)
        self.assertGradientChecks(gc, op, [X], 0, [0])
        self.assertDeviceChecks(dc, op, [X], [0])

    @given(
        n=st.integers(1, 5),
        c=st.integers(1, 5),
        h=st.integers(1, 5),
        w=st.integers(1, 5),
        **hu.gcs)
    def test_nhwc2nchw(self, n, c, h, w, gc, dc):
        X = np.random.randn(n, h, w, c).astype(np.float32)

        op = core.CreateOperator("NHWC2NCHW", ["X"], ["Y"],
                                 device_option=gc)

        def nhwc2nchw_ref(X):
            X_reshaped = X.transpose((0, 3, 1, 2))
            return (X_reshaped,)

        self.assertReferenceChecks(gc, op, [X], nhwc2nchw_ref)
        self.assertGradientChecks(gc, op, [X], 0, [0])
        self.assertDeviceChecks(dc, op, [X], [0])
