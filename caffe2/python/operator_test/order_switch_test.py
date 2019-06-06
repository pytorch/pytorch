from __future__ import absolute_import, division, print_function, unicode_literals

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
from caffe2.python import core, utils
from hypothesis import given


class OrderSwitchOpsTest(hu.HypothesisTestCase):
    @given(
        X=hu.tensor(min_dim=3, max_dim=5, min_value=1, max_value=5),
        engine=st.sampled_from(["", "CUDNN"]),
        **hu.gcs
    )
    def test_nchw2nhwc(self, X, engine, gc, dc):
        op = core.CreateOperator("NCHW2NHWC", ["X"], ["Y"], engine=engine)

        def nchw2nhwc_ref(X):
            return (utils.NCHW2NHWC(X),)

        self.assertReferenceChecks(gc, op, [X], nchw2nhwc_ref)
        self.assertGradientChecks(gc, op, [X], 0, [0])
        self.assertDeviceChecks(dc, op, [X], [0])

    @given(
        X=hu.tensor(min_dim=3, max_dim=5, min_value=1, max_value=5),
        engine=st.sampled_from(["", "CUDNN"]),
        **hu.gcs
    )
    def test_nhwc2nchw(self, X, engine, gc, dc):
        op = core.CreateOperator("NHWC2NCHW", ["X"], ["Y"], engine=engine)

        def nhwc2nchw_ref(X):
            return (utils.NHWC2NCHW(X),)

        self.assertReferenceChecks(gc, op, [X], nhwc2nchw_ref)
        self.assertGradientChecks(gc, op, [X], 0, [0])
        self.assertDeviceChecks(dc, op, [X], [0])
