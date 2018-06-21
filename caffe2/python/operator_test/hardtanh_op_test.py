fro __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given
import hypothesis.strategies as st
import caffe2.python.hypothesis_test_util as hu
import numpy as np

import unittest


class TestHardtanh(hu.HypothesisTestCase):

    @given(X=hu.tensor(),
           engine=st.sampled_from(["", "CUDNN"]),
            **hu.gcs)
    def test_hardtanh_1(self, X, gc, dc, engine):
        min_val = -1.0
        max_val = 1.0
        op = core.CreateOperator("Hardtanh", ["X"], ["Y"],
                                 min_val=min_val, max_val=max_val, engine=engine)
        X = TestHardtanh.fix0(X)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0])
        self.assertReferenceChecks(
            gc, op, [X], lambda x: TestHardtanh.hardtanh_ref(x, min_val=min_val, max_val=max_val)
        )

    @given(X=hu.tensor(),
           engine=st.sampled_from(["", "CUDNN"]),
            **hu.gcs)
    def test_hardtanh_2(self, X, gc, dc, engine):
        max_val = 1.6732
        min_val = 1.0507
        op = core.CreateOperator("Hardtanh", ["X"], ["Y"],
                                 min_val=min_val, max_val=max_val, engine=engine)

        X = TestHardtanh.fix0(X)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0], stepsize=1e-2, threshold=1e-2)
        self.assertReferenceChecks(
            gc, op, [X], lambda x: TestHardtanh.hardtanh_ref(x, min_val=min_val, max_val=max_val)
        )

    @given(X=hu.tensor(),
           engine=st.sampled_from(["", "CUDNN"]),
            **hu.gcs)
    def test_hardtanh_3(self, X, gc, dc, engine):
        max_val = 1.3
        min_val = 1.1
        op = core.CreateOperator("Hardtanh", ["X"], ["Y"],
                                 min_val=min_val, max_val=max_val, engine=engine)

        X = TestHardtanh.fix0(X)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0])
        self.assertReferenceChecks(
            gc, op, [X], lambda x: TestHardtanh.hardtanh_ref(x, min_val=min_val, max_val=max_val)
        )

    @given(X=hu.tensor(),
           engine=st.sampled_from(["", "CUDNN"]),
            **hu.gcs)
    def test_hardtanh_inplace(self, X, gc, dc, engine):
        max_val = 1.3
        min_val = 1.1
        op = core.CreateOperator("Hardtanh", ["X"], ["X"],
                                 min_val=min_val, max_val=max_val, engine=engine)

        X = TestHardtanh.fix0(X)
        self.assertDeviceChecks(dc, op, [X], [0])

        # inplace gradient
        Y = TestHardtanh.hardtanh_ref(X, min_val=min_val, max_val=max_val)
        dX = np.ones_like(X)
        op2 = core.CreateOperator("HardtanhGradient", ["Y", "dX"], ["dX"],
                                  min_val=min_val, max_val=max_val, engine=engine)
        self.assertDeviceChecks(dc, op2, [Y, dX], [0])

    @staticmethod
    def fix0(X):
        # go away from the origin point to avoid kink problems
        X += 0.02 * np.sign(X)
        X[X == 0.0] += 0.02
        return X

    @staticmethod
    def hardtanh_ref(x, min_val, max_val):
        ret = scale * ((x > 0) * x + (x <= 0) * (alpha * (np.exp(x) - 1)))
        return [ret]


if __name__ == "__main__":
    unittest.main()
