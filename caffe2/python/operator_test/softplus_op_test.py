




from caffe2.python import core
from hypothesis import given, settings
import caffe2.python.hypothesis_test_util as hu

import unittest


class TestSoftplus(hu.HypothesisTestCase):

    @given(X=hu.tensor(),
           **hu.gcs)
    @settings(deadline=1000)
    def test_softplus(self, X, gc, dc):
        op = core.CreateOperator("Softplus", ["X"], ["Y"])
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0])


if __name__ == "__main__":
    unittest.main()
