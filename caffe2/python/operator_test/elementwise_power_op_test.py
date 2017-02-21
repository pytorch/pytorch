from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given
from hypothesis import strategies as st
import caffe2.python.hypothesis_test_util as hu

import unittest


class TestElementwisePowerOp(hu.HypothesisTestCase):

    @given(X=hu.tensor(),
           exponent=st.floats(min_value=-1.0, max_value=1.0),
           **hu.gcs_cpu_only)
    def test_elementwise_power(self, X, exponent, gc, dc):
        def elementwise_power(X):
            return (X ** exponent,)

        op = core.CreateOperator(
            "ElementwisePower", ["X"], ["Y"], exponent=exponent)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertReferenceChecks(gc, op, [X], elementwise_power)


if __name__ == "__main__":
    unittest.main()
