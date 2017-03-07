from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given
from hypothesis import strategies as st
import caffe2.python.hypothesis_test_util as hu

import unittest


class TestPowOp(hu.HypothesisTestCase):

    @given(X=hu.tensor(),
           exponent=st.floats(min_value=2.0, max_value=3.0),
           **hu.gcs)
    def test_elementwise_power(self, X, exponent, gc, dc):
        def powf(X):
            return (X ** exponent,)

        def powf_grad(g_out, outputs, fwd_inputs):
            return (exponent * (fwd_inputs[0] ** (exponent - 1)) * g_out,)

        op = core.CreateOperator(
            "Pow", ["X"], ["Y"], exponent=exponent)

        self.assertReferenceChecks(gc, op, [X], powf,
                                   output_to_grad="Y",
                                   grad_reference=powf_grad),


if __name__ == "__main__":
    unittest.main()
