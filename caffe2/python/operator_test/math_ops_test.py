from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given
from hypothesis import strategies as st
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial

import numpy as np
import unittest


class TestMathOps(serial.SerializedTestCase):

    @given(X=hu.tensor(),
           exponent=st.floats(min_value=2.0, max_value=3.0),
           **hu.gcs)
    def test_elementwise_power(self, X, exponent, gc, dc):
        # negative integer raised with non-integer exponent is domain error
        X = np.abs(X)
        def powf(X):
            return (X ** exponent,)

        def powf_grad(g_out, outputs, fwd_inputs):
            return (exponent * (fwd_inputs[0] ** (exponent - 1)) * g_out,)

        op = core.CreateOperator(
            "Pow", ["X"], ["Y"], exponent=exponent)

        self.assertReferenceChecks(gc, op, [X], powf,
                                   output_to_grad="Y",
                                   grad_reference=powf_grad,
                                   ensure_outputs_are_inferred=True)

    @serial.given(X=hu.tensor(),
           exponent=st.floats(min_value=-3.0, max_value=3.0),
           **hu.gcs)
    def test_sign(self, X, exponent, gc, dc):
        def signf(X):
            return [np.sign(X)]

        op = core.CreateOperator(
            "Sign", ["X"], ["Y"])

        self.assertReferenceChecks(
            gc, op, [X], signf, ensure_outputs_are_inferred=True)
        self.assertDeviceChecks(dc, op, [X], [0])


if __name__ == "__main__":
    unittest.main()
