




from caffe2.python import core
from hypothesis import given, settings
import hypothesis.strategies as st
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import numpy as np

import unittest


class TestThresholdedRelu(serial.SerializedTestCase):

    # test case 1 - default alpha - we do reference and dc checks.
    # test case 2 does dc and reference checks over range of alphas.
    # test case 3 does gc over range of alphas.
    @serial.given(input=hu.tensor(),
           engine=st.sampled_from(["", "CUDNN"]),
           **hu.gcs)
    def test_thresholded_relu_1(self, input, gc, dc, engine):
        X = input
        op = core.CreateOperator("ThresholdedRelu", ["X"], ["Y"],
                                 engine=engine)

        def defaultRef(X):
            Y = np.copy(X)
            Y[Y <= 1.0] = 0.0
            return (Y,)

        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertReferenceChecks(gc, op, [X], defaultRef)

    @given(input=hu.tensor(),
           alpha=st.floats(min_value=1.0, max_value=5.0),
           engine=st.sampled_from(["", "CUDNN"]),
           **hu.gcs)
    def test_thresholded_relu_2(self, input, alpha, gc, dc, engine):
        X = input
        op = core.CreateOperator("ThresholdedRelu", ["X"], ["Y"],
                                 alpha=alpha, engine=engine)

        def ref(X):
            Y = np.copy(X)
            Y[Y <= alpha] = 0.0
            return (Y,)

        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertReferenceChecks(gc, op, [X], ref)

    @given(input=hu.tensor(),
           alpha=st.floats(min_value=1.1, max_value=5.0),
           engine=st.sampled_from(["", "CUDNN"]),
           **hu.gcs)
    @settings(deadline=10000)
    def test_thresholded_relu_3(self, input, alpha, gc, dc, engine):
        X = TestThresholdedRelu.fix_input(input)
        op = core.CreateOperator("ThresholdedRelu", ["X"], ["Y"],
                                 alpha=float(alpha), engine=engine)
        self.assertGradientChecks(gc, op, [X], 0, [0])

    @staticmethod
    def fix_input(input):
        # go away from alpha to avoid derivative discontinuities
        input += 0.02 * np.sign(input)
        return input


if __name__ == "__main__":
    unittest.main()
