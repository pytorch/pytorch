




from caffe2.python import core
from hypothesis import given, settings
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np

import unittest


class TestSelu(serial.SerializedTestCase):

    @serial.given(X=hu.tensor(),
           engine=st.sampled_from(["", "CUDNN"]),
            **hu.gcs)
    def test_selu_1(self, X, gc, dc, engine):
        alpha = 1.0
        scale = 2.0
        op = core.CreateOperator("Selu", ["X"], ["Y"],
                                 alpha=alpha, scale=scale, engine=engine)
        X = TestSelu.fix0(X)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0])
        self.assertReferenceChecks(
            gc, op, [X], lambda x: TestSelu.selu_ref(x, alpha=alpha, scale=scale)
        )

    @given(X=hu.tensor(),
           engine=st.sampled_from(["", "CUDNN"]),
            **hu.gcs)
    @settings(deadline=10000)
    def test_selu_2(self, X, gc, dc, engine):
        alpha = 1.6732
        scale = 1.0507
        op = core.CreateOperator("Selu", ["X"], ["Y"],
                                 alpha=alpha, scale=scale, engine=engine)

        X = TestSelu.fix0(X)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0], stepsize=1e-2, threshold=1e-2)
        self.assertReferenceChecks(
            gc, op, [X], lambda x: TestSelu.selu_ref(x, alpha=alpha, scale=scale)
        )

    @given(X=hu.tensor(),
           engine=st.sampled_from(["", "CUDNN"]),
            **hu.gcs)
    @settings(deadline=10000)
    def test_selu_3(self, X, gc, dc, engine):
        alpha = 1.3
        scale = 1.1
        op = core.CreateOperator("Selu", ["X"], ["Y"],
                                 alpha=alpha, scale=scale, engine=engine)

        X = TestSelu.fix0(X)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0])
        self.assertReferenceChecks(
            gc, op, [X], lambda x: TestSelu.selu_ref(x, alpha=alpha, scale=scale)
        )

    @given(X=hu.tensor(),
           engine=st.sampled_from(["", "CUDNN"]),
            **hu.gcs)
    def test_selu_inplace(self, X, gc, dc, engine):
        alpha = 1.3
        scale = 1.1
        op = core.CreateOperator("Selu", ["X"], ["X"],
                                 alpha=alpha, scale=scale, engine=engine)

        X = TestSelu.fix0(X)
        self.assertDeviceChecks(dc, op, [X], [0])

        # inplace gradient
        Y = TestSelu.selu_ref(X, alpha=alpha, scale=scale)
        dX = np.ones_like(X)
        op2 = core.CreateOperator("SeluGradient", ["Y", "dX"], ["dX"],
                                  alpha=alpha, scale=scale, engine=engine)
        self.assertDeviceChecks(dc, op2, [Y, dX], [0])

    @staticmethod
    def fix0(X):
        # go away from the origin point to avoid kink problems
        X += 0.02 * np.sign(X)
        X[X == 0.0] += 0.02
        return X

    @staticmethod
    def selu_ref(x, scale, alpha):
        ret = scale * ((x > 0) * x + (x <= 0) * (alpha * (np.exp(x) - 1)))
        return [ret]


if __name__ == "__main__":
    unittest.main()
