




from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np


class TestElementwiseLinearOp(serial.SerializedTestCase):

    @serial.given(n=st.integers(2, 100), d=st.integers(2, 10), **hu.gcs)
    # @given(n=st.integers(2, 50), d=st.integers(2, 50), **hu.gcs_cpu_only)
    def test(self, n, d, gc, dc):
        X = np.random.rand(n, d).astype(np.float32)
        a = np.random.rand(d).astype(np.float32)
        b = np.random.rand(d).astype(np.float32)

        def ref_op(X, a, b):
            d = a.shape[0]
            return [np.multiply(X, a.reshape(1, d)) + b.reshape(1, d)]

        op = core.CreateOperator(
            "ElementwiseLinear",
            ["X", "a", "b"],
            ["Y"]
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X, a, b],
            reference=ref_op,
        )

        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [X, a, b], [0])
        # Gradient check wrt X
        self.assertGradientChecks(gc, op, [X, a, b], 0, [0])
        # Gradient check wrt a
        self.assertGradientChecks(gc, op, [X, a, b], 1, [0])
        # # Gradient check wrt b
        self.assertGradientChecks(gc, op, [X, a, b], 2, [0])
