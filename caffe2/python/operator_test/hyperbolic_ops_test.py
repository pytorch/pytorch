




from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np


class TestHyperbolicOps(serial.SerializedTestCase):
    def _test_hyperbolic_op(self, op_name, np_ref, X, in_place, engine, gc, dc):
        op = core.CreateOperator(
            op_name,
            ["X"],
            ["X"] if in_place else ["Y"],
            engine=engine,)

        def ref(X):
            return [np_ref(X)]

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=ref,
            ensure_outputs_are_inferred=True,
        )
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0], ensure_outputs_are_inferred=True)

    @serial.given(X=hu.tensor(dtype=np.float32), **hu.gcs)
    def test_sinh(self, X, gc, dc):
        self._test_hyperbolic_op("Sinh", np.sinh, X, False, "", gc, dc)

    @serial.given(X=hu.tensor(dtype=np.float32), **hu.gcs)
    def test_cosh(self, X, gc, dc):
        self._test_hyperbolic_op("Cosh", np.cosh, X, False, "", gc, dc)

    @serial.given(X=hu.tensor(dtype=np.float32), in_place=st.booleans(),
           engine=st.sampled_from(["", "CUDNN"]), **hu.gcs)
    def test_tanh(self, X, in_place, engine, gc, dc):
        self._test_hyperbolic_op("Tanh", np.tanh, X, in_place, engine, gc, dc)
