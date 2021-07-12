




import numpy as np
import hypothesis.strategies as st
import unittest
import caffe2.python.hypothesis_test_util as hu
from caffe2.python import core, workspace
from hypothesis import given
import caffe2.python.ideep_test_util as mu


@unittest.skipIf(not workspace.C.use_mkldnn, "No MKLDNN support.")
class TestAdamOps(hu.HypothesisTestCase):
    @given(inputs=hu.tensors(n=4),
           ITER=st.integers(min_value=0, max_value=10000),
           LR=st.floats(min_value=0.01, max_value=0.99,
                        allow_nan=False, allow_infinity=False),
           beta1=st.floats(min_value=0.01, max_value=0.99,
                           allow_nan=False, allow_infinity=False),
           beta2=st.floats(min_value=0.01, max_value=0.99,
                           allow_nan=False, allow_infinity=False),
           epsilon=st.floats(min_value=0.01, max_value=0.99,
                             allow_nan=False, allow_infinity=False),
           **mu.gcs)
    def test_adam(self, inputs, ITER, LR, beta1, beta2, epsilon, gc, dc):
        param, mom1, mom2, grad = inputs
        ITER = np.array([ITER], dtype=np.int64)
        LR = np.array([LR], dtype=np.float32)
        mom2 = np.absolute(mom2)
        op = core.CreateOperator(
            "Adam",
            ["param", "mom1", "mom2", "grad", "lr", "iter"],
            ["output_param", "output_mom1", "output_mom2"],
            beta1=beta1, beta2=beta2, epsilon=epsilon)
        # Iter lives on the CPU
        input_device_options = {'iter': hu.cpu_do, 'lr': hu.cpu_do}

        self.assertDeviceChecks(
            dc, op,
            [param, mom1, mom2, grad, LR, ITER],
            [0],
            input_device_options=input_device_options,
            threshold=0.001)

    @given(inputs=hu.tensors(n=4),
           ITER=st.integers(min_value=0, max_value=10000),
           LR=st.floats(min_value=0.01, max_value=0.99,
                        allow_nan=False, allow_infinity=False),
           beta1=st.floats(min_value=0.01, max_value=0.99,
                           allow_nan=False, allow_infinity=False),
           beta2=st.floats(min_value=0.01, max_value=0.99,
                           allow_nan=False, allow_infinity=False),
           epsilon=st.floats(min_value=0.01, max_value=0.99,
                             allow_nan=False, allow_infinity=False),
           **mu.gcs)
    def test_adam_output_grad(self, inputs, ITER, LR, beta1, beta2, epsilon, gc, dc):
        param, mom1, mom2, grad = inputs
        ITER = np.array([ITER], dtype=np.int64)
        LR = np.array([LR], dtype=np.float32)
        mom2 = np.absolute(mom2)

        op = core.CreateOperator(
            "Adam",
            ["param", "mom1", "mom2", "grad", "lr", "iter"],
            ["output_param", "output_mom1", "output_mom2", "output_grad"],
            beta1=beta1, beta2=beta2, epsilon=epsilon)

        # Iter lives on the CPU
        input_device_options = {'iter': hu.cpu_do, 'lr': hu.cpu_do}

        self.assertDeviceChecks(
            dc, op,
            [param, mom1, mom2, grad, LR, ITER],
            [0],
            input_device_options=input_device_options,
            threshold=0.001)

if __name__ == "__main__":
    unittest.main()
