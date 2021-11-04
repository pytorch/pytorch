import functools

from hypothesis import given
import hypothesis.strategies as st
import numpy as np

from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu


class TestDecayAdagrad(hu.HypothesisTestCase):

    @staticmethod
    def ref_decay_adagrad(param, mom1, mom2, grad, LR, ITER,
                 beta1, beta2, epsilon, weight_decay, bias_correction_first, output_grad=False):
        t = ITER + 1
        mom1_out = (beta1 * mom1) + (1 - beta1) * grad
        mom2_out = mom2 + np.square(grad)
        if bias_correction_first:
            c = 1 - np.power(beta1, t)
        else:
            c = 1.0
        grad_out = mom1_out / c / (np.sqrt(mom2_out) + epsilon) + weight_decay * param
        param_out = param + LR * grad_out

        return param_out, mom1_out, mom2_out

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
           weight_decay=st.floats(min_value=0.01, max_value=0.99,
                             allow_nan=False, allow_infinity=False),
           **hu.gcs_cpu_only)
    def test_decay_adagrad(self, inputs, ITER, LR, beta1, beta2, epsilon, weight_decay, gc, dc):
        bias_correction_first = True

        param, mom1, mom2, grad = inputs
        mom2 = np.abs(mom2)
        ITER = np.array([ITER], dtype=np.int64)
        LR = np.array([LR], dtype=np.float32)

        op = core.CreateOperator(
            "DecayAdagrad",
            ["param", "mom1", "mom2", "grad", "lr", "iter"],
            ["output_param", "output_mom1", "output_mom2"],
            beta1=beta1, beta2=beta2, epsilon=epsilon, weight_decay=weight_decay, bias_correction_first=bias_correction_first)

        # Iter lives on the CPU
        input_device_options = {'iter': hu.cpu_do}

        self.assertReferenceChecks(
            gc, op,
            [param, mom1, mom2, grad, LR, ITER],
            functools.partial(
                self.ref_decay_adagrad,
                beta1=beta1, beta2=beta2, epsilon=epsilon, weight_decay=weight_decay, bias_correction_first=bias_correction_first),
            input_device_options=input_device_options)

if __name__ == "__main__":
    import unittest
    unittest.main()
