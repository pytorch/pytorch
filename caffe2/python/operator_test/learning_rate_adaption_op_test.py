




from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial

from hypothesis import given, settings
import hypothesis.strategies as st
import numpy as np


class TestLearningRateAdaption(serial.SerializedTestCase):
    @given(inputs=hu.tensors(n=2),
           lr=st.floats(min_value=0.01, max_value=0.99,
                        allow_nan=False, allow_infinity=False),
           lr_alpha=st.floats(min_value=0.01, max_value=0.99,
                           allow_nan=False, allow_infinity=False),
           **hu.gcs_cpu_only)
    @settings(deadline=None, max_examples=50)
    def test_learning_rate_adaption_op_normalization(self, inputs, lr, lr_alpha,
                                                     gc, dc):
        grad, effgrad = inputs
        lr = np.array([lr], dtype=np.float32)

        op = core.CreateOperator(
            'LearningRateAdaption',
            ['lr', 'grad', 'effgrad'],
            ['output_lr'],
            lr_alpha=lr_alpha)

        def ref(lr, grad, effgrad):
            flattened_grad = grad.flatten()
            flattened_effgrad = effgrad.flatten()
            x = np.dot(flattened_grad, flattened_effgrad)
            kEps = 1e-12
            y = np.linalg.norm(flattened_grad, ord=2)
            y = np.maximum(y, kEps)
            z = np.linalg.norm(flattened_effgrad, ord=2)
            z = np.maximum(z, kEps)
            output_lr = lr
            output_lr[0] -= lr[0] * lr_alpha * float(x / (y * z))
            return output_lr,

        self.assertReferenceChecks(
            gc, op,
            [lr, grad, effgrad],
            ref)

    @given(inputs=hu.tensors(n=2),
           lr=st.floats(min_value=0.01, max_value=0.99,
                        allow_nan=False, allow_infinity=False),
           lr_alpha=st.floats(min_value=0.01, max_value=0.99,
                           allow_nan=False, allow_infinity=False),
           **hu.gcs_cpu_only)
    def test_learning_rate_adaption_op_without_normalization(self, inputs, lr,
                                                             lr_alpha, gc, dc):
        grad, effgrad = inputs
        lr = np.array([lr], dtype=np.float32)

        op = core.CreateOperator(
            'LearningRateAdaption',
            ['lr', 'grad', 'effgrad'],
            ['output_lr'],
            lr_alpha=lr_alpha,
            normalized_lr_adaption=False)

        def ref(lr, grad, effgrad):
            flattened_grad = grad.flatten()
            flattened_effgrad = effgrad.flatten()
            x = np.dot(flattened_grad, flattened_effgrad)
            output_lr = lr
            output_lr[0] -= lr_alpha * x
            return output_lr,

        self.assertReferenceChecks(
            gc, op,
            [lr, grad, effgrad],
            ref)
