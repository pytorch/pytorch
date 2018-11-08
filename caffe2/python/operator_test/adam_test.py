from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools

import hypothesis
from hypothesis import given
import hypothesis.strategies as st
import numpy as np

from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu


class TestAdam(hu.HypothesisTestCase):

    @staticmethod
    def ref_adam(param, mom1, mom2, grad, LR, ITER,
                 beta1, beta2, epsilon, output_grad=False):
        t = ITER + 1
        corrected_local_rate = np.sqrt(1 - np.power(beta2, t)) / \
            (1 - np.power(beta1, t))
        mom1_out = (beta1 * mom1) + (1 - beta1) * grad
        mom2_out = (beta2 * mom2) + (1 - beta2) * np.square(grad)
        grad_out = corrected_local_rate * mom1_out / \
            (np.sqrt(mom2_out) + epsilon)
        param_out = param + LR * grad_out
        if output_grad:
            return param_out, mom1_out, mom2_out, grad_out
        else:
            return param_out, mom1_out, mom2_out

    @staticmethod
    def ref_row_wise_adam(param, mom1, mom2, grad, LR, ITER,
                          beta1, beta2, epsilon, output_grad=False):
        t = ITER + 1
        corrected_local_rate = np.sqrt(1 - np.power(beta2, t)) / \
            (1 - np.power(beta1, t))
        mom1_out = (beta1 * mom1) + (1 - beta1) * grad
        mom2_out = (beta2 * mom2) + (1 - beta2) * np.mean(np.square(grad))
        grad_out = corrected_local_rate * mom1_out / (np.sqrt(mom2_out) + epsilon)
        param_out = param + LR * grad_out
        if output_grad:
            return param_out, mom1_out, mom2_out, grad_out
        else:
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
           **hu.gcs)
    def test_adam(self, inputs, ITER, LR, beta1, beta2, epsilon, gc, dc):
        param, mom1, mom2, grad = inputs
        ITER = np.array([ITER], dtype=np.int64)
        LR = np.array([LR], dtype=np.float32)

        op = core.CreateOperator(
            "Adam",
            ["param", "mom1", "mom2", "grad", "lr", "iter"],
            ["output_param", "output_mom1", "output_mom2"],
            beta1=beta1, beta2=beta2, epsilon=epsilon)

        # Iter lives on the CPU
        input_device_options = {'iter': hu.cpu_do}

        self.assertReferenceChecks(
            gc, op,
            [param, mom1, mom2, grad, LR, ITER],
            functools.partial(
                self.ref_adam,
                beta1=beta1, beta2=beta2, epsilon=epsilon),
            input_device_options=input_device_options)

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
           **hu.gcs_cpu_only)
    def test_adam_output_grad(self, inputs, ITER, LR, beta1, beta2, epsilon, gc, dc):
        param, mom1, mom2, grad = inputs
        ITER = np.array([ITER], dtype=np.int64)
        LR = np.array([LR], dtype=np.float32)

        op = core.CreateOperator(
            "Adam",
            ["param", "mom1", "mom2", "grad", "lr", "iter"],
            ["output_param", "output_mom1", "output_mom2", "output_grad"],
            beta1=beta1, beta2=beta2, epsilon=epsilon)

        # Iter lives on the CPU
        input_device_options = {'iter': hu.cpu_do}

        self.assertReferenceChecks(
            gc, op,
            [param, mom1, mom2, grad, LR, ITER],
            functools.partial(
                self.ref_adam,
                beta1=beta1, beta2=beta2, epsilon=epsilon, output_grad=True),
            input_device_options=input_device_options)

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
           data_strategy=st.data(),
           **hu.gcs)
    def test_sparse_adam(self, inputs, ITER, LR, beta1, beta2, epsilon,
                         data_strategy, gc, dc):
        param, mom1, mom2, grad = inputs
        mom2 = np.absolute(mom2)
        ITER = np.array([ITER], dtype=np.int64)
        LR = np.array([LR], dtype=np.float32)

        # Create an indexing array containing values which index into grad
        indices = data_strategy.draw(
            hu.tensor(
                max_dim=1,
                min_value=1,
                max_value=grad.shape[0],
                dtype=np.int64,
                elements=st.sampled_from(np.arange(grad.shape[0])),
            ),
        )

        # Verify that the generated indices are unique
        hypothesis.assume(
            np.array_equal(
                np.unique(indices.flatten()),
                np.sort(indices.flatten())))

        # Sparsify grad
        grad = grad[indices]

        op = core.CreateOperator(
            "SparseAdam",
            ["param", "mom1", "mom2", "indices", "grad", "lr", "iter"],
            ["param", "mom1", "mom2"],
            beta1=beta1, beta2=beta2, epsilon=epsilon)

        def ref_sparse(param, mom1, mom2, indices, grad, LR, ITER):
            param_out = np.copy(param)
            mom1_out = np.copy(mom1)
            mom2_out = np.copy(mom2)

            for i, index in enumerate(indices):
                param_out[index], mom1_out[index], mom2_out[index] = \
                    self.ref_adam(param[index], mom1[index], mom2[index],
                                  grad[i], LR, ITER,
                                  beta1, beta2, epsilon)
            return (param_out, mom1_out, mom2_out)

        # Iter lives on the CPU
        input_device_options = {'iter': hu.cpu_do}

        self.assertReferenceChecks(
            gc, op,
            [param, mom1, mom2, indices, grad, LR, ITER],
            ref_sparse,
            input_device_options=input_device_options)

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
           data_strategy=st.data(),
           **hu.gcs)
    def test_sparse_adam_output_grad(self, inputs, ITER, LR, beta1, beta2, epsilon,
                         data_strategy, gc, dc):
        param, mom1, mom2, grad = inputs
        mom2 = np.absolute(mom2)
        ITER = np.array([ITER], dtype=np.int64)
        LR = np.array([LR], dtype=np.float32)

        # Create an indexing array containing values which index into grad
        indices = data_strategy.draw(
            hu.tensor(
                max_dim=1,
                min_value=1,
                max_value=grad.shape[0],
                dtype=np.int64,
                elements=st.sampled_from(np.arange(grad.shape[0])),
            ),
        )

        # Verify that the generated indices are unique
        hypothesis.assume(
            np.array_equal(
                np.unique(indices.flatten()),
                np.sort(indices.flatten())))

        # Sparsify grad
        grad = grad[indices]

        op = core.CreateOperator(
            "SparseAdam",
            ["param", "mom1", "mom2", "indices", "grad", "lr", "iter"],
            ["param", "mom1", "mom2", "output_grad"],
            beta1=beta1, beta2=beta2, epsilon=epsilon)

        def ref_sparse_output_grad(param, mom1, mom2, indices, grad, LR, ITER,
                                beta1, beta2, epsilon, output_grad):
            param_out = np.copy(param)
            mom1_out = np.copy(mom1)
            mom2_out = np.copy(mom2)
            grad_out = np.copy(grad)

            for i, index in enumerate(indices):
                param_out[index], mom1_out[index], mom2_out[index], grad_out[i] = \
                    self.ref_adam(param[index], mom1[index], mom2[index],
                                  grad[i], LR, ITER,
                                  beta1, beta2, epsilon, output_grad)
            return (param_out, mom1_out, mom2_out, grad_out)

        # Iter lives on the CPU
        input_device_options = {'iter': hu.cpu_do}

        self.assertReferenceChecks(
            gc, op,
            [param, mom1, mom2, indices, grad, LR, ITER],
            functools.partial(
                ref_sparse_output_grad,
                beta1=beta1, beta2=beta2, epsilon=epsilon, output_grad=True),
            input_device_options=input_device_options)

    @given(inputs=hu.tensors(n=3),
           ITER=st.integers(min_value=0, max_value=10000),
           LR=st.floats(min_value=0.01, max_value=0.99,
                        allow_nan=False, allow_infinity=False),
           beta1=st.floats(min_value=0.01, max_value=0.99,
                           allow_nan=False, allow_infinity=False),
           beta2=st.floats(min_value=0.01, max_value=0.99,
                           allow_nan=False, allow_infinity=False),
           epsilon=st.floats(min_value=0.01, max_value=0.99,
                             allow_nan=False, allow_infinity=False),
           data_strategy=st.data(),
               **hu.gcs_cpu_only)
    def test_row_wise_sparse_adam(self, inputs, ITER, LR, beta1, beta2, epsilon,
                                  data_strategy, gc, dc):
        param, mom1, grad = inputs
        ITER = np.array([ITER], dtype=np.int64)
        LR = np.array([LR], dtype=np.float32)

        # Create a 1D row-wise average 2nd moment tensor.
        mom2 = data_strategy.draw(
            hu.tensor1d(min_len=param.shape[0], max_len=param.shape[0],
                        elements=hu.elements_of_type(dtype=np.float32))
        )
        mom2 = np.absolute(mom2)

        # Create an indexing array containing values which index into grad
        indices = data_strategy.draw(
            hu.tensor(
                max_dim=1,
                min_value=1,
                max_value=grad.shape[0],
                dtype=np.int64,
                elements=st.sampled_from(np.arange(grad.shape[0])),
            ),
        )

        # Note that unlike SparseAdam, RowWiseSparseAdam uses a moment
        # tensor that is strictly 1-dimensional and equal in length to the
        # first dimension of the parameters, so indices must also be
        # 1-dimensional.
        indices = indices.flatten()

        hypothesis.note('indices.shape: %s' % str(indices.shape))

        # Verify that the generated indices are unique
        hypothesis.assume(np.array_equal(np.unique(indices), np.sort(indices)))

        # Sparsify grad
        grad = grad[indices]

        op = core.CreateOperator(
            "RowWiseSparseAdam",
            ["param", "mom1", "mom2", "indices", "grad", "lr", "iter"],
            ["param", "mom1", "mom2"],
            beta1=beta1, beta2=beta2, epsilon=epsilon)

        def ref_row_wise_sparse(param, mom1, mom2, indices, grad, LR, ITER):
            param_out = np.copy(param)
            mom1_out = np.copy(mom1)
            mom2_out = np.copy(mom2)
            for i, index in enumerate(indices):
                param_out[index], mom1_out[index], mom2_out[index] = \
                    self.ref_row_wise_adam(param[index], mom1[index], mom2[index],
                                           grad[i], LR, ITER,
                                           beta1, beta2, epsilon)
            return (param_out, mom1_out, mom2_out)

        # Iter lives on the CPU
        input_device_options = {'iter': hu.cpu_do}

        self.assertReferenceChecks(
            gc, op,
            [param, mom1, mom2, indices, grad, LR, ITER],
            ref_row_wise_sparse,
            input_device_options=input_device_options)

    @given(inputs=hu.tensors(n=3),
           ITER=st.integers(min_value=0, max_value=10000),
           LR=st.floats(min_value=0.01, max_value=0.99,
                        allow_nan=False, allow_infinity=False),
           beta1=st.floats(min_value=0.01, max_value=0.99,
                           allow_nan=False, allow_infinity=False),
           beta2=st.floats(min_value=0.01, max_value=0.99,
                           allow_nan=False, allow_infinity=False),
           epsilon=st.floats(min_value=0.01, max_value=0.99,
                             allow_nan=False, allow_infinity=False),
           data_strategy=st.data(),
               **hu.gcs_cpu_only)
    def test_row_wise_sparse_adam_output_grad(self, inputs, ITER, LR, beta1, beta2,
                                  epsilon, data_strategy, gc, dc):
        param, mom1, grad = inputs
        ITER = np.array([ITER], dtype=np.int64)
        LR = np.array([LR], dtype=np.float32)

        # Create a 1D row-wise average 2nd moment tensor.
        mom2 = data_strategy.draw(
            hu.tensor1d(min_len=param.shape[0], max_len=param.shape[0],
                        elements=hu.elements_of_type(dtype=np.float32))
        )
        mom2 = np.absolute(mom2)

        # Create an indexing array containing values which index into grad
        indices = data_strategy.draw(
            hu.tensor(
                max_dim=1,
                min_value=1,
                max_value=grad.shape[0],
                dtype=np.int64,
                elements=st.sampled_from(np.arange(grad.shape[0])),
            ),
        )

        # Note that unlike SparseAdam, RowWiseSparseAdam uses a moment
        # tensor that is strictly 1-dimensional and equal in length to the
        # first dimension of the parameters, so indices must also be
        # 1-dimensional.
        indices = indices.flatten()

        hypothesis.note('indices.shape: %s' % str(indices.shape))

        # Verify that the generated indices are unique
        hypothesis.assume(np.array_equal(np.unique(indices), np.sort(indices)))

        # Sparsify grad
        grad = grad[indices]

        op = core.CreateOperator(
            "RowWiseSparseAdam",
            ["param", "mom1", "mom2", "indices", "grad", "lr", "iter"],
            ["param", "mom1", "mom2", "output_grad"],
            beta1=beta1, beta2=beta2, epsilon=epsilon)

        def ref_row_wise_sparse_output_grad(param, mom1, mom2, indices, grad, LR, ITER,
                                        beta1, beta2, epsilon, output_grad):
            param_out = np.copy(param)
            mom1_out = np.copy(mom1)
            mom2_out = np.copy(mom2)
            grad_out = np.copy(grad)

            for i, index in enumerate(indices):
                param_out[index], mom1_out[index], mom2_out[index], grad_out[i] = \
                    self.ref_row_wise_adam(param[index], mom1[index], mom2[index],
                                           grad[i], LR, ITER,
                                           beta1, beta2, epsilon, output_grad)
            return (param_out, mom1_out, mom2_out, grad_out)

        # Iter lives on the CPU
        input_device_options = {'iter': hu.cpu_do}

        self.assertReferenceChecks(
            gc, op,
            [param, mom1, mom2, indices, grad, LR, ITER],
            functools.partial(
                ref_row_wise_sparse_output_grad,
                beta1=beta1, beta2=beta2, epsilon=epsilon, output_grad=True),
            input_device_options=input_device_options)


if __name__ == "__main__":
    import unittest
    unittest.main()
