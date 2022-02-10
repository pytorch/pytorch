




import functools

import logging

import hypothesis
from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st
import numpy as np

from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial

logger = logging.getLogger(__name__)

def ref_wngrad(param_in, seq_b_in, grad, lr, epsilon,
                output_effective_lr=False,
                output_effective_lr_and_update=False):
    # helper functions for wngrad operator test
    seq_b_out = seq_b_in + 1.0 / (seq_b_in + epsilon) * np.sum(grad * grad)
    effective_lr = lr / (seq_b_in + epsilon)
    grad_adj = effective_lr * grad
    param_out = param_in + grad_adj
    if output_effective_lr_and_update:
        return (param_out.astype(np.float32), seq_b_out.astype(np.float32),
                effective_lr.astype(np.float32),
                grad_adj.astype(np.float32))
    elif output_effective_lr:
        return (param_out.astype(np.float32), seq_b_out.astype(np.float32),
                effective_lr.astype(np.float32))
    return (param_out.astype(np.float32), seq_b_out.astype(np.float32))


def wngrad_sparse_test_helper(parent_test, inputs, seq_b, lr, epsilon,
     engine, gc, dc):
    # helper functions for wngrad operator test
    param, grad = inputs
    seq_b = np.array([seq_b, ], dtype=np.float32)
    lr = np.array([lr], dtype=np.float32)

    # Create an indexing array containing values that are lists of indices,
    # which index into grad
    indices = np.random.choice(np.arange(grad.shape[0]),
        size=np.random.randint(grad.shape[0]), replace=False)

    # Sparsify grad
    grad = grad[indices]

    op = core.CreateOperator(
        "SparseWngrad",
        ["param", "seq_b", "indices", "grad", "lr"],
        ["param", "seq_b"],
        epsilon=epsilon,
        engine=engine,
        device_option=gc)

    def ref_sparse(param, seq_b, indices, grad, lr):
        param_out = np.copy(param)
        seq_b_out = np.copy(seq_b)
        seq_b_out = seq_b + 1.0 / seq_b * np.sum(grad * grad)
        for i, index in enumerate(indices):
            param_out[index] = param[index] + lr / (seq_b + epsilon) * grad[i]
        return (param_out, seq_b_out)

    logger.info('test_sparse_adagrad with full precision embedding')
    seq_b_i = seq_b.astype(np.float32)
    param_i = param.astype(np.float32)

    parent_test.assertReferenceChecks(
        gc, op, [param_i, seq_b_i, indices, grad, lr],
        ref_sparse
    )


class TestWngrad(serial.SerializedTestCase):
    @given(inputs=hu.tensors(n=2),
           seq_b=st.floats(min_value=0.01, max_value=0.99,
                        allow_nan=False, allow_infinity=False),
           lr=st.floats(min_value=0.01, max_value=0.99,
                        allow_nan=False, allow_infinity=False),
           epsilon=st.floats(min_value=0.01, max_value=0.99,
                             allow_nan=False, allow_infinity=False),
           **hu.gcs_cpu_only)
    @settings(deadline=10000)
    def test_wngrad_dense_base(self, inputs, seq_b, lr, epsilon, gc, dc):
        param, grad = inputs
        seq_b = np.array([seq_b, ], dtype=np.float32)
        lr = np.array([lr], dtype=np.float32)

        op = core.CreateOperator(
            "Wngrad",
            ["param", "seq_b", "grad", "lr"],
            ["param", "seq_b"],
            epsilon=epsilon,
            device_option=gc,
        )

        self.assertReferenceChecks(
            gc, op,
            [param, seq_b, grad, lr],
            functools.partial(ref_wngrad, epsilon=epsilon))

    @given(inputs=hu.tensors(n=2),
           seq_b=st.floats(min_value=0.01, max_value=0.99,
                        allow_nan=False, allow_infinity=False),
           lr=st.floats(min_value=0.01, max_value=0.99,
                        allow_nan=False, allow_infinity=False),
           epsilon=st.floats(min_value=0.01, max_value=0.99,
                             allow_nan=False, allow_infinity=False),
           **hu.gcs_cpu_only)
    @settings(deadline=10000)
    def test_wngrad_dense_output_effective_lr(self, inputs, seq_b,
                                              lr, epsilon, gc, dc):
        param, grad = inputs
        seq_b = np.array([seq_b, ], dtype=np.float32)
        lr = np.array([lr], dtype=np.float32)

        op = core.CreateOperator(
            "Wngrad",
            ["param", "seq_b", "grad", "lr"],
            ["param", "seq_b", "effective_lr"],
            epsilon=epsilon,
            device_option=gc,
        )

        self.assertReferenceChecks(
            gc, op,
            [param, seq_b, grad, lr],
            functools.partial(ref_wngrad, epsilon=epsilon,
                              output_effective_lr=True))

    @given(inputs=hu.tensors(n=2),
           seq_b=st.floats(min_value=0.01, max_value=0.99,
                        allow_nan=False, allow_infinity=False),
           lr=st.floats(min_value=0.01, max_value=0.99,
                        allow_nan=False, allow_infinity=False),
           epsilon=st.floats(min_value=0.01, max_value=0.99,
                             allow_nan=False, allow_infinity=False),
           **hu.gcs_cpu_only)
    @settings(deadline=10000)
    def test_wngrad_dense_output_effective_lr_and_update(
            self, inputs, seq_b, lr, epsilon, gc, dc):
        param, grad = inputs
        seq_b = np.abs(np.array([seq_b, ], dtype=np.float32))
        lr = np.array([lr], dtype=np.float32)

        op = core.CreateOperator(
            "Wngrad",
            ["param", "seq_b", "grad", "lr"],
            ["param", "seq_b", "effective_lr", "update"],
            epsilon=epsilon,
            device_option=gc,
        )

        self.assertReferenceChecks(
            gc, op,
            [param, seq_b, grad, lr],
            functools.partial(ref_wngrad, epsilon=epsilon,
                              output_effective_lr_and_update=True))

    # Suppress filter_too_much health check.
    # Likely caused by `assume` call falling through too often.
    @settings(suppress_health_check=[HealthCheck.filter_too_much], deadline=10000)
    @given(inputs=hu.tensors(n=2),
           seq_b=st.floats(min_value=0.01, max_value=0.99,
                        allow_nan=False, allow_infinity=False),
           lr=st.floats(min_value=0.01, max_value=0.99,
                        allow_nan=False, allow_infinity=False),
           epsilon=st.floats(min_value=0.01, max_value=0.99,
                             allow_nan=False, allow_infinity=False),
           **hu.gcs_cpu_only)
    def test_sparse_wngrad(self, inputs, seq_b, lr, epsilon, gc, dc):
        return wngrad_sparse_test_helper(self, inputs, seq_b, lr, epsilon,
            None, gc, dc)

    @given(inputs=hu.tensors(n=1),
           lr=st.floats(min_value=0.01, max_value=0.99,
                        allow_nan=False, allow_infinity=False),
           seq_b=st.floats(min_value=0.01, max_value=0.99,
                        allow_nan=False, allow_infinity=False),
           epsilon=st.floats(min_value=0.01, max_value=0.99,
                             allow_nan=False, allow_infinity=False),
           **hu.gcs_cpu_only)
    @settings(deadline=10000)
    def test_sparse_wngrad_empty(self, inputs, seq_b, lr, epsilon, gc, dc):
        param = inputs[0]
        seq_b = np.array([seq_b, ], dtype=np.float32)
        lr = np.array([lr], dtype=np.float32)

        grad = np.empty(shape=(0,) + param.shape[1:], dtype=np.float32)
        indices = np.empty(shape=(0,), dtype=np.int64)

        hypothesis.note('indices.shape: %s' % str(indices.shape))

        op = core.CreateOperator(
            "SparseWngrad",
            ["param", "seq_b", "indices", "grad", "lr"],
            ["param", "seq_b"],
            epsilon=epsilon,
            device_option=gc)

        def ref_sparse(param, seq_b, indices, grad, lr):
            param_out = np.copy(param)
            seq_b_out = np.copy(seq_b)
            return (param_out, seq_b_out)

        print('test_sparse_adagrad_empty with full precision embedding')
        seq_b_i = seq_b.astype(np.float32)
        param_i = param.astype(np.float32)

        self.assertReferenceChecks(
            gc, op, [param_i, seq_b_i, indices, grad, lr], ref_sparse
        )
