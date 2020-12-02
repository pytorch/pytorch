

import functools

import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core
from caffe2.python.operator_test.adagrad_test_helper import (
    adagrad_sparse_test_helper,
    ref_adagrad,
)
from hypothesis import HealthCheck, given, settings


class TestAdagrad(serial.SerializedTestCase):
    @given(
        inputs=hu.tensors(n=3),
        lr=st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
        epsilon=st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
        weight_decay=st.sampled_from([0.0, 0.1]),
        **hu.gcs
    )
    @settings(deadline=1000)
    def test_adagrad(self, inputs, lr, epsilon, weight_decay, gc, dc):
        param, momentum, grad = inputs
        momentum = np.abs(momentum)
        lr = np.array([lr], dtype=np.float32)

        op = core.CreateOperator(
            "Adagrad",
            ["param", "momentum", "grad", "lr"],
            ["param", "momentum"],
            epsilon=epsilon,
            weight_decay=weight_decay,
            device_option=gc,
        )

        self.assertReferenceChecks(
            gc,
            op,
            [param, momentum, grad, lr],
            functools.partial(ref_adagrad, epsilon=epsilon, weight_decay=weight_decay),
        )

    @given(
        inputs=hu.tensors(n=3),
        lr=st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
        epsilon=st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
        weight_decay=st.sampled_from([0.0, 0.1]),
        **hu.gcs_cpu_only
    )
    @settings(deadline=10000)
    def test_adagrad_output_effective_lr(
        self, inputs, lr, epsilon, weight_decay, gc, dc
    ):
        param, momentum, grad = inputs
        momentum = np.abs(momentum)
        lr = np.array([lr], dtype=np.float32)

        op = core.CreateOperator(
            "Adagrad",
            ["param", "momentum", "grad", "lr"],
            ["param", "momentum", "effective_lr"],
            epsilon=epsilon,
            weight_decay=weight_decay,
            device_option=gc,
        )

        self.assertReferenceChecks(
            gc,
            op,
            [param, momentum, grad, lr],
            functools.partial(
                ref_adagrad,
                epsilon=epsilon,
                output_effective_lr=True,
                weight_decay=weight_decay,
            ),
        )

    @given(
        inputs=hu.tensors(n=3),
        lr=st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
        epsilon=st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
        **hu.gcs_cpu_only
    )
    @settings(deadline=1000)
    def test_adagrad_output_effective_lr_and_update(self, inputs, lr, epsilon, gc, dc):
        param, momentum, grad = inputs
        momentum = np.abs(momentum)
        lr = np.array([lr], dtype=np.float32)

        op = core.CreateOperator(
            "Adagrad",
            ["param", "momentum", "grad", "lr"],
            ["param", "momentum", "effective_lr", "update"],
            epsilon=epsilon,
            device_option=gc,
        )

        self.assertReferenceChecks(
            gc,
            op,
            [param, momentum, grad, lr],
            functools.partial(
                ref_adagrad, epsilon=epsilon, output_effective_lr_and_update=True
            ),
        )

    # Suppress filter_too_much health check.
    # Likely caused by `assume` call falling through too often.
    @settings(suppress_health_check=[HealthCheck.filter_too_much], deadline=10000)
    @given(
        inputs=hu.tensors(n=3),
        lr=st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
        epsilon=st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
        weight_decay=st.sampled_from([0.0, 0.1]),
        **hu.gcs
    )
    def test_sparse_adagrad(self, inputs, lr, epsilon, weight_decay, gc, dc):
        adagrad_sparse_test_helper(
            self,
            inputs,
            lr,
            epsilon,
            None,
            ref_adagrad,
            gc,
            dc,
            weight_decay=weight_decay,
        )

    @given(
        inputs=hu.tensors(n=2),
        lr=st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
        epsilon=st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
        **hu.gcs
    )
    @settings(deadline=1000)
    def test_sparse_adagrad_empty(self, inputs, lr, epsilon, gc, dc):
        param, momentum = inputs
        grad = np.empty(shape=(0,) + param.shape[1:], dtype=np.float32)

        ref_using_fp16_values = [False]
        if gc == hu.gpu_do:
            ref_using_fp16_values.append(True)

        for ref_using_fp16 in ref_using_fp16_values:
            if ref_using_fp16:
                print("test_sparse_adagrad_empty with half precision embedding")
                momentum_i = momentum.astype(np.float16)
                param_i = param.astype(np.float16)
            else:
                print("test_sparse_adagrad_empty with full precision embedding")
                momentum_i = momentum.astype(np.float32)
                param_i = param.astype(np.float32)

            adagrad_sparse_test_helper(
                self,
                [param_i, momentum_i, grad],
                lr,
                epsilon,
                None,
                ref_adagrad,
                gc,
                dc,
            )

    # Suppress filter_too_much health check.
    # Likely caused by `assume` call falling through too often.
    @settings(suppress_health_check=[HealthCheck.filter_too_much], deadline=1000)
    @given(
        inputs=hu.tensors(n=3),
        lr=st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
        epsilon=st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
        weight_decay=st.sampled_from([0.0, 0.1]),
        **hu.gcs
    )
    def test_row_wise_sparse_adagrad(self, inputs, lr, epsilon, weight_decay, gc, dc):
        adagrad_sparse_test_helper(
            self,
            inputs,
            lr,
            epsilon,
            None,
            functools.partial(ref_adagrad, row_wise=True),
            gc,
            dc,
            row_wise=True,
            weight_decay=weight_decay,
        )

    @given(
        inputs=hu.tensors(n=2),
        lr=st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
        epsilon=st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
        **hu.gcs
    )
    @settings(deadline=1000)
    def test_row_wise_sparse_adagrad_empty(self, inputs, lr, epsilon, gc, dc):
        param, momentum = inputs
        grad = np.empty(shape=(0,) + param.shape[1:], dtype=np.float32)
        adagrad_sparse_test_helper(
            self,
            [param, momentum, grad],
            lr,
            epsilon,
            None,
            ref_adagrad,
            gc,
            dc,
            row_wise=True,
        )
