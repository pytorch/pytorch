# Copyright (c) 2016-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

from __future__ import absolute_import, division, print_function, unicode_literals

import functools

import caffe2.python.hypothesis_test_util as hu
import hypothesis
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, dyndep
from caffe2.python.operator_test.adagrad_test_helper import (
    adagrad_sparse_test_helper,
    ref_adagrad,
)
from hypothesis import HealthCheck, given, settings


dyndep.InitOpsLibrary("//caffe2/caffe2/intra_op_parallel:intra_op_parallel_ops")
# Add gflag options to use async_scheduling in nets created inside assertReferenceChecks
core.GlobalInit(
    [
        "parallel_adagrad_test",
        "--caffe2_override_executor=simple,async_scheduling",
        "--caffe2_net_async_thread_pool_size=7",
    ]
)


class TestParallelAdagrad(hu.HypothesisTestCase):
    @given(
        inputs=hu.tensors(n=3),
        engine=st.sampled_from(["INTRA_OP_PARALLEL", "TBB"]),
        lr=st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
        epsilon=st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
        **hu.gcs_cpu_only
    )
    def test_adagrad(self, inputs, engine, lr, epsilon, gc, dc):
        param, momentum, grad = inputs
        momentum = np.abs(momentum)
        lr = np.array([lr], dtype=np.float32)

        op = core.CreateOperator(
            "Adagrad",
            ["param", "momentum", "grad", "lr"],
            ["param", "momentum"],
            epsilon=epsilon,
            device_option=gc,
            engine=engine,
        )

        self.assertReferenceChecks(
            gc,
            op,
            [param, momentum, grad, lr],
            functools.partial(ref_adagrad, epsilon=epsilon),
        )

    # Suppress filter_too_much health check.
    # Likely caused by `assume` call falling through too often.
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    @given(
        inputs=hu.tensors(n=3),
        engine=st.sampled_from(["INTRA_OP_PARALLEL", "TBB"]),
        lr=st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
        epsilon=st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
        data_strategy=st.data(),
        **hu.gcs_cpu_only
    )
    def test_sparse_adagrad(self, inputs, engine, lr, epsilon, data_strategy, gc, dc):
        adagrad_sparse_test_helper(
            self, inputs, lr, epsilon, engine, ref_adagrad, gc, dc
        )

    @given(
        inputs=hu.tensors(n=2),
        engine=st.sampled_from(["INTRA_OP_PARALLEL", "TBB"]),
        lr=st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
        epsilon=st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
        data_strategy=st.data(),
        **hu.gcs_cpu_only
    )
    def test_sparse_adagrad_empty(
        self, inputs, engine, lr, epsilon, data_strategy, gc, dc
    ):
        param, momentum = inputs
        grad = np.empty(shape=(0,) + param.shape[1:], dtype=np.float32)

        adagrad_sparse_test_helper(
            self, [param, momentum, grad], lr, epsilon, engine, ref_adagrad, gc, dc
        )

    # Suppress filter_too_much health check.
    # Likely caused by `assume` call falling through too often.
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    @given(
        inputs=hu.tensors(n=3),
        engine=st.sampled_from(["INTRA_OP_PARALLEL", "TBB"]),
        lr=st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
        epsilon=st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
        data_strategy=st.data(),
        **hu.gcs
    )
    def test_row_wise_sparse_adagrad(self, inputs, engine, lr, epsilon, data_strategy, gc, dc):
        adagrad_sparse_test_helper(
            self,
            inputs,
            lr,
            epsilon,
            engine,
            functools.partial(ref_adagrad, row_wise=True),
            gc,
            dc,
            row_wise=True,
        )

    @given(
        inputs=hu.tensors(n=2),
        engine=st.sampled_from(["INTRA_OP_PARALLEL", "TBB"]),
        lr=st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
        epsilon=st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
        data_strategy=st.data(),
        **hu.gcs
    )
    def test_row_wise_sparse_adagrad_empty(
        self, inputs, engine, lr, epsilon, data_strategy, gc, dc
    ):
        param, momentum = inputs
        grad = np.empty(shape=(0,) + param.shape[1:], dtype=np.float32)
        adagrad_sparse_test_helper(
            self,
            [param, momentum, grad],
            lr,
            epsilon,
            engine,
            ref_adagrad,
            gc,
            dc,
            row_wise=True,
        )
