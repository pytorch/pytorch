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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools

import hypothesis
from hypothesis import given, strategies as st
import numpy as np

from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu


class TestAdagrad(hu.HypothesisTestCase):

    @staticmethod
    def ref_adagrad(param_in, mom_in, grad, lr, epsilon):
        mom_out = mom_in + np.square(grad)
        grad_adj = lr * grad / (np.sqrt(mom_out) + epsilon)
        param_out = param_in + grad_adj
        return (param_out, mom_out)

    @staticmethod
    def ref_row_wise_adagrad(param_in, mom_in, grad, lr, epsilon):
        mom_out = mom_in + np.mean(np.square(grad))
        grad_adj = lr * grad / (np.sqrt(mom_out) + epsilon)
        param_out = param_in + grad_adj
        return (param_out, mom_out)

    @given(inputs=hu.tensors(n=3),
           lr=st.floats(min_value=0.01, max_value=0.99,
                        allow_nan=False, allow_infinity=False),
           epsilon=st.floats(min_value=0.01, max_value=0.99,
                             allow_nan=False, allow_infinity=False),
           **hu.gcs)
    def test_adagrad(self, inputs, lr, epsilon, gc, dc):
        param, momentum, grad = inputs
        lr = np.array([lr], dtype=np.float32)

        op = core.CreateOperator(
            "Adagrad",
            ["param", "momentum", "grad", "lr"],
            ["param", "momentum"],
            epsilon=epsilon,
            device_option=gc,
        )

        self.assertReferenceChecks(
            gc, op,
            [param, momentum, grad, lr],
            functools.partial(self.ref_adagrad, epsilon=epsilon))

    @given(inputs=hu.tensors(n=3),
           lr=st.floats(min_value=0.01, max_value=0.99,
                        allow_nan=False, allow_infinity=False),
           epsilon=st.floats(min_value=0.01, max_value=0.99,
                             allow_nan=False, allow_infinity=False),
           data_strategy=st.data(),
           **hu.gcs)
    def test_sparse_adagrad(self, inputs, lr, epsilon,
                            data_strategy, gc, dc):
        param, momentum, grad = inputs
        momentum = np.abs(momentum)
        lr = np.array([lr], dtype=np.float32)

        # Create an indexing array containing values that are lists of indices,
        # which index into grad
        indices = data_strategy.draw(
            hu.tensor(dtype=np.int64,
                      elements=st.sampled_from(np.arange(grad.shape[0]))),
        )
        hypothesis.note('indices.shape: %s' % str(indices.shape))

        # For now, the indices must be unique
        hypothesis.assume(np.array_equal(np.unique(indices.flatten()),
                                         np.sort(indices.flatten())))

        # Sparsify grad
        grad = grad[indices]

        op = core.CreateOperator(
            "SparseAdagrad",
            ["param", "momentum", "indices", "grad", "lr"],
            ["param", "momentum"],
            epsilon=epsilon,
            device_option=gc)

        def ref_sparse(param, momentum, indices, grad, lr):
            param_out = np.copy(param)
            momentum_out = np.copy(momentum)
            for i, index in enumerate(indices):
                param_out[index], momentum_out[index] = self.ref_adagrad(
                    param[index], momentum[index], grad[i], lr, epsilon)
            return (param_out, momentum_out)

        self.assertReferenceChecks(
            gc, op,
            [param, momentum, indices, grad, lr],
            ref_sparse)

    @given(inputs=hu.tensors(n=2),
           lr=st.floats(min_value=0.01, max_value=0.99,
                        allow_nan=False, allow_infinity=False),
           epsilon=st.floats(min_value=0.01, max_value=0.99,
                             allow_nan=False, allow_infinity=False),
           data_strategy=st.data(),
           **hu.gcs)
    def test_sparse_adagrad_empty(self, inputs, lr, epsilon,
                                  data_strategy, gc, dc):
        param, momentum = inputs
        momentum = np.abs(momentum)
        lr = np.array([lr], dtype=np.float32)

        grad = np.empty(shape=(0,) + param.shape[1:], dtype=np.float32)
        indices = np.empty(shape=(0,), dtype=np.int64)

        hypothesis.note('indices.shape: %s' % str(indices.shape))

        op = core.CreateOperator(
            "SparseAdagrad",
            ["param", "momentum", "indices", "grad", "lr"],
            ["param", "momentum"],
            epsilon=epsilon,
            device_option=gc)

        def ref_sparse(param, momentum, indices, grad, lr):
            param_out = np.copy(param)
            momentum_out = np.copy(momentum)
            return (param_out, momentum_out)

        self.assertReferenceChecks(
            gc, op,
            [param, momentum, indices, grad, lr],
            ref_sparse)

    @given(inputs=hu.tensors(n=2),
           lr=st.floats(min_value=0.01, max_value=0.99,
                        allow_nan=False, allow_infinity=False),
           epsilon=st.floats(min_value=0.01, max_value=0.99,
                             allow_nan=False, allow_infinity=False),
           data_strategy=st.data(),
           **hu.gcs)
    def test_row_wise_sparse_adagrad(self, inputs, lr, epsilon,
                                     data_strategy, gc, dc):
        param, grad = inputs
        lr = np.array([lr], dtype=np.float32)

        # Create a 1D row-wise average sum of squared gradients tensor.
        momentum = data_strategy.draw(
            hu.tensor1d(min_len=param.shape[0], max_len=param.shape[0],
                        elements=hu.elements_of_type(dtype=np.float32))
        )
        momentum = np.abs(momentum)

        # Create an indexing array containing values which index into grad
        indices = data_strategy.draw(
            hu.tensor(dtype=np.int64,
                      elements=st.sampled_from(np.arange(grad.shape[0]))),
        )

        # Note that unlike SparseAdagrad, RowWiseSparseAdagrad uses a moment
        # tensor that is strictly 1-dimensional and equal in length to the
        # first dimension of the parameters, so indices must also be
        # 1-dimensional.
        indices = indices.flatten()

        hypothesis.note('indices.shape: %s' % str(indices.shape))

        # The indices must be unique
        hypothesis.assume(np.array_equal(np.unique(indices), np.sort(indices)))

        # Sparsify grad
        grad = grad[indices]

        op = core.CreateOperator(
            "RowWiseSparseAdagrad",
            ["param", "momentum", "indices", "grad", "lr"],
            ["param", "momentum"],
            epsilon=epsilon,
            device_option=gc)

        def ref_row_wise_sparse(param, momentum, indices, grad, lr):
            param_out = np.copy(param)
            momentum_out = np.copy(momentum)
            for i, index in enumerate(indices):
                param_out[index], momentum_out[index] = self.ref_row_wise_adagrad(
                    param[index], momentum[index], grad[i], lr, epsilon)
            return (param_out, momentum_out)

        self.assertReferenceChecks(
            gc, op,
            [param, momentum, indices, grad, lr],
            ref_row_wise_sparse)

    @given(inputs=hu.tensors(n=1),
           lr=st.floats(min_value=0.01, max_value=0.99,
                        allow_nan=False, allow_infinity=False),
           epsilon=st.floats(min_value=0.01, max_value=0.99,
                             allow_nan=False, allow_infinity=False),
           data_strategy=st.data(),
           **hu.gcs)
    def test_row_wise_sparse_adagrad_empty(self, inputs, lr, epsilon,
                                           data_strategy, gc, dc):
        param = inputs[0]
        lr = np.array([lr], dtype=np.float32)

        momentum = data_strategy.draw(
            hu.tensor1d(min_len=param.shape[0], max_len=param.shape[0],
                        elements=hu.elements_of_type(dtype=np.float32))
        )
        momentum = np.abs(momentum)

        grad = np.empty(shape=(0,) + param.shape[1:], dtype=np.float32)
        indices = np.empty(shape=(0,), dtype=np.int64)

        hypothesis.note('indices.shape: %s' % str(indices.shape))

        op = core.CreateOperator(
            "RowWiseSparseAdagrad",
            ["param", "momentum", "indices", "grad", "lr"],
            ["param", "momentum"],
            epsilon=epsilon,
            device_option=gc)

        def ref_row_wise_sparse(param, momentum, indices, grad, lr):
            param_out = np.copy(param)
            momentum_out = np.copy(momentum)
            return (param_out, momentum_out)

        self.assertReferenceChecks(
            gc, op,
            [param, momentum, indices, grad, lr],
            ref_row_wise_sparse)
