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

import numpy as np

from hypothesis import assume, given, settings
import hypothesis.strategies as st

from caffe2.proto import caffe2_pb2
from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu


class TestMatMul(hu.HypothesisTestCase):
    @given(
        M=st.integers(min_value=1, max_value=10),
        K=st.integers(min_value=1, max_value=10),
        N=st.integers(min_value=1, max_value=10),
        trans_a=st.booleans(),
        trans_b=st.booleans(),
        **hu.gcs
    )
    def test_matmul(self, M, K, N, trans_a, trans_b, gc, dc):
        X = np.random.rand(M, K).astype(np.float32) - 0.5
        if trans_a:
            X = X.transpose()

        Y = np.random.rand(K, N).astype(np.float32) - 0.5
        if trans_b:
            Y = Y.transpose()

        op = core.CreateOperator(
            'MatMul', ['X', 'Y'], 'out', trans_a=trans_a, trans_b=trans_b
        )

        def matmul_ref(X, Y, trans_a, trans_b):
            XX = X.transpose() if trans_a else X
            YY = Y.transpose() if trans_b else Y
            return (XX.dot(YY), )

        # Check against numpy reference
        self.assertReferenceChecks(gc, op, [X, Y, trans_a, trans_b], matmul_ref)
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [X, Y], [0])
        # Gradient check wrt X
        self.assertGradientChecks(gc, op, [X, Y], 0, [0])
        # Gradient check wrt Y
        self.assertGradientChecks(gc, op, [X, Y], 1, [0])

    @given(
        M=st.integers(min_value=1, max_value=10),
        K=st.integers(min_value=1, max_value=10),
        N=st.integers(min_value=1, max_value=10),
        axis_a=st.sampled_from([-3, -2, -1, 1, 2, 3]),
        axis_b=st.sampled_from([-3, -2, -1, 1, 2, 3]),
        trans_a=st.booleans(),
        trans_b=st.booleans(),
        **hu.gcs
    )
    def test_matmul_axis(
        self, M, K, N, axis_a, axis_b, trans_a, trans_b, gc, dc
    ):
        X = np.random.rand(M, K).astype(np.float32) - 0.5
        if trans_a:
            X = X.transpose()
        shape_x = [X.shape[0], 1, 1, 1]
        shape_x[axis_a] = X.shape[1]
        X = X.reshape(*shape_x)

        Y = np.random.rand(K, N).astype(np.float32) - 0.5
        if trans_b:
            Y = Y.transpose()
        shape_y = [Y.shape[0], 1, 1, 1]
        shape_y[axis_b] = Y.shape[1]
        Y = Y.reshape(*shape_y)
        op = core.CreateOperator(
            'MatMul', ['X', 'Y'],
            'out',
            axis_a=axis_a,
            axis_b=axis_b,
            trans_a=trans_a,
            trans_b=trans_b
        )

        def size_to_dim(X, axis):
            dim = 1
            for i in range(axis):
                dim *= X.shape[i]
            return dim

        def size_from_dim(X, axis):
            dim = 1
            for i in range(axis, X.ndim):
                dim *= X.shape[i]
            return dim

        def reshape(X, axis):
            dim_0, dim_1 = size_to_dim(X, axis), size_from_dim(X, axis)
            return X.reshape(dim_0, dim_1)

        def canonical_axis(axis, ndim):
            return ndim + axis if axis < 0 else axis

        def matmul_ref(X, Y, axis_a, axis_b, trans_a, trans_b):
            can_axis_a = canonical_axis(axis_a, X.ndim)
            can_axis_b = canonical_axis(axis_b, Y.ndim)
            X, Y = reshape(X, can_axis_a), reshape(Y, can_axis_b)
            XX = X.transpose() if trans_a else X
            YY = Y.transpose() if trans_b else Y
            return (XX.dot(YY), )

        # Check against numpy reference
        self.assertReferenceChecks(
            gc, op, [X, Y, axis_a, axis_b, trans_a, trans_b], matmul_ref
        )
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [X, Y], [0])
        # Gradient check wrt X
        self.assertGradientChecks(gc, op, [X, Y], 0, [0])
        # Gradient check wrt Y
        self.assertGradientChecks(gc, op, [X, Y], 1, [0])


class TestBatchMatMul(hu.HypothesisTestCase):
    @settings(max_examples=30)
    @given(
        C=st.integers(min_value=0, max_value=3),  # number of batch dims
        M=st.integers(min_value=1, max_value=10),
        K=st.integers(min_value=1, max_value=10),
        N=st.integers(min_value=1, max_value=10),
        trans_a=st.booleans(),
        trans_b=st.booleans(),
        dtype=st.sampled_from([np.float32, np.float16]),
        **hu.gcs
    )
    def test_batch_matmul(self, C, M, K, N, trans_a, trans_b, dtype, gc, dc):
        if dtype == np.float16:
            # fp16 is only supported with CUDA
            assume(gc.device_type == caffe2_pb2.CUDA)
            dc = [d for d in dc if d.device_type == caffe2_pb2.CUDA]

        batch_dims = np.random.randint(
            low=1,
            high=3,
            size=C,
            dtype=np.int64).tolist()
        X = np.random.rand(*(batch_dims + [M, K])).astype(dtype) - 0.5
        if trans_a:
            X = X.swapaxes(-1, -2)
        Y = np.random.rand(*(batch_dims + [K, N])).astype(dtype) - 0.5
        if trans_b:
            Y = Y.swapaxes(-1, -2)

        op = core.CreateOperator(
            'BatchMatMul', ['X', 'Y'], 'out', trans_a=trans_a, trans_b=trans_b
        )

        def matmul_ref(X, Y, trans_a, trans_b):
            XX = X.swapaxes(-1, -2) if trans_a else X
            YY = Y.swapaxes(-1, -2) if trans_b else Y
            return (np.matmul(XX, YY),)

        # Check against numpy reference
        self.assertReferenceChecks(gc, op, [X, Y, trans_a, trans_b], matmul_ref)
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [X, Y], [0])

        kwargs = {}
        if dtype == np.float16:
            kwargs['threshold'] = 0.75  # default is 0.005

        # Gradient check wrt X
        self.assertGradientChecks(gc, op, [X, Y], 0, [0], **kwargs)
        # Gradient check wrt Y
        self.assertGradientChecks(gc, op, [X, Y], 1, [0], **kwargs)


if __name__ == "__main__":
    import unittest
    unittest.main()
