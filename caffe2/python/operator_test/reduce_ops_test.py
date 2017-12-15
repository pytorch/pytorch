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

from caffe2.python import core
from hypothesis import given

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


class TestReduceFrontReductions(hu.HypothesisTestCase):

    def reduce_op_test(self, op_name, op_ref, in_data, num_reduce_dims, device):
        op = core.CreateOperator(
            op_name,
            ["inputs"],
            ["outputs"],
            num_reduce_dim=num_reduce_dims
        )

        self.assertReferenceChecks(
            device_option=device,
            op=op,
            inputs=[in_data],
            reference=op_ref
        )

        self.assertGradientChecks(
            device, op, [in_data], 0, [0], stepsize=1e-2, threshold=1e-2)

    @given(num_reduce_dim=st.integers(0, 4), **hu.gcs)
    def test_reduce_front_sum(self, num_reduce_dim, gc, dc):
        X = np.random.rand(7, 4, 3, 5).astype(np.float32)

        def ref_sum(X):
            return [np.sum(X, axis=(tuple(range(num_reduce_dim))))]

        self.reduce_op_test("ReduceFrontSum", ref_sum, X, num_reduce_dim, gc)

    @given(num_reduce_dim=st.integers(0, 4), **hu.gcs)
    def test_reduce_front_mean(self, num_reduce_dim, gc, dc):
        X = np.random.rand(6, 7, 8, 2).astype(np.float32)

        def ref_mean(X):
            return [np.mean(X, axis=(tuple(range(num_reduce_dim))))]

        self.reduce_op_test("ReduceFrontMean", ref_mean, X, num_reduce_dim, gc)

    @given(num_reduce_dim=st.integers(0, 4), **hu.gcs)
    def test_reduce_front_max(self, num_reduce_dim, gc, dc):
        X = np.random.rand(6, 7, 8, 2).astype(np.float32)

        def ref_frontmax(X):
            return [np.max(X, axis=(tuple(range(num_reduce_dim))))]

        op = core.CreateOperator(
            "ReduceFrontMax",
            ["inputs"],
            ["outputs"],
            num_reduce_dim=num_reduce_dim
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=ref_frontmax,
        )

        # Skip gradient check because it is too unreliable with max.
        # Just check CPU and CUDA have same results
        Y = np.array(ref_frontmax(X)[0]).astype(np.float32)
        dY = np.array(np.random.rand(*Y.shape)).astype(np.float32)
        grad_op = core.CreateOperator(
            "ReduceFrontMaxGradient",
            ["dY", "X", "Y"],
            ["dX"],
            num_reduce_dim=num_reduce_dim
        )
        self.assertDeviceChecks(dc, grad_op, [dY, X, Y], [0])

    @given(num_reduce_dim=st.integers(0, 4), **hu.gcs)
    def test_reduce_back_max(self, num_reduce_dim, gc, dc):
        X = np.random.rand(6, 7, 8, 2).astype(np.float32)

        def ref_backmax(X):
            return [np.max(X, axis=(0, 1, 2, 3)[4 - num_reduce_dim:])]

        op = core.CreateOperator(
            "ReduceBackMax",
            ["inputs"],
            ["outputs"],
            num_reduce_dim=num_reduce_dim
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=ref_backmax
        )

        # Skip gradient check because it is too unreliable with max
        # Just check CPU and CUDA have same results
        Y = np.array(ref_backmax(X)[0]).astype(np.float32)
        dY = np.array(np.random.rand(*Y.shape)).astype(np.float32)
        grad_op = core.CreateOperator(
            "ReduceBackMaxGradient",
            ["dY", "X", "Y"],
            ["dX"],
            num_reduce_dim=num_reduce_dim
        )
        self.assertDeviceChecks(dc, grad_op, [dY, X, Y], [0])

    @given(num_reduce_dim=st.integers(0, 4), **hu.gcs)
    def test_reduce_back_sum(self, num_reduce_dim, dc, gc):
        X = np.random.rand(6, 7, 8, 2).astype(np.float32)

        def ref_sum(X):
            return [np.sum(X, axis=(0, 1, 2, 3)[4 - num_reduce_dim:])]

        self.reduce_op_test("ReduceBackSum", ref_sum, X, num_reduce_dim, gc)

    @given(num_reduce_dim=st.integers(0, 4), **hu.gcs)
    def test_reduce_back_mean(self, num_reduce_dim, dc, gc):
        num_reduce_dim = 2
        X = np.random.rand(6, 7, 8, 2).astype(np.float32)

        def ref_sum(X):
            return [np.mean(X, axis=(0, 1, 2, 3)[4 - num_reduce_dim:])]

        self.reduce_op_test("ReduceBackMean", ref_sum, X, num_reduce_dim, gc)
