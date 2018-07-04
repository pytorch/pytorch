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
from scipy.sparse import coo_matrix

from hypothesis import given
import hypothesis.strategies as st

from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu


class TestFunHash(hu.HypothesisTestCase):
    @given(n_out=st.integers(min_value=5, max_value=20),
           n_in=st.integers(min_value=10, max_value=20),
           n_data=st.integers(min_value=2, max_value=8),
           n_weight=st.integers(min_value=8, max_value=15),
           n_alpha=st.integers(min_value=3, max_value=8),
           sparsity=st.floats(min_value=0.1, max_value=1.0),
           **hu.gcs)
    def test_funhash(self, n_out, n_in, n_data, n_weight, n_alpha, sparsity,
                     gc, dc):
        A = np.random.rand(n_data, n_in)
        A[A > sparsity] = 0
        A_coo = coo_matrix(A)
        val, key, seg = A_coo.data, A_coo.col, A_coo.row

        weight = np.random.rand(n_weight).astype(np.float32)
        alpha = np.random.rand(n_alpha).astype(np.float32)
        val = val.astype(np.float32)
        key = key.astype(np.int64)
        seg = seg.astype(np.int32)

        op = core.CreateOperator(
            'SparseFunHash',
            ['val', 'key', 'seg', 'weight', 'alpha'],
            ['out'],
            num_outputs=n_out)

        # Gradient check wrt weight
        self.assertGradientChecks(
            gc, op, [val, key, seg, weight, alpha], 3, [0])
        # Gradient check wrt alpha
        self.assertGradientChecks(
            gc, op, [val, key, seg, weight, alpha], 4, [0])

        op2 = core.CreateOperator(
            'SparseFunHash',
            ['val', 'key', 'seg', 'weight'],
            ['out'],
            num_outputs=n_out)

        # Gradient check wrt weight
        self.assertGradientChecks(
            gc, op2, [val, key, seg, weight], 3, [0])
