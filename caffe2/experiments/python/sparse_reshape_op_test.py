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

from caffe2.python import core, workspace
from caffe2.python.test_util import TestCase


def test_reshape(old_shape, new_shape, stride_only=False):
    blob_in0 = 'col'
    blob_out0 = 'col_out'

    blob_in1 = 'row'
    blob_out1 = 'row_out'

    old_shape_for_op = (-1, old_shape[1]) if stride_only else old_shape

    op = core.CreateOperator('SparseMatrixReshape',
                             [blob_in0, blob_in1],
                             [blob_out0, blob_out1],
                             old_shape=old_shape_for_op,
                             new_shape=new_shape)

    A = np.random.random_sample(old_shape)
    A[np.random.random_sample(old_shape) > .5] = 0
    A_coo = coo_matrix(A)
    old_row, old_col = A_coo.row, A_coo.col

    workspace.FeedBlob(blob_in0, old_col.astype(np.int64))
    workspace.FeedBlob(blob_in1, old_row.astype(np.int32))

    workspace.RunOperatorOnce(op)

    A_new_coo = coo_matrix(A.reshape(new_shape))
    new_row, new_col = A_new_coo.row, A_new_coo.col

    col_out = workspace.FetchBlob(blob_out0)
    row_out = workspace.FetchBlob(blob_out1)

    np.testing.assert_array_equal(col_out, new_col)
    np.testing.assert_array_equal(row_out, new_row)


class TestSparseMatrixReshapeOp(TestCase):
    def test_basic_reshape(self):
        test_reshape(old_shape=(3, 4), new_shape=(4, 3))

    def test_missing_dim(self):
        test_reshape(old_shape=(2, 8), new_shape=(-1, 4))

    def test_stride_only(self):
        test_reshape(old_shape=(2, 8), new_shape=(-1, 4), stride_only=True)

    def test_sparse_reshape_mm(self):
        M, N, K = 300, 400, 500
        A = np.random.rand(M, K).astype(np.float32)
        A_sparse = A * (np.random.rand(*A.shape) > .5)
        A_sparse = A_sparse.reshape((K, M))
        A_coo = coo_matrix(A_sparse)
        idx0, idx1, a = A_coo.row, A_coo.col, A_coo.data
        B = np.random.rand(K, N).astype(np.float32)

        workspace.FeedBlob('col', idx1.astype(np.int64))
        workspace.FeedBlob('row', idx0.astype(np.int32))
        workspace.FeedBlob('B', B)
        workspace.FeedBlob('a', a)

        reshape_op = core.CreateOperator(
            'SparseMatrixReshape',
            ['col', 'row'],
            ['new_col', 'new_row'],
            old_shape=(K, M),
            new_shape=(M, K))

        mm_op = core.CreateOperator(
            'SparseUnsortedSegmentWeightedSum',
            ['B', 'a', 'new_col', 'new_row'],
            ['Y'])

        workspace.RunOperatorOnce(reshape_op)
        workspace.RunOperatorOnce(mm_op)

        Y = workspace.FetchBlob('Y')
        np.testing.assert_allclose(A_sparse.reshape(M, K).dot(B), Y,
                                   rtol=1e-4)
