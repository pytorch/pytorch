/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "caffe2/experiments/operators/sparse_matrix_reshape_op.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(SparseMatrixReshape, SparseMatrixReshapeOp<CPUContext>);

OPERATOR_SCHEMA(SparseMatrixReshape)
    .NumInputs(2)
    .NumOutputs(2)
    .DisallowInputFillers() // TODO: enable the filler
    .AllowInplace({{0, 0}, {1, 1}})
    .SetDoc(R"DOC(
Compute the indices of the reshaped sparse matrix.

It takes two 1D tensors as input: the column indices (in int64) and
the row indices (in int), which correspond to `INDICES` and `SEGMENT_IDS`
in `SparseSortedSegment` family.
It outputs the corresponding reshaped column and row indices.

Two arguments are required:
an argument `old_shape` specifies the original shape of the matrix,
and `new_shape` specifies the new shape.
One of the dimension in `old_shape` and `new_shape` can be -1.
The valid combinations are listed below, where p, q, r, s are
strictly positive integers.

old_shape=(p, q)
new_shape=(r, s)

old_shape=(p, q)
new_shape=(-1, s)

old_shape=(p, q)
new_shape=(r, -1)

old_shape=(-1, q)
new_shape=(-1, s)

Note that only the first dimension in `old_shape` can be -1. In that case
the second dimension in `new_shape` must NOT be -1.
)DOC")
    .Arg("old_shape", "Old shape.")
    .Arg("new_shape", "New shape.")
    .Input(0, "old_col", "Original column indices.")
    .Input(1, "old_row", "Original row indices.")
    .Output(0, "new_col", "New column indices.")
    .Output(1, "new_row", "New row indices.");

SHOULD_NOT_DO_GRADIENT(SparseMatrixReshape);

} // namespace
} // namespace caffe2
