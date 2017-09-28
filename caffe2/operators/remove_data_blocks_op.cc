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

#include "caffe2/operators/remove_data_blocks_op.h"

namespace caffe2 {
namespace {
REGISTER_CPU_OPERATOR(RemoveDataBlocks, RemoveDataBlocksOp<CPUContext>);

OPERATOR_SCHEMA(RemoveDataBlocks)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Shrink the data tensor by removing data blocks with given zero-based indices in
the outermost dimension of the tensor. Indices are not assumed in any order or
unique but with the range [0, blocks_size). Indices could be empty.
  )DOC")
    .Input(0, "data", "a N-D data tensor, N >= 1")
    .Input(1, "indices", "zero-based indices of blocks to be removed")
    .Output(
        0,
        "shrunk data",
        "data after removing data blocks indexed by 'indices'");

SHOULD_NOT_DO_GRADIENT(RemoveDataBlocks);
} // namespace
} // namespace caffe2
