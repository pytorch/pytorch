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

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/concat_split_op.h"

namespace caffe2 {
REGISTER_CUDA_OPERATOR(Split, SplitOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(Concat, ConcatOp<CUDAContext>);

// Backward compatibility settings
REGISTER_CUDA_OPERATOR(DepthSplit, SplitOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(DepthConcat, ConcatOp<CUDAContext>);
}  // namespace caffe2
