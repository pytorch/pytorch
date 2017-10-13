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

#ifndef CAFFE2_OPERATORS_TRANSPOSE_GPU_H_
#define CAFFE2_OPERATORS_TRANSPOSE_GPU_H_

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

template <typename T>
bool TransposeCUDA(
    vector<int>& axes,
    CUDAContext& context,
    const Tensor<CUDAContext>& input,
    Tensor<CUDAContext>* output,
    TensorCPU& buffer_cpu,
    Tensor<CUDAContext>& buffer);
}

#endif // CAFFE2_OPERATORS_TRANSPOSE_H_
