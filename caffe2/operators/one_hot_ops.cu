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

#include <cub/block/block_reduce.cuh>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/one_hot_ops.h"

namespace caffe2 {

__global__ void OneHotOpKernel(
    const TIndex batch_size,
    const TIndex index_size,
    const TIndex* indices,
    float* output) {
  CUDA_1D_KERNEL_LOOP(i, batch_size) {
    output[i * index_size + indices[i]] = 1.;
  }
}

template <>
void OneHotOp<CUDAContext>::DoOneHotOp(
    TIndex batch_size,
    TIndex index_size,
    const Tensor<CUDAContext>& indices,
    Tensor<CUDAContext>* output) {
  float* output_ptr = output->mutable_data<float>();
  math::Set<float, CUDAContext>(output->size(), 0., output_ptr, &context_);
  OneHotOpKernel<<<
      CAFFE_GET_BLOCKS(batch_size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      batch_size, index_size, indices.data<TIndex>(), output_ptr);
}

REGISTER_CUDA_OPERATOR(OneHot, OneHotOp<CUDAContext>);
} // namespace
