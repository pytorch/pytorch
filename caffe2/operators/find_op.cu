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
#include "caffe2/operators/find_op.h"

namespace caffe2 {

template <typename T>
__global__ void FindKernel(
    int num_needles,
    int idx_size,
    const T* idx,
    const T* needles,
    int* out,
    int missing_value) {
  int needle_idx = blockIdx.x; // One cuda block per needle
  T q = needles[needle_idx];
  int res = (-1);
  for (int j = threadIdx.x; j < idx_size; j += CAFFE_CUDA_NUM_THREADS) {
    if (idx[j] == q) {
      res = max(res, j);
    }
  }
  typedef cub::BlockReduce<int, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int min_res = BlockReduce(temp_storage).Reduce(res, cub::Max());
  if (threadIdx.x == 0) {
    out[needle_idx] = min_res == (-1) ? missing_value : min_res;
  }
}

template <>
template <typename T>
bool FindOp<CUDAContext>::DoRunWithType() {
  auto& idx = Input(0);
  auto& needles = Input(1);
  auto* res_indices = Output(0);
  res_indices->ResizeLike(needles);

  const T* idx_data = idx.data<T>();
  const T* needles_data = needles.data<T>();
  int* res_data = res_indices->mutable_data<int>();

  FindKernel<
      T><<<needles.size(), CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
      needles.size(),
      idx.size(),
      idx_data,
      needles_data,
      res_data,
      missing_value_);
  return true;
}

REGISTER_CUDA_OPERATOR(Find, FindOp<CUDAContext>)

} // namespace caffe2
