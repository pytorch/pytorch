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
#include "caffe2/operators/floor_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T>
__global__ void FloorKernel(const int N, const T* X, T* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = std::floor(X[i]);
  }
}

template <>
bool FloorOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  CAFFE_ENFORCE_GT(X.size(), 0);
  Y->ResizeLike(X);
  FloorKernel<<<
      CAFFE_GET_BLOCKS(X.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      X.size(), X.data<float>(), Y->mutable_data<float>());
  return true;
}

REGISTER_CUDA_OPERATOR(Floor, FloorOp<float, CUDAContext>);

} // namespace caffe2
