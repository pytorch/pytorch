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

#include "batch_permutation_op.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {
template <bool forward>
__global__ void BatchPermutationKernel(
    int N,
    int C,
    int H,
    int W,
    const float* src,
    const int* indices,
    float* dst) {
  CUDA_1D_KERNEL_LOOP(index, N * C * H * W) {
    int w = index % W;
    int h = (index / W) % H;
    int c = (index / W / H) % C;
    int n = (index / W / H / C);
    int idx = indices[n];
    if (forward) {
      dst[n * C * H * W + c * H * W + h * W + w] =
          src[idx * C * H * W + c * H * W + h * W + w];
    } else {
      dst[idx * C * H * W + c * H * W + h * W + w] =
          src[n * C * H * W + c * H * W + h * W + w];
    }
  }
}
}

template <>
bool BatchPermutationOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& indices = Input(1);
  auto* Y = Output(0);

  CAFFE_ENFORCE(indices.ndim() == 1, "indices must be 1-d");
  CAFFE_ENFORCE(
      X.dim32(0) == indices.dim32(0),
      "X.dim32(0) must be equal to indices.dim32(0)",
      "(",
      X.dim32(0),
      " vs. ",
      indices.dim32(0),
      ")");

  Y->ResizeLike(X);

  BatchPermutationKernel<true><<<
      CAFFE_GET_BLOCKS(X.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      X.dim32(0),
      X.dim32(1),
      X.dim32(2),
      X.dim32(3),
      X.data<float>(),
      indices.data<int>(),
      Y->mutable_data<float>());

  return true;
}

template <>
bool BatchPermutationGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& indices = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  dX->ResizeLike(dY);

  BatchPermutationKernel<false><<<
      CAFFE_GET_BLOCKS(dY.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      dY.dim32(0),
      dY.dim32(1),
      dY.dim32(2),
      dY.dim32(3),
      dY.data<float>(),
      indices.data<int>(),
      dX->mutable_data<float>());

  return true;
}

REGISTER_CUDA_OPERATOR(
    BatchPermutation,
    BatchPermutationOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    BatchPermutationGradient,
    BatchPermutationGradientOp<float, CUDAContext>);
} // namespace caffe2
