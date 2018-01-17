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

#include "affine_channel_op.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {
template <typename T>
__global__ void ScaleBiasForward(
    const int n,
    const T* in,
    const T* scale,
    const T* bias,
    const int scale_dim,
    const int hxw_dim,
    T* out) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int scale_index = (index / hxw_dim) % scale_dim;
    out[index] = in[index] * scale[scale_index] + bias[scale_index];
  }
}

template <typename T>
__global__ void ScaleForward(
    const int n,
    const T* in,
    const T* scale,
    const int scale_dim,
    const int hxw_dim,
    T* out) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int scale_index = (index / hxw_dim) % scale_dim;
    out[index] = in[index] * scale[scale_index];
  }
}
} // namespace

template <>
bool AffineChannelOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& scale = Input(1);
  auto& bias = Input(2);
  auto* Y = Output(0);

  Y->ResizeLike(X);
  const int output_size = Y->size();
  ScaleBiasForward<float>
      <<<CAFFE_GET_BLOCKS(output_size),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(
          output_size,
          X.data<float>(),
          scale.data<float>(),
          bias.data<float>(),
          X.dim32(1),
          X.dim32(2) * X.dim32(3),
          Y->mutable_data<float>());
  return true;
}

template <>
bool AffineChannelGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& scale = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);

  dX->ResizeLike(dY);
  ScaleForward<float>
      <<<CAFFE_GET_BLOCKS(dY.size()),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(
          dY.size(),
          dY.data<float>(),
          scale.data<float>(),
          dY.dim32(1),
          dY.dim32(2) * dY.dim32(3),
          dX->mutable_data<float>());
  return true;
}

REGISTER_CUDA_OPERATOR(AffineChannel, AffineChannelOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    AffineChannelGradient,
    AffineChannelGradientOp<float, CUDAContext>);
} // namespace caffe2
