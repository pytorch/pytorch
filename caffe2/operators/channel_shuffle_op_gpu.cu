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
#include "channel_shuffle_op.h"

namespace caffe2 {

__global__ void ChannelShuffleKernel(
    const int N,
    const int S,
    const int C,
    const int G,
    const int K,
    const float* Xdata,
    float* Ydata) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const int out_s = i % S;
    const int i_2 = i / S;
    const int out_c = i_2 % C;
    const int n = i_2 / C;

    const int g = out_c % G;
    const int k = out_c / G;
    const int in_c = k + K * g;
    Ydata[out_s + S * out_c + S * C * n] = Xdata[out_s + S * in_c + S * C * n];
  }
}

template <>
bool ChannelShuffleOp<CUDAContext>::RunOnDeviceWithOrderNCHW() {
  const auto& X = Input(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);
  const auto C = X.dim32(1);
  CAFFE_ENFORCE(C % this->group_ == 0, "");
  const auto K = C / this->group_;
  const auto S = X.dim32(2) * X.dim32(3);
  const auto G = this->group_;
  ChannelShuffleKernel<<<
      CAFFE_GET_BLOCKS(X.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      X.size(), S, C, G, K, X.data<float>(), Y->mutable_data<float>());
  return true;
}

template <>
bool ChannelShuffleGradientOp<CUDAContext>::RunOnDeviceWithOrderNCHW() {
  const auto& dY = Input(0);
  auto* dX = Output(0);
  dX->ResizeLike(dY);
  const auto C = dY.dim32(1);
  CAFFE_ENFORCE(C % this->group_ == 0, "");
  const auto K = C / this->group_;
  const auto S = dY.dim32(2) * dY.dim32(3);
  const auto G = this->group_;
  ChannelShuffleKernel<<<
      CAFFE_GET_BLOCKS(dY.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      dY.size(), S, C, K, G, dY.data<float>(), dX->mutable_data<float>());
  return true;
}

REGISTER_CUDA_OPERATOR(ChannelShuffle, ChannelShuffleOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(
    ChannelShuffleGradient,
    ChannelShuffleGradientOp<CUDAContext>);
}
