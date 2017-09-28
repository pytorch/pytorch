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

#include <cmath>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {

template <typename T>
__global__ void SigmoidKernel(const int N, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = 1. / (1. + exp(-x[i]));
  }
}

template <typename T>
__global__ void SigmoidGradientKernel(const int N, const T* y, const T* dy,
                              T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = dy[i] * y[i] * (1. - y[i]);
  }
}

struct SigmoidCUDAFunctor {
  template <typename T>
  inline void operator()(const int n, const T* x,
                         T* y, CUDAContext* device_context) {
    SigmoidKernel<T><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS,
                    0, device_context->cuda_stream()>>>(n, x, y);
    return;
  }
};

struct SigmoidGradientCUDAFunctor {
  template <typename T>
  inline void Run(const int n, const T* y, const T* dy,
                  T* dx, CUDAContext* device_context) {
    SigmoidGradientKernel<T><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS,
                            0, device_context->cuda_stream()>>>(n, y, dy, dx);
    return;
  }
};

REGISTER_CUDA_OPERATOR(
    Sigmoid,
    UnaryElementwiseOp<TensorTypes<float>, CUDAContext, SigmoidCUDAFunctor>);
REGISTER_CUDA_OPERATOR(
    SigmoidGradient, BinaryElementwiseOp<
        TensorTypes<float>, CUDAContext,
        WithoutBroadcast<SigmoidGradientCUDAFunctor>>);
}  // namespace caffe2
