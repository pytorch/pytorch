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

#include "hip/hip_runtime.h"
#include "caffe2/core/context_hip.h"
#include "caffe2/operators/leaky_relu_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
namespace {
template <typename T>
__global__ void LeakyReluKernel(const int N, const T alpha, const T* X, T* Y)
{
    HIP_1D_KERNEL_LOOP(i, N) { Y[i] = X[i] >= 0 ? X[i] : X[i] * alpha; }
}

template <typename T>
__global__ void LeakyReluGradientKernel(const int N, const T alpha, const T* Y, const T* dY, T* dX)
{
    HIP_1D_KERNEL_LOOP(i, N) { dX[i] = Y[i] >= 0 ? dY[i] : dY[i] * alpha; }
}
} // namespace

template <>
bool LeakyReluOp<float, HIPContext>::RunOnDevice()
{
    const auto& X = Input(0);
    CAFFE_ENFORCE_GT(X.size(), 0);
    auto* Y = Output(0);
    Y->ResizeLike(X);
    hipLaunchKernelGGL((LeakyReluKernel),
                       dim3(CAFFE_GET_BLOCKS(X.size())),
                       dim3(CAFFE_HIP_NUM_THREADS),
                       0,
                       context_.hip_stream(),
                       static_cast<const int>(X.size()),
                       alpha_,
                       X.data<float>(),
                       Y->mutable_data<float>());
    return true;
}

template <>
bool LeakyReluGradientOp<float, HIPContext>::RunOnDevice()
{
    const auto& Y  = Input(0);
    const auto& dY = Input(1);
    auto* dX       = Output(0);
    dX->ResizeLike(Y);
    CAFFE_ENFORCE_EQ(Y.size(), dY.size());
    hipLaunchKernelGGL((LeakyReluGradientKernel),
                       dim3(CAFFE_GET_BLOCKS(Y.size())),
                       dim3(CAFFE_HIP_NUM_THREADS),
                       0,
                       context_.hip_stream(),
                       static_cast<const int>(Y.size()),
                       alpha_,
                       Y.data<float>(),
                       dY.data<float>(),
                       dX->mutable_data<float>());
    return true;
}

REGISTER_HIP_OPERATOR(LeakyRelu, LeakyReluOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(LeakyReluGradient, LeakyReluGradientOp<float, HIPContext>);
} // namespace caffe2
