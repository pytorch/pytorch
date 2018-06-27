#include "hip/hip_runtime.h"
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

#include "caffe2/core/context_hip.h"
#include "caffe2/operators/relu_op.h"

namespace caffe2 {
namespace {
template <typename T>
__global__ void ReluKernel(const int N, const T* X, T* Y)
{
    HIP_1D_KERNEL_LOOP(i, N) { Y[i] = X[i] > 0 ? X[i] : 0; }
}

template <typename T>
__global__ void ReluGradientKernel(const int N, const T* Y, const T* dY, T* dX)
{
    HIP_1D_KERNEL_LOOP(i, N) { dX[i] = Y[i] > 0 ? dY[i] : 0; }
}
} // namespace

template <>
bool ReluOp<float, HIPContext>::RunOnDevice()
{
    auto& X = Input(0);
    auto* Y = Output(0);
    CAFFE_ENFORCE_GT(X.size(), 0);
    Y->ResizeLike(X);
    hipLaunchKernelGGL((ReluKernel),
                       dim3(CAFFE_GET_BLOCKS(X.size())),
                       dim3(CAFFE_HIP_NUM_THREADS),
                       0,
                       context_.hip_stream(),
                       static_cast<const int>(X.size()),
                       static_cast<const float*>(X.data<float>()),
                       static_cast<float*>(Y->mutable_data<float>()));
    return true;
}

template <>
bool ReluGradientOp<float, HIPContext>::RunOnDevice()
{
    auto& Y  = Input(0);
    auto& dY = Input(1);
    auto* dX = Output(0);
    CAFFE_ENFORCE_GT(Y.size(), 0);
    CAFFE_ENFORCE_EQ(dY.size(), Y.size());
    dX->ResizeLike(Y);
    hipLaunchKernelGGL((ReluGradientKernel),
                       dim3(CAFFE_GET_BLOCKS(Y.size())),
                       dim3(CAFFE_HIP_NUM_THREADS),
                       0,
                       context_.hip_stream(),
                       static_cast<const int>(Y.size()),
                       static_cast<const float*>(Y.data<float>()),
                       static_cast<const float*>(dY.data<float>()),
                       static_cast<float*>(dX->mutable_data<float>()));
    return true;
}

REGISTER_HIP_OPERATOR(Relu, ReluOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(ReluGradient, ReluGradientOp<float, HIPContext>);
} // namespace caffe2
