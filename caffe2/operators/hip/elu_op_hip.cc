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

#include "caffe2/operators/elu_op.h"
#include "hip/hip_runtime.h"
#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

namespace {
__global__ void elu_kernel(const int N, const float alpha, const float* x, float* y)
{
    HIP_1D_KERNEL_LOOP(i, N)
    {
        if(x[i] > 0)
        {
            y[i] = x[i];
        }
        else
        {
            y[i] = alpha * (expf(x[i]) - 1);
        }
    }
}

__global__ void
elu_gradient_kernel(const int N, const float alpha, const float* y, const float* dy, float* dx)
{
    HIP_1D_KERNEL_LOOP(i, N)
    {
        if(y[i] > 0)
        {
            dx[i] = dy[i];
        }
        else
        {
            dx[i] = dy[i] * (y[i] + alpha);
        }
    }
}
} // namespace

template <>
bool EluOp<float, HIPContext>::RunOnDevice()
{
    auto& X = Input(0);
    auto* Y = Output(0);
    // Otherwise inplace gradient and Elu dosen't make sense.
    CAFFE_ENFORCE_GE(alpha_, 0);
    Y->ResizeLike(X);
    const auto* Xdata = X.data<float>();
    auto* Ydata       = Y->mutable_data<float>();
    hipLaunchKernelGGL((elu_kernel),
                       dim3(CAFFE_GET_BLOCKS(X.size())),
                       dim3(CAFFE_HIP_NUM_THREADS),
                       0,
                       context_.hip_stream(),
                       static_cast<const int>(X.size()),
                       alpha_,
                       Xdata,
                       Ydata);
    return true;
}

template <>
bool EluGradientOp<float, HIPContext>::RunOnDevice()
{
    auto& Y  = Input(0);
    auto& dY = Input(1);
    auto* dX = Output(0);
    DCHECK_GT(Y.size(), 0);
    DCHECK_EQ(dY.size(), Y.size());
    dX->ResizeLike(Y);

    const float* Ydata  = Y.data<float>();
    const float* dYdata = dY.data<float>();
    float* dXdata       = dX->mutable_data<float>();
    hipLaunchKernelGGL((elu_gradient_kernel),
                       dim3(CAFFE_GET_BLOCKS(Y.size())),
                       dim3(CAFFE_HIP_NUM_THREADS),
                       0,
                       context_.hip_stream(),
                       static_cast<const int>(Y.size()),
                       alpha_,
                       Ydata,
                       dYdata,
                       dXdata);
    return true;
}

REGISTER_HIP_OPERATOR(Elu, EluOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(EluGradient, EluGradientOp<float, HIPContext>);
}
