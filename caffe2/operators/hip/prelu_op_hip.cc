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

#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/prelu_op.h"
#include "hip/hip_runtime.h"
#include <cub/block/block_reduce.cuh>

namespace caffe2 {
namespace {
template <typename T>
__global__ void PReluKernel(const int N, const T* X, const T* W, T* Y)
{
    HIP_1D_KERNEL_LOOP(i, N) { Y[i] = (X[i] > 0) * X[i] + (X[i] < 0) * X[i] * W[0]; }
}

template <typename T>
__global__ void
PReluKernelNCHW(const int N, const int C, const int dim, const T* X, const T* W, T* Y)
{
    HIP_1D_KERNEL_LOOP(i, N * C * dim)
    {
        int c = (i / dim) % C;
        Y[i]  = (X[i] > 0) * X[i] + (X[i] < 0) * X[i] * W[c];
    }
}

template <typename T>
__global__ void PReluKernelNHWC(const int nitems, const int C, const T* X, const T* W, T* Y)
{
    HIP_1D_KERNEL_LOOP(i, nitems)
    {
        int c = i % C;
        Y[i]  = (X[i] > 0) * X[i] + (X[i] < 0) * X[i] * W[c];
    }
}

template <typename T>
__global__ void PReluGradientKernel(const int N, const T* X, const T* W, const T* dY, T* dX)
{
    HIP_1D_KERNEL_LOOP(i, N) { dX[i] = (X[i] > 0) * dY[i] + (X[i] <= 0) * dY[i] * W[0]; }
}

template <typename T>
__global__ void PReluGradientKernelNCHW(
    const int N, const int C, const int dim, const T* X, const T* W, const T* dY, T* dX)
{
    HIP_1D_KERNEL_LOOP(i, N * C * dim)
    {
        int c = (i / dim) % C;
        dX[i] = (X[i] > 0) * dY[i] + (X[i] <= 0) * dY[i] * W[c];
    }
}

template <typename T>
__global__ void
PReluGradientKernelNHWC(const int nitems, const int C, const T* X, const T* W, const T* dY, T* dX)
{
    HIP_1D_KERNEL_LOOP(i, nitems)
    {
        int c = i % C;
        dX[i] = (X[i] > 0) * dY[i] + (X[i] <= 0) * dY[i] * W[c];
    }
}

template <typename T>
__global__ void
PReluSharedWGradientKernelNCHW(const int num_items, const T* Xdata, const T* dYdata, T* dW)
{
    T wsum = 0.0;
    for(int i = hipThreadIdx_x; i < num_items; i += hipBlockDim_x)
    {
        wsum += (Xdata[i] <= 0) * dYdata[i] * Xdata[i];
    }

    typedef cub::BlockReduce<T, CAFFE_HIP_NUM_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    T sum = BlockReduce(temp_storage).Sum(wsum);
    if(hipThreadIdx_x == 0)
    {
        *dW = sum;
    }
}

template <typename T>
__global__ void PReluWGradientKernelNCHW(
    const int C, const int N, const int num_items, const T* Xdata, const T* dYdata, T* dW)
{
    int c = hipBlockIdx_x;

    T wsum                       = 0.0;
    int items_per_channel        = num_items / C;
    int items_per_sample_channel = items_per_channel / N;
    for(int i = hipThreadIdx_x; i < items_per_channel; i += hipBlockDim_x)
    {
        // TODO: simplify
        int n  = i / items_per_sample_channel;
        int ii = n * items_per_sample_channel * C + c * items_per_sample_channel +
                 i % items_per_sample_channel;
        wsum += (Xdata[ii] <= 0) * dYdata[ii] * Xdata[ii];
    }

    typedef cub::BlockReduce<T, CAFFE_HIP_NUM_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    T sum = BlockReduce(temp_storage).Sum(wsum);
    if(hipThreadIdx_x == 0)
    {
        dW[c] = sum;
    }
}

template <typename T>
__global__ void PReluWGradientKernelNHWC(
    const int C, const int N, const int num_items, const T* Xdata, const T* dYdata, T* dW)
{
    int c                 = hipBlockIdx_x;
    T wsum                = 0.0;
    int items_per_channel = num_items / C;
    for(int i = hipThreadIdx_x; i < items_per_channel; i += hipBlockDim_x)
    {
        int ii = i * C + c;
        wsum += (Xdata[ii] <= 0) * dYdata[ii] * Xdata[ii];
    }

    typedef cub::BlockReduce<T, CAFFE_HIP_NUM_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    T sum = BlockReduce(temp_storage).Sum(wsum);
    if(hipThreadIdx_x == 0)
    {
        dW[c] = sum;
    }
}

} // namespace

template <>
bool PReluOp<float, HIPContext>::RunOnDevice()
{
    const auto& X = Input(0);
    const auto& W = Input(1);
    auto* Y       = Output(0);
    Y->ResizeLike(X);
    const auto* Xdata = X.data<float>();
    const auto* Wdata = W.data<float>();
    auto* Ydata       = Y->mutable_data<float>();

    const auto C        = order_ == StorageOrder::NCHW ? X.dim(1) : X.dim(X.ndim() - 1);
    const auto C_shared = (W.size() == 1);

    if(!C_shared)
    {
        CAFFE_ENFORCE_EQ(C, W.size());
    }
    if(C_shared)
    {
        hipLaunchKernelGGL((PReluKernel),
                           dim3(CAFFE_GET_BLOCKS(X.size())),
                           dim3(CAFFE_HIP_NUM_THREADS),
                           0,
                           context_.hip_stream(),
                           static_cast<const int>(X.size()),
                           Xdata,
                           Wdata,
                           Ydata);
        return true;
    }
    // non-shared case.
    switch(order_)
    {
    case StorageOrder::NCHW:
    {
        const auto N   = X.dim(0);
        const auto dim = X.size_from_dim(2);
        CHECK(N * C * dim == X.size());
        hipLaunchKernelGGL((PReluKernelNCHW),
                           dim3(CAFFE_GET_BLOCKS(X.size())),
                           dim3(CAFFE_HIP_NUM_THREADS),
                           0,
                           context_.hip_stream(),
                           static_cast<const int>(N),
                           static_cast<const int>(C),
                           static_cast<const int>(dim),
                           Xdata,
                           Wdata,
                           Ydata);

        break;
    }
    case StorageOrder::NHWC:
    {
        hipLaunchKernelGGL((PReluKernelNHWC),
                           dim3(CAFFE_GET_BLOCKS(X.size())),
                           dim3(CAFFE_HIP_NUM_THREADS),
                           0,
                           context_.hip_stream(),
                           static_cast<const int>(X.size()),
                           static_cast<const int>(C),
                           Xdata,
                           Wdata,
                           Ydata);

        break;
    }
    default: CAFFE_THROW("Unknown storage order: ", order_);
    }
    return true;
}

template <>
bool PReluGradientOp<float, HIPContext>::RunOnDevice()
{
    auto& Y  = Input(0);
    auto& dY = Input(1);
    auto& X  = Input(2);
    auto& W  = Input(3);

    CAFFE_ENFORCE(&Y != &X, "Cannot backpropagate through an in-place PReLU");
    auto* dX = Output(0);
    auto* dW = Output(1);

    DCHECK_EQ(dY.size(), Y.size());
    dX->ResizeLike(Y);
    dW->ResizeLike(W);

    const auto C        = order_ == StorageOrder::NCHW ? X.dim(1) : X.dim(X.ndim() - 1);
    const auto C_shared = (W.size() == 1);

    const float* Ydata  = Y.data<float>();
    const float* dYdata = dY.data<float>();
    const float* Xdata  = X.data<float>();
    const float* Wdata  = W.data<float>();
    float* dXdata       = dX->mutable_data<float>();
    float* dWdata       = dW->mutable_data<float>();
    int N               = Y.dim(0);

    if(C_shared)
    {
        hipLaunchKernelGGL((PReluSharedWGradientKernelNCHW),
                           dim3(1),
                           dim3(CAFFE_HIP_NUM_THREADS),
                           0,
                           context_.hip_stream(),
                           static_cast<const int>(X.size()),
                           Xdata,
                           dYdata,
                           dWdata);
        hipLaunchKernelGGL((PReluGradientKernel),
                           dim3(CAFFE_GET_BLOCKS(X.size())),
                           dim3(CAFFE_HIP_NUM_THREADS),
                           0,
                           context_.hip_stream(),
                           static_cast<const int>(X.size()),
                           Xdata,
                           Wdata,
                           dYdata,
                           dXdata);

        return true;
    }
    // non-shared case.
    switch(order_)
    {
    case StorageOrder::NCHW:
    {
        const auto dim = Y.size_from_dim(2);
        hipLaunchKernelGGL((PReluWGradientKernelNCHW),
                           dim3(C),
                           dim3(CAFFE_HIP_NUM_THREADS),
                           0,
                           context_.hip_stream(),
                           static_cast<const int>(C),
                           static_cast<const int>(N),
                           static_cast<const int>(X.size()),
                           Xdata,
                           dYdata,
                           dWdata);
        hipLaunchKernelGGL((PReluGradientKernelNCHW),
                           dim3(CAFFE_GET_BLOCKS(X.size())),
                           dim3(CAFFE_HIP_NUM_THREADS),
                           0,
                           context_.hip_stream(),
                           static_cast<const int>(N),
                           static_cast<const int>(C),
                           static_cast<const int>(dim),
                           Xdata,
                           Wdata,
                           dYdata,
                           dXdata);

        break;
    }
    case StorageOrder::NHWC:
    {
        hipLaunchKernelGGL((PReluWGradientKernelNHWC),
                           dim3(C),
                           dim3(CAFFE_HIP_NUM_THREADS),
                           0,
                           context_.hip_stream(),
                           static_cast<const int>(C),
                           static_cast<const int>(N),
                           static_cast<const int>(X.size()),
                           Xdata,
                           dYdata,
                           dWdata);
        hipLaunchKernelGGL((PReluGradientKernelNHWC),
                           dim3(CAFFE_GET_BLOCKS(Y.size())),
                           dim3(CAFFE_HIP_NUM_THREADS),
                           0,
                           context_.hip_stream(),
                           static_cast<const int>(X.size()),
                           static_cast<const int>(C),
                           Xdata,
                           Wdata,
                           dYdata,
                           dXdata);

        break;
    }
    default: CAFFE_THROW("Unknown storage order: ", order_);
    }
    return true;
}

REGISTER_HIP_OPERATOR(PRelu, PReluOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(PReluGradient, PReluGradientOp<float, HIPContext>);
} // namespace caffe2
