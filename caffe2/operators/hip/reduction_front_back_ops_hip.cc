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
#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/reduction_front_back_ops.h"
#include "hip/hip_runtime.h"

namespace caffe2 {

namespace {
template <typename T>
__global__ void
columnwise_fill_kernel(const int rows, const int cols, const float alpha, const T* dY, T* dX)
{
    HIP_1D_KERNEL_LOOP(i, rows * cols) { dX[i] = dY[i % cols] * alpha; }
}

template <typename T>
__global__ void
rowwise_fill_kernel(const int rows, const int cols, const float alpha, const T* dY, T* dX)
{
    HIP_1D_KERNEL_LOOP(i, rows * cols) { dX[i] = dY[i / cols] * alpha; }
}

template <typename T>
__global__ void
rowwise_sum_kernel(const int rows, const int cols, const float alpha, const T* data, T* out)
{
    typedef cub::BlockReduce<float, CAFFE_HIP_NUM_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    for(int rowIndex = hipBlockIdx_x; rowIndex < rows; rowIndex += hipGridDim_x)
    {
        T sum               = 0;
        const int rowOffset = rowIndex * cols;
        for(int colIndex = hipThreadIdx_x; colIndex < cols; colIndex += hipBlockDim_x)
        {
            sum += data[rowOffset + colIndex];
        }
        sum = BlockReduce(temp_storage).Reduce(sum, cub::Sum());
        if(hipThreadIdx_x == 0)
        {
            out[rowIndex] = alpha * sum;
        }
        __syncthreads();
    }
}

template <typename T>
__global__ void
columnwise_sum_kernel(const int rows, const int cols, const float alpha, const T* data, T* out)
{
    typedef cub::BlockReduce<float, CAFFE_HIP_NUM_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    for(int colIndex = hipBlockIdx_x; colIndex < cols; colIndex += hipGridDim_x)
    {
        T sum = 0;
        for(int rowIndex = hipThreadIdx_x; rowIndex < rows; rowIndex += hipBlockDim_x)
        {
            sum += data[rowIndex * cols + colIndex];
        }
        sum = BlockReduce(temp_storage).Reduce(sum, cub::Sum());
        if(hipThreadIdx_x == 0)
        {
            out[colIndex] = alpha * sum;
        }
        __syncthreads();
    }
}

} // anonymous namespace

/***
  Sum Ops
***/

// ReduceFrontSum: columnwise sum
template <>
template <typename T>
void SumReduceDimsOp<HIPContext, true, false>::Compute(int rows,
                                                       int cols,
                                                       const T* in_data,
                                                       T* out_data)
{
    hipLaunchKernelGGL((columnwise_sum_kernel<T>),
                       dim3(std::min(cols, CAFFE_MAXIMUM_NUM_BLOCKS)),
                       dim3(CAFFE_HIP_NUM_THREADS),
                       0,
                       context_.hip_stream(),
                       rows,
                       cols,
                       1.0f,
                       in_data,
                       out_data);
}

// ReduceBackSum: rowwise sum
template <>
template <typename T>
void SumReduceDimsOp<HIPContext, false, false>::Compute(int rows,
                                                        int cols,
                                                        const T* in_data,
                                                        T* out_data)
{
    hipLaunchKernelGGL((rowwise_sum_kernel<T>),
                       dim3(std::min(rows, CAFFE_MAXIMUM_NUM_BLOCKS)),
                       dim3(CAFFE_HIP_NUM_THREADS),
                       0,
                       context_.hip_stream(),
                       rows,
                       cols,
                       1.0f,
                       in_data,
                       out_data);
}

// ReduceFrontSumGradient
template <>
template <typename T>
void SumReduceDimsGradientOp<HIPContext, true, false>::Compute(int rows,
                                                               int cols,
                                                               const T* dYdata,
                                                               T* dXdata)
{
    hipLaunchKernelGGL((columnwise_fill_kernel<T>),
                       dim3(CAFFE_GET_BLOCKS(rows * cols)),
                       dim3(CAFFE_HIP_NUM_THREADS),
                       0,
                       context_.hip_stream(),
                       rows,
                       cols,
                       1.0f,
                       dYdata,
                       dXdata);
}

// ReduceBackSumGradient
template <>
template <typename T>
void SumReduceDimsGradientOp<HIPContext, false, false>::Compute(int rows,
                                                                int cols,
                                                                const T* dYdata,
                                                                T* dXdata)
{
    hipLaunchKernelGGL((rowwise_fill_kernel<T>),
                       dim3(CAFFE_GET_BLOCKS(rows * cols)),
                       dim3(CAFFE_HIP_NUM_THREADS),
                       0,
                       context_.hip_stream(),
                       rows,
                       cols,
                       1.0f,
                       dYdata,
                       dXdata);
}

REGISTER_HIP_OPERATOR(ReduceFrontSum, SumReduceDimsOp<HIPContext, true, false>);
REGISTER_HIP_OPERATOR(ReduceFrontSumGradient, SumReduceDimsGradientOp<HIPContext, true, false>);

REGISTER_HIP_OPERATOR(ReduceBackSum, SumReduceDimsOp<HIPContext, false, false>);
REGISTER_HIP_OPERATOR(ReduceBackSumGradient, SumReduceDimsGradientOp<HIPContext, false, false>);

/***
  Mean Ops
***/

// ReduceFrontMean: columnwise mean
template <>
template <typename T>
void SumReduceDimsOp<HIPContext, true, true>::Compute(int rows,
                                                      int cols,
                                                      const T* in_data,
                                                      T* out_data)
{
    hipLaunchKernelGGL((columnwise_sum_kernel),
                       dim3(std::min(cols, CAFFE_MAXIMUM_NUM_BLOCKS)),
                       dim3(CAFFE_HIP_NUM_THREADS),
                       0,
                       context_.hip_stream(),
                       rows,
                       cols,
                       static_cast<const float>(1.0 / static_cast<float>(rows)),
                       in_data,
                       out_data);
}

// ReduceBackMean: rowwise mean
template <>
template <typename T>
void SumReduceDimsOp<HIPContext, false, true>::Compute(int rows,
                                                       int cols,
                                                       const T* in_data,
                                                       T* out_data)
{
    hipLaunchKernelGGL((rowwise_sum_kernel),
                       dim3(std::min(rows, CAFFE_MAXIMUM_NUM_BLOCKS)),
                       dim3(CAFFE_HIP_NUM_THREADS),
                       0,
                       context_.hip_stream(),
                       rows,
                       cols,
                       static_cast<const float>(1.0 / static_cast<float>(cols)),
                       in_data,
                       out_data);
}

// ReduceFrontMeanGradient
template <>
template <typename T>
void SumReduceDimsGradientOp<HIPContext, true, true>::Compute(int rows,
                                                              int cols,
                                                              const T* dYdata,
                                                              T* dXdata)
{
    hipLaunchKernelGGL((columnwise_fill_kernel<T>),
                       dim3(CAFFE_GET_BLOCKS(rows * cols)),
                       dim3(CAFFE_HIP_NUM_THREADS),
                       0,
                       context_.hip_stream(),
                       rows,
                       cols,
                       static_cast<const float>(1.0 / static_cast<float>(rows)),
                       dYdata,
                       dXdata);
}

// ReduceBackMeanGradient
template <>
template <typename T>
void SumReduceDimsGradientOp<HIPContext, false, true>::Compute(int rows,
                                                               int cols,
                                                               const T* dYdata,
                                                               T* dXdata)
{
    hipLaunchKernelGGL((rowwise_fill_kernel<T>),
                       dim3(CAFFE_GET_BLOCKS(rows * cols)),
                       dim3(CAFFE_HIP_NUM_THREADS),
                       0,
                       context_.hip_stream(),
                       rows,
                       cols,
                       static_cast<const float>(1.0 / static_cast<float>(cols)),
                       dYdata,
                       dXdata);
}

REGISTER_HIP_OPERATOR(ReduceFrontMean, SumReduceDimsOp<HIPContext, true, true>);
REGISTER_HIP_OPERATOR(ReduceFrontMeanGradient, SumReduceDimsGradientOp<HIPContext, true, true>);

REGISTER_HIP_OPERATOR(ReduceBackMean, SumReduceDimsOp<HIPContext, false, true>);
REGISTER_HIP_OPERATOR(ReduceBackMeanGradient, SumReduceDimsGradientOp<HIPContext, false, true>);

/***
  Max Ops
***/

namespace {

__global__ void columnwise_max_kernel(const int rows, const int cols, const float* data, float* out)
{
    typedef cub::BlockReduce<float, CAFFE_HIP_NUM_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    for(int colIndex = hipBlockIdx_x; colIndex < cols; colIndex += hipGridDim_x)
    {
        float mx = FLT_MIN;
        for(int rowIndex = hipThreadIdx_x; rowIndex < rows; rowIndex += hipBlockDim_x)
        {
            mx = fmaxf(mx, data[rowIndex * cols + colIndex]);
        }
        mx = BlockReduce(temp_storage).Reduce(mx, cub::Max());
        if(hipThreadIdx_x == 0)
        {
            out[colIndex] = mx;
        }
        __syncthreads();
    }
}

__global__ void rowwise_max_kernel(const int rows, const int cols, const float* data, float* out)
{
    typedef cub::BlockReduce<float, CAFFE_HIP_NUM_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    for(int rowIndex = hipBlockIdx_x; rowIndex < rows; rowIndex += hipGridDim_x)
    {
        float mx = FLT_MIN;
        for(int colIndex = hipThreadIdx_x; colIndex < cols; colIndex += hipBlockDim_x)
        {
            mx = fmaxf(mx, data[rowIndex * cols + colIndex]);
        }
        mx = BlockReduce(temp_storage).Reduce(mx, cub::Max());
        if(hipThreadIdx_x == 0)
        {
            out[rowIndex] = mx;
        }
        __syncthreads();
    }
}

__global__ void columnwise_max_grad_kernel(const int rows,
                                           const int cols,
                                           const float* dYdata,
                                           const float* Xdata,
                                           const float* Ydata,
                                           float* dXdata)
{
    HIP_1D_KERNEL_LOOP(i, rows * cols)
    {
        int col   = i % cols;
        dXdata[i] = (Xdata[i] == Ydata[col]) * dYdata[col];
    }
}

__global__ void rowwise_max_grad_kernel(const int rows,
                                        const int cols,
                                        const float* dYdata,
                                        const float* Xdata,
                                        const float* Ydata,
                                        float* dXdata)
{
    HIP_1D_KERNEL_LOOP(i, rows * cols)
    {
        int row   = i / cols;
        dXdata[i] = (Xdata[i] == Ydata[row]) * dYdata[row];
    }
}
} // anonymous namespace

// ReduceFrontmax
template <>
void MaxReduceDimsOp<float, HIPContext, true>::Compute(int rows,
                                                       int cols,
                                                       const float* data,
                                                       float* out_data)
{
    hipLaunchKernelGGL((columnwise_max_kernel),
                       dim3(std::min(cols, CAFFE_MAXIMUM_NUM_BLOCKS)),
                       dim3(CAFFE_HIP_NUM_THREADS),
                       0,
                       context_.hip_stream(),
                       rows,
                       cols,
                       data,
                       out_data);
}

// ReduceBackMax
template <>
void MaxReduceDimsOp<float, HIPContext, false>::Compute(int rows,
                                                        int cols,
                                                        const float* data,
                                                        float* out_data)
{
    hipLaunchKernelGGL((rowwise_max_kernel),
                       dim3(std::min(rows, CAFFE_MAXIMUM_NUM_BLOCKS)),
                       dim3(CAFFE_HIP_NUM_THREADS),
                       0,
                       context_.hip_stream(),
                       rows,
                       cols,
                       data,
                       out_data);
}

// ReduceFrontMaxGradient
template <>
void MaxReduceDimsGradientOp<float, HIPContext, true>::Compute(
    int rows, int cols, const float* dYdata, const float* Xdata, const float* Ydata, float* dXdata)
{
    hipLaunchKernelGGL((columnwise_max_grad_kernel),
                       dim3(CAFFE_GET_BLOCKS(rows * cols)),
                       dim3(CAFFE_HIP_NUM_THREADS),
                       0,
                       context_.hip_stream(),
                       rows,
                       cols,
                       dYdata,
                       Xdata,
                       Ydata,
                       dXdata);
}

// ReduceBackMaxGradient
template <>
void MaxReduceDimsGradientOp<float, HIPContext, false>::Compute(
    int rows, int cols, const float* dYdata, const float* Xdata, const float* Ydata, float* dXdata)
{
    hipLaunchKernelGGL((rowwise_max_grad_kernel),
                       dim3(CAFFE_GET_BLOCKS(rows * cols)),
                       dim3(CAFFE_HIP_NUM_THREADS),
                       0,
                       context_.hip_stream(),
                       rows,
                       cols,
                       dYdata,
                       Xdata,
                       Ydata,
                       dXdata);
}

REGISTER_HIP_OPERATOR(ReduceFrontMax, MaxReduceDimsOp<float, HIPContext, true>);
REGISTER_HIP_OPERATOR(ReduceFrontMaxGradient, MaxReduceDimsGradientOp<float, HIPContext, true>);

REGISTER_HIP_OPERATOR(ReduceBackMax, MaxReduceDimsOp<float, HIPContext, false>);
REGISTER_HIP_OPERATOR(ReduceBackMaxGradient, MaxReduceDimsGradientOp<float, HIPContext, false>);

} // namespace caffe2
