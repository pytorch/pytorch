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
#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/reduction_front_back_ops.h"

namespace caffe2 {

namespace {
template <typename T>
__global__ void columnwise_fill_kernel(
    const int rows,
    const int cols,
    const float alpha,
    const T* dY,
    T* dX) {
  CUDA_1D_KERNEL_LOOP(i, rows * cols) {
    dX[i] = dY[i % cols] * alpha;
  }
}

template <typename T>
__global__ void rowwise_fill_kernel(
    const int rows,
    const int cols,
    const float alpha,
    const T* dY,
    T* dX) {
  CUDA_1D_KERNEL_LOOP(i, rows * cols) {
    dX[i] = dY[i / cols] * alpha;
  }
}

template <typename T>
__global__ void rowwise_sum_kernel(
    const int rows,
    const int cols,
    const float alpha,
    const T* data,
    T* out) {
  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  for (int rowIndex = blockIdx.x; rowIndex < rows; rowIndex += gridDim.x) {
    T sum = 0;
    const int rowOffset = rowIndex * cols;
    for (int colIndex = threadIdx.x; colIndex < cols; colIndex += blockDim.x) {
      sum += data[rowOffset + colIndex];
    }
    sum = BlockReduce(temp_storage).Reduce(sum, cub::Sum());
    if (threadIdx.x == 0) {
      out[rowIndex] = alpha * sum;
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void columnwise_sum_kernel(
    const int rows,
    const int cols,
    const float alpha,
    const T* data,
    T* out) {
  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  for (int colIndex = blockIdx.x; colIndex < cols; colIndex += gridDim.x) {
    T sum = 0;
    for (int rowIndex = threadIdx.x; rowIndex < rows; rowIndex += blockDim.x) {
      sum += data[rowIndex * cols + colIndex];
    }
    sum = BlockReduce(temp_storage).Reduce(sum, cub::Sum());
    if (threadIdx.x == 0) {
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
void SumReduceDimsOp<CUDAContext, true, false>::Compute(
    int rows,
    int cols,
    const T* in_data,
    T* out_data) {
  columnwise_sum_kernel<T>
      <<<std::min(cols, CAFFE_MAXIMUM_NUM_BLOCKS),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(rows, cols, 1.0, in_data, out_data);
}

// ReduceBackSum: rowwise sum
template <>
template <typename T>
void SumReduceDimsOp<CUDAContext, false, false>::Compute(
    int rows,
    int cols,
    const T* in_data,
    T* out_data) {
  rowwise_sum_kernel<T>
      <<<std::min(rows, CAFFE_MAXIMUM_NUM_BLOCKS),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(rows, cols, 1.0, in_data, out_data);
}

// ReduceFrontSumGradient
template <>
template <typename T>
void SumReduceDimsGradientOp<CUDAContext, true, false>::Compute(
    int rows,
    int cols,
    const T* dYdata,
    T* dXdata) {
  columnwise_fill_kernel<T>
      <<<CAFFE_GET_BLOCKS(rows * cols),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(rows, cols, 1.0, dYdata, dXdata);
}

// ReduceBackSumGradient
template <>
template <typename T>
void SumReduceDimsGradientOp<CUDAContext, false, false>::Compute(
    int rows,
    int cols,
    const T* dYdata,
    T* dXdata) {
  rowwise_fill_kernel<T>
      <<<CAFFE_GET_BLOCKS(rows * cols),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(rows, cols, 1.0, dYdata, dXdata);
}

REGISTER_CUDA_OPERATOR(
    ReduceFrontSum,
    SumReduceDimsOp<CUDAContext, true, false>);
REGISTER_CUDA_OPERATOR(
    ReduceFrontSumGradient,
    SumReduceDimsGradientOp<CUDAContext, true, false>);

REGISTER_CUDA_OPERATOR(
    ReduceBackSum,
    SumReduceDimsOp<CUDAContext, false, false>);
REGISTER_CUDA_OPERATOR(
    ReduceBackSumGradient,
    SumReduceDimsGradientOp<CUDAContext, false, false>);

/***
  Mean Ops
***/

// ReduceFrontMean: columnwise mean
template <>
template <typename T>
void SumReduceDimsOp<CUDAContext, true, true>::Compute(
    int rows,
    int cols,
    const T* in_data,
    T* out_data) {
  columnwise_sum_kernel<<<
      std::min(cols, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(rows, cols, 1.0 / rows, in_data, out_data);
}

// ReduceBackMean: rowwise mean
template <>
template <typename T>
void SumReduceDimsOp<CUDAContext, false, true>::Compute(
    int rows,
    int cols,
    const T* in_data,
    T* out_data) {
  rowwise_sum_kernel<<<
      std::min(rows, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(rows, cols, 1.0 / cols, in_data, out_data);
}

// ReduceFrontMeanGradient
template <>
template <typename T>
void SumReduceDimsGradientOp<CUDAContext, true, true>::Compute(
    int rows,
    int cols,
    const T* dYdata,
    T* dXdata) {
  columnwise_fill_kernel<T>
      <<<CAFFE_GET_BLOCKS(rows * cols),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(rows, cols, 1.0 / rows, dYdata, dXdata);
}

// ReduceBackMeanGradient
template <>
template <typename T>
void SumReduceDimsGradientOp<CUDAContext, false, true>::Compute(
    int rows,
    int cols,
    const T* dYdata,
    T* dXdata) {
  rowwise_fill_kernel<T>
      <<<CAFFE_GET_BLOCKS(rows * cols),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(rows, cols, 1.0 / cols, dYdata, dXdata);
}

REGISTER_CUDA_OPERATOR(
    ReduceFrontMean,
    SumReduceDimsOp<CUDAContext, true, true>);
REGISTER_CUDA_OPERATOR(
    ReduceFrontMeanGradient,
    SumReduceDimsGradientOp<CUDAContext, true, true>);

REGISTER_CUDA_OPERATOR(
    ReduceBackMean,
    SumReduceDimsOp<CUDAContext, false, true>);
REGISTER_CUDA_OPERATOR(
    ReduceBackMeanGradient,
    SumReduceDimsGradientOp<CUDAContext, false, true>);

/***
  Max Ops
***/

namespace {

__global__ void columnwise_max_kernel(
    const int rows,
    const int cols,
    const float* data,
    float* out) {
  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  for (int colIndex = blockIdx.x; colIndex < cols; colIndex += gridDim.x) {
    float mx = FLT_MIN;
    for (int rowIndex = threadIdx.x; rowIndex < rows; rowIndex += blockDim.x) {
      mx = max(mx, data[rowIndex * cols + colIndex]);
    }
    mx = BlockReduce(temp_storage).Reduce(mx, cub::Max());
    if (threadIdx.x == 0) {
      out[colIndex] = mx;
    }
    __syncthreads();
  }
}

__global__ void rowwise_max_kernel(
    const int rows,
    const int cols,
    const float* data,
    float* out) {
  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  for (int rowIndex = blockIdx.x; rowIndex < rows; rowIndex += gridDim.x) {
    float mx = FLT_MIN;
    for (int colIndex = threadIdx.x; colIndex < cols; colIndex += blockDim.x) {
      mx = max(mx, data[rowIndex * cols + colIndex]);
    }
    mx = BlockReduce(temp_storage).Reduce(mx, cub::Max());
    if (threadIdx.x == 0) {
      out[rowIndex] = mx;
    }
    __syncthreads();
  }
}

__global__ void columnwise_max_grad_kernel(
    const int rows,
    const int cols,
    const float* dYdata,
    const float* Xdata,
    const float* Ydata,
    float* dXdata) {
  CUDA_1D_KERNEL_LOOP(i, rows * cols) {
    int col = i % cols;
    dXdata[i] = (Xdata[i] == Ydata[col]) * dYdata[col];
  }
}

__global__ void rowwise_max_grad_kernel(
    const int rows,
    const int cols,
    const float* dYdata,
    const float* Xdata,
    const float* Ydata,
    float* dXdata) {
  CUDA_1D_KERNEL_LOOP(i, rows * cols) {
    int row = i / cols;
    dXdata[i] = (Xdata[i] == Ydata[row]) * dYdata[row];
  }
}
} // anonymous namespace

// ReduceFrontmax
template <>
void MaxReduceDimsOp<float, CUDAContext, true>::Compute(
    int rows,
    int cols,
    const float* data,
    float* out_data) {
  columnwise_max_kernel<<<
      std::min(cols, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(rows, cols, data, out_data);
}

// ReduceBackMax
template <>
void MaxReduceDimsOp<float, CUDAContext, false>::Compute(
    int rows,
    int cols,
    const float* data,
    float* out_data) {
  rowwise_max_kernel<<<
      std::min(rows, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(rows, cols, data, out_data);
}

// ReduceFrontMaxGradient
template <>
void MaxReduceDimsGradientOp<float, CUDAContext, true>::Compute(
    int rows,
    int cols,
    const float* dYdata,
    const float* Xdata,
    const float* Ydata,
    float* dXdata) {
  columnwise_max_grad_kernel<<<
      CAFFE_GET_BLOCKS(rows * cols),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(rows, cols, dYdata, Xdata, Ydata, dXdata);
}

// ReduceBackMaxGradient
template <>
void MaxReduceDimsGradientOp<float, CUDAContext, false>::Compute(
    int rows,
    int cols,
    const float* dYdata,
    const float* Xdata,
    const float* Ydata,
    float* dXdata) {
  rowwise_max_grad_kernel<<<
      CAFFE_GET_BLOCKS(rows * cols),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(rows, cols, dYdata, Xdata, Ydata, dXdata);
}

REGISTER_CUDA_OPERATOR(
    ReduceFrontMax,
    MaxReduceDimsOp<float, CUDAContext, true>);
REGISTER_CUDA_OPERATOR(
    ReduceFrontMaxGradient,
    MaxReduceDimsGradientOp<float, CUDAContext, true>);

REGISTER_CUDA_OPERATOR(
    ReduceBackMax,
    MaxReduceDimsOp<float, CUDAContext, false>);
REGISTER_CUDA_OPERATOR(
    ReduceBackMaxGradient,
    MaxReduceDimsGradientOp<float, CUDAContext, false>);

} // namespace caffe2
