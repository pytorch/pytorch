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
#include "caffe2/utils/math.h"

namespace caffe2 {

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
