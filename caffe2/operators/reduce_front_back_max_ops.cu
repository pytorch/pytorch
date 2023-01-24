#include <cub/block/block_reduce.cuh>
#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/reduce_front_back_max_ops.h"
#include "caffe2/utils/cub_namespace.cuh"

#if defined(USE_ROCM)
#include <cfloat>
#endif

namespace caffe2 {

/***
  Max Ops
***/

namespace {

__global__ void columnwise_max_kernel(
    const int rows,
    const int cols,
    const float* data,
    const int* lengths,
    float* out) {
  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  for (int colIndex = blockIdx.x; colIndex < cols; colIndex += gridDim.x) {
    float mx = FLT_MIN;
    const int length = lengths == nullptr ? rows : lengths[colIndex];
    for (int rowIndex = threadIdx.x; rowIndex < length;
         rowIndex += blockDim.x) {
      mx = fmaxf(mx, data[rowIndex * cols + colIndex]);
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
    const int* lengths,
    float* out) {
  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  for (int rowIndex = blockIdx.x; rowIndex < rows; rowIndex += gridDim.x) {
    float mx = FLT_MIN;
    const int length = lengths == nullptr ? cols : lengths[rowIndex];
    for (int colIndex = threadIdx.x; colIndex < length;
         colIndex += blockDim.x) {
      mx = fmaxf(mx, data[rowIndex * cols + colIndex]);
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
    const int* lengths,
    float* dXdata) {
  CUDA_1D_KERNEL_LOOP(i, rows * cols) {
    int col = i % cols;
    int row = i / cols;
    if (lengths != nullptr && row >= lengths[col]) {
      dXdata[i] = 0.0f;
    } else {
      dXdata[i] = (Xdata[i] == Ydata[col]) * dYdata[col];
    }
  }
}

__global__ void rowwise_max_grad_kernel(
    const int rows,
    const int cols,
    const float* dYdata,
    const float* Xdata,
    const float* Ydata,
    const int* lengths,
    float* dXdata) {
  CUDA_1D_KERNEL_LOOP(i, rows * cols) {
    int col = i % cols;
    int row = i / cols;
    if (lengths != nullptr && col >= lengths[row]) {
      dXdata[i] = 0.0f;
    } else {
      dXdata[i] = (Xdata[i] == Ydata[row]) * dYdata[row];
    }
  }
}
} // anonymous namespace

// ReduceFrontmax
template <>
void MaxReduceDimsOp<float, CUDAContext, true>::Compute(
    int rows,
    int cols,
    const float* data,
    const int32_t* lengths_data,
    float* out_data) {
  columnwise_max_kernel<<<
      std::min(cols, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(rows, cols, data, lengths_data, out_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ReduceBackMax
template <>
void MaxReduceDimsOp<float, CUDAContext, false>::Compute(
    int rows,
    int cols,
    const float* data,
    const int32_t* lengths_data,
    float* out_data) {
  rowwise_max_kernel<<<
      std::min(rows, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(rows, cols, data, lengths_data, out_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ReduceFrontMaxGradient
template <>
void MaxReduceDimsGradientOp<float, CUDAContext, true>::Compute(
    int rows,
    int cols,
    const float* dYdata,
    const float* Xdata,
    const float* Ydata,
    const int32_t* lengths_data,
    float* dXdata) {
  columnwise_max_grad_kernel<<<
      CAFFE_GET_BLOCKS(rows * cols),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      rows, cols, dYdata, Xdata, Ydata, lengths_data, dXdata);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ReduceBackMaxGradient
template <>
void MaxReduceDimsGradientOp<float, CUDAContext, false>::Compute(
    int rows,
    int cols,
    const float* dYdata,
    const float* Xdata,
    const float* Ydata,
    const int* lengths_data,
    float* dXdata) {
  rowwise_max_grad_kernel<<<
      CAFFE_GET_BLOCKS(rows * cols),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      rows, cols, dYdata, Xdata, Ydata, lengths_data, dXdata);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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
