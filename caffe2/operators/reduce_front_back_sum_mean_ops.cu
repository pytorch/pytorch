#include <cub/block/block_reduce.cuh>
#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/reduce_front_back_sum_mean_ops.h"

namespace caffe2 {

namespace {
template <typename T, bool NORMALIZE>
__global__ void columnwise_fill_kernel(
    const int rows,
    const int cols,
    const T* dY,
    const int* lengths,
    T* dX) {
  CUDA_1D_KERNEL_LOOP(i, rows * cols) {
    int row = i / cols;
    int col = i % cols;
    if (lengths == nullptr) {
      dX[i] = NORMALIZE ? dY[col] / rows : dY[col];
    } else if (row < lengths[col]) {
      dX[i] = NORMALIZE ? dY[col] / lengths[col] : dY[col];
    } else {
      dX[i] = 0;
    }
  }
}

template <typename T, bool NORMALIZE>
__global__ void rowwise_fill_kernel(
    const int rows,
    const int cols,
    const T* dY,
    const int* lengths,
    T* dX) {
  CUDA_1D_KERNEL_LOOP(i, rows * cols) {
    int row = i / cols;
    int col = i % cols;
    if (lengths == nullptr) {
      dX[i] = NORMALIZE ? dY[row] / cols : dY[row];
    } else if (col < lengths[row]) {
      dX[i] = NORMALIZE ? dY[row] / lengths[row] : dY[row];
    } else {
      dX[i] = 0;
    }
  }
}

template <typename T, bool NORMALIZE>
__global__ void rowwise_sum_kernel(
    const int rows,
    const int cols,
    const T* data,
    const int* lengths,
    T* out) {
  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  for (int rowIndex = blockIdx.x; rowIndex < rows; rowIndex += gridDim.x) {
    T sum = 0;
    const int rowOffset = rowIndex * cols;
    const int length = lengths == nullptr ? cols : lengths[rowIndex];
    for (int colIndex = threadIdx.x; colIndex < length;
         colIndex += blockDim.x) {
      sum += data[rowOffset + colIndex];
    }
    sum = BlockReduce(temp_storage).Reduce(sum, cub::Sum());
    if (threadIdx.x == 0) {
      out[rowIndex] = NORMALIZE ? sum / length : sum;
    }
    __syncthreads();
  }
}

template <typename T, bool NORMALIZE>
__global__ void columnwise_sum_kernel(
    const int rows,
    const int cols,
    const T* data,
    const int* lengths,
    T* out) {
  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  for (int colIndex = blockIdx.x; colIndex < cols; colIndex += gridDim.x) {
    T sum = 0;
    const int length = lengths == nullptr ? rows : lengths[colIndex];
    for (int rowIndex = threadIdx.x; rowIndex < length;
         rowIndex += blockDim.x) {
      sum += data[rowIndex * cols + colIndex];
    }
    sum = BlockReduce(temp_storage).Reduce(sum, cub::Sum());
    if (threadIdx.x == 0) {
      out[colIndex] = NORMALIZE ? sum / length : sum;
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
    const int* lengths_data,
    T* out_data) {
  columnwise_sum_kernel<T, false>
      <<<std::min(cols, CAFFE_MAXIMUM_NUM_BLOCKS),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(rows, cols, in_data, lengths_data, out_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ReduceBackSum: rowwise sum
template <>
template <typename T>
void SumReduceDimsOp<CUDAContext, false, false>::Compute(
    int rows,
    int cols,
    const T* in_data,
    const int* lengths_data,
    T* out_data) {
  rowwise_sum_kernel<T, false>
      <<<std::min(rows, CAFFE_MAXIMUM_NUM_BLOCKS),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(rows, cols, in_data, lengths_data, out_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ReduceFrontSumGradient
template <>
template <typename T>
void SumReduceDimsGradientOp<CUDAContext, true, false>::Compute(
    int rows,
    int cols,
    const T* dYdata,
    const int* lengths_data,
    T* dXdata) {
  columnwise_fill_kernel<T, false>
      <<<CAFFE_GET_BLOCKS(rows * cols),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(rows, cols, dYdata, lengths_data, dXdata);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ReduceBackSumGradient
template <>
template <typename T>
void SumReduceDimsGradientOp<CUDAContext, false, false>::Compute(
    int rows,
    int cols,
    const T* dYdata,
    const int* lengths_data,
    T* dXdata) {
  rowwise_fill_kernel<T, false>
      <<<CAFFE_GET_BLOCKS(rows * cols),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(rows, cols, dYdata, lengths_data, dXdata);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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
    const int* lengths_data,
    T* out_data) {
  columnwise_sum_kernel<T, true>
      <<<std::min(cols, CAFFE_MAXIMUM_NUM_BLOCKS),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(rows, cols, in_data, lengths_data, out_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ReduceBackMean: rowwise mean
template <>
template <typename T>
void SumReduceDimsOp<CUDAContext, false, true>::Compute(
    int rows,
    int cols,
    const T* in_data,
    const int* lengths_data,
    T* out_data) {
  rowwise_sum_kernel<T, true>
      <<<std::min(rows, CAFFE_MAXIMUM_NUM_BLOCKS),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(rows, cols, in_data, lengths_data, out_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ReduceFrontMeanGradient
template <>
template <typename T>
void SumReduceDimsGradientOp<CUDAContext, true, true>::Compute(
    int rows,
    int cols,
    const T* dYdata,
    const int* lengths_data,
    T* dXdata) {
  columnwise_fill_kernel<T, true>
      <<<CAFFE_GET_BLOCKS(rows * cols),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(rows, cols, dYdata, lengths_data, dXdata);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ReduceBackMeanGradient
template <>
template <typename T>
void SumReduceDimsGradientOp<CUDAContext, false, true>::Compute(
    int rows,
    int cols,
    const T* dYdata,
    const int* lengths_data,
    T* dXdata) {
  rowwise_fill_kernel<T, true>
      <<<CAFFE_GET_BLOCKS(rows * cols),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(rows, cols, dYdata, lengths_data, dXdata);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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

} // namespace caffe2
