#include <cub/block/block_reduce.cuh>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/normalize_op.h"

namespace caffe2 {

__global__ void
NormalizeKernel(const int M, const int N, const float* data_in, float* out) {
  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ BlockReduce::TempStorage temp_storage;
  for (int i = blockIdx.x; i < M; i += gridDim.x) {
    float sum_squares = 0.0;
    __shared__ float row_sum_squares;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
      const float x_ij = data_in[i * N + j];
      sum_squares += x_ij * x_ij;
    }
    float reduce_result = BlockReduce(temp_storage).Sum(sum_squares);

    if (threadIdx.x == 0) {
      row_sum_squares = reduce_result;
    }
    __syncthreads();
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
      const int index = i * N + j;
      out[index] = data_in[index] / sqrt(row_sum_squares);
    }
  }
}

__global__ void NormalizeGradientKernel(
    const int M,
    const int N,
    const float* in_mat,
    const float* grad_out_mat,
    float* grad_mat) {
  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ BlockReduce::TempStorage temp_storage_sum;
  __shared__ BlockReduce::TempStorage temp_storage_norm;
  for (int i = blockIdx.x; i < M; i += gridDim.x) {
    float sum = 0.0;
    float norm = 0.0;
    __shared__ float row_sum;
    __shared__ float row_norm;
    __shared__ float row_norm_3;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
      const int index = i * N + j;
      sum += in_mat[index] * grad_out_mat[index];
      norm += in_mat[index] * in_mat[index];
    }
    float reduce_result = BlockReduce(temp_storage_sum).Sum(sum);
    float reduce_norm = BlockReduce(temp_storage_norm).Sum(norm);

    if (threadIdx.x == 0) {
      row_sum = reduce_result;
      row_norm = sqrt(reduce_norm);
      row_norm_3 = pow(row_norm, 3);
    }
    __syncthreads();
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
      const int index = i * N + j;
      const float x_ij = in_mat[index];
      const float dy_ij = grad_out_mat[index];
      grad_mat[index] = (dy_ij / row_norm) - ((x_ij / row_norm_3) * row_sum);
    }
  }
}

template <>
bool NormalizeOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);
  int N = X.dim32(X.ndim() - 1);
  int M = X.size() / N;
  NormalizeKernel<<<
      min(M, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      M, N, X.data<float>(), Y->mutable_data<float>());
  return true;
}

template <>
bool NormalizeGradientOp<float, CUDAContext>::RunOnDevice() {
  const auto& X = Input(0);
  const auto& dY = Input(1);
  auto* dX = Output(0);
  dX->ResizeLike(X);
  int N = X.dim32(X.ndim() - 1);
  int M = X.size() / N;
  NormalizeGradientKernel<<<
      min(M, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      M,
      N,
      X.data<float>(),
      dY.data<float>(),
      dX->mutable_data<float>());
  return true;
}

namespace {
__global__ void NormalizeL1Kernel(
    const int m,
    const int n,
    const int sf,
    const float* xData,
    float* yData) {
  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ BlockReduce::TempStorage temp_storage;

  for (int i = blockIdx.x; i < n; i += gridDim.x) {
    auto base = (i / sf) * sf * m + (i % sf);

    float sum = 0.0;
    __shared__ float norm;
    for (int j = threadIdx.x; j < m; j += blockDim.x) {
      const auto x_ij = xData[base + j * sf];
      sum += abs(x_ij);
    }
    float reduce_result = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
      norm = reduce_result;
    }
    __syncthreads();
    if (norm != 0) {
      for (int j = threadIdx.x; j < m; j += blockDim.x) {
        const auto index = base + j * sf;
        yData[index] = xData[index] / norm;
      }
    }
  }
}
} // namespace

template <>
void NormalizeL1Op<float, CUDAContext>::DoNormalize(
    const float* xData,
    float* yData,
    const int m,
    const int n,
    const int sf) {
  NormalizeL1Kernel<<<
      min(n, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(m, n, sf, xData, yData);
}

REGISTER_CUDA_OPERATOR(Normalize, NormalizeOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    NormalizeGradient,
    NormalizeGradientOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(NormalizeL1, NormalizeL1Op<float, CUDAContext>);
} // namespace
