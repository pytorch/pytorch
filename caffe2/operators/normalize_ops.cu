#include <algorithm>

#include <cub/block/block_reduce.cuh>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/normalize_l1_op.h"
#include "caffe2/operators/normalize_op.h"

namespace caffe2 {

__global__ void NormalizeKernel(
    const int m,
    const int n,
    const int sf,
    const float* xData,
    float* yData,
    const float kEps) {
  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ BlockReduce::TempStorage temp_storage;

  for (int i = blockIdx.x; i < n; i += gridDim.x) {
    auto base = (i / sf) * sf * m + (i % sf);

    float sum = 0.0;
    __shared__ float norm;
    for (int j = threadIdx.x; j < m; j += blockDim.x) {
      const auto x_ij = xData[base + j * sf];
      sum += x_ij * x_ij;
    }
    float reduce_result = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
      norm = sqrtf(reduce_result);
      norm = fmaxf(norm, kEps);
    }
    __syncthreads();
    for (int j = threadIdx.x; j < m; j += blockDim.x) {
      const auto index = base + j * sf;
      yData[index] = xData[index] / norm;
    }
  }
}

__global__ void NormalizeGradientKernel(
    const int M,
    const int N,
    const int SF,
    const float* in_mat,
    const float* grad_out_mat,
    float* grad_mat,
    const float kEps) {
  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ BlockReduce::TempStorage temp_storage_sum;
  __shared__ BlockReduce::TempStorage temp_storage_norm;
  for (int i = blockIdx.x; i < M; i += gridDim.x) {
    float sum = 0.0;
    float norm = 0.0;
    __shared__ float row_sum;
    __shared__ float row_norm;
    __shared__ float row_norm_3;
    auto base = (i / SF) * SF * N + (i % SF);
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
      int index = base + j * SF;
      sum += in_mat[index] * grad_out_mat[index];
      norm += in_mat[index] * in_mat[index];
    }
    float reduce_result = BlockReduce(temp_storage_sum).Sum(sum);
    float reduce_norm = BlockReduce(temp_storage_norm).Sum(norm);

    if (threadIdx.x == 0) {
      row_sum = reduce_result;
      row_norm = sqrtf(reduce_norm);
      row_norm = fmaxf(row_norm, kEps);
      row_norm_3 = powf(row_norm, 3);
    }
    __syncthreads();
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
      int index = base + j * SF;
      const float x_ij = in_mat[index];
      const float dy_ij = grad_out_mat[index];
      grad_mat[index] = (dy_ij / row_norm) - ((x_ij / row_norm_3) * row_sum);
    }
  }
}

template <>
void NormalizeOp<float, CUDAContext>::DoNormalize(
    const float* xData,
    float* yData,
    const int m,
    const int n,
    const int sf) {
  NormalizeKernel<<<
      std::min(n, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(m, n, sf, xData, yData, kEps_);
}

template <>
bool NormalizeGradientOp<float, CUDAContext>::RunOnDevice() {
  const auto& X = Input(0);
  const auto& dY = Input(1);
  auto* dX = Output(0);
  dX->ResizeLike(X);

  const auto canonical_axis =
      X.canonical_axis_index(OperatorBase::GetSingleArgument<int>("axis", -1));
  int N = X.dim32(canonical_axis);
  int M = X.numel() / N;
  const int SF = X.size_from_dim(canonical_axis + 1);
  NormalizeGradientKernel<<<
      std::min(M, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      M,
      N,
      SF,
      X.data<float>(),
      dY.data<float>(),
      dX->template mutable_data<float>(),
      kEps_);
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
      sum += fabsf(x_ij);
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
      std::min(n, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(m, n, sf, xData, yData);
}

REGISTER_CUDA_OPERATOR(Normalize, NormalizeOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    NormalizeGradient,
    NormalizeGradientOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(NormalizeL1, NormalizeL1Op<float, CUDAContext>);
} // namespace caffe2
