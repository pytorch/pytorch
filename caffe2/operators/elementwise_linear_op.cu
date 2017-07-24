#include <assert.h>

#include "elementwise_linear_op.h"

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/operator_fallback_gpu.h"

#include <cub/block/block_reduce.cuh>

namespace caffe2 {

namespace {
__global__ void ElementwiseLinearKernel(const int N, const int D,
  const float* X_data, const float* a_data, const float* b_data,
  float* Y_data) {
    CUDA_1D_KERNEL_LOOP(i, N * D) {
      int d = i % D;
      Y_data[i] = X_data[i] * a_data[d] + b_data[d];
    }
}

__global__ void ElementwiseLinearGradientKernel(const int N, const int D,
  const float* g_o_data, const float* X_data, const float* a_data,
  float* g_X_data, float* g_a_data, float* g_b_data) {
  int d = blockIdx.x; // One block per D

  float g_a_sum = 0;
  float g_b_sum = 0;
  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    const float gox = g_o_data[n * D + d];
    g_X_data[n * D + d] = gox * a_data[d];
    g_a_sum += gox * X_data[n * D + d];
    g_b_sum += gox;
  }

  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  float g_a_sum_tot = BlockReduce(temp_storage).Sum(g_a_sum);
  __syncthreads();
  float g_b_sum_tot = BlockReduce(temp_storage).Sum(g_b_sum);

  if (threadIdx.x == 0) {
    g_a_data[d] = g_a_sum_tot;
    g_b_data[d] = g_b_sum_tot;
  }
}

}  // namespace


template<>
bool ElementwiseLinearOp<float, CUDAContext>::RunOnDevice(){
  const auto& X = Input(0);
  const auto& a = Input(1);
  const auto& b = Input(2);
  auto* Y = Output(0);

  const auto canonical_axis = X.canonical_axis_index(axis_);
  const int N = X.size_to_dim(canonical_axis);
  const int D = X.size_from_dim(canonical_axis);

  CAFFE_ENFORCE_EQ(a.ndim(), 1, a.ndim());
  CAFFE_ENFORCE_EQ(a.dim(0), D, a.ndim());
  CAFFE_ENFORCE_EQ(b.ndim(), 1, b.ndim());
  CAFFE_ENFORCE_EQ(b.dim(0), D, b.ndim());

  Y->ResizeLike(X);

  ElementwiseLinearKernel<<<CAFFE_GET_BLOCKS(N * D), CAFFE_CUDA_NUM_THREADS,
                          0, context_.cuda_stream()>>>(
    N, D, X.data<float>(), a.data<float>(), b.data<float>(),
    Y->mutable_data<float>());
  return true;
}


template<>
bool ElementwiseLinearGradientOp<float, CUDAContext>::RunOnDevice(){
  const auto& g_o = Input(0);
  const auto& X = Input(1);
  const auto& a = Input(2);

  const auto canonical_axis = X.canonical_axis_index(axis_);
  const int N = X.size_to_dim(canonical_axis);
  const int D = X.size_from_dim(canonical_axis);

  CAFFE_ENFORCE_EQ(a.ndim(), 1, a.ndim());
  CAFFE_ENFORCE_EQ(a.dim(0), D, a.ndim());

  auto *g_X = Output(0);
  auto *g_a = Output(1);
  auto *g_b = Output(2);
  g_X->ResizeLike(X);
  g_a->ResizeLike(a);
  g_b->ResizeLike(a);

  float* g_a_data = g_a->mutable_data<float>();
  float* g_b_data = g_b->mutable_data<float>();

  ElementwiseLinearGradientKernel<<<
      D,
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N,
      D,
      g_o.data<float>(),
      X.data<float>(),
      a.data<float>(),
      g_X->mutable_data<float>(),
      g_a_data,
      g_b_data);
  return true;
}

REGISTER_CUDA_OPERATOR(ElementwiseLinear,
                       ElementwiseLinearOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(ElementwiseLinearGradient,
                       ElementwiseLinearGradientOp<float, CUDAContext>);

}  // namespace caffe2
