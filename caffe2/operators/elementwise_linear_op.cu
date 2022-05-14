#include <assert.h>

#include "caffe2/operators/elementwise_linear_op.h"

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/operator_fallback_gpu.h"

#include "caffe2/utils/cub_namespace.cuh"
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


  const auto canonical_axis = X.canonical_axis_index(axis_);
  const int N = X.size_to_dim(canonical_axis);
  const int D = X.size_from_dim(canonical_axis);

  CAFFE_ENFORCE_EQ(a.dim(), 1, a.dim());
  CAFFE_ENFORCE_EQ(a.dim(0), D, a.dim());
  CAFFE_ENFORCE_EQ(b.dim(), 1, b.dim());
  CAFFE_ENFORCE_EQ(b.dim(0), D, b.dim());

  auto* Y = Output(0, X.sizes(), at::dtype<float>());

  ElementwiseLinearKernel<<<
      CAFFE_GET_BLOCKS(N * D),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N,
      D,
      X.data<float>(),
      a.data<float>(),
      b.data<float>(),
      Y->template mutable_data<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

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

  CAFFE_ENFORCE_EQ(a.dim(), 1, a.dim());
  CAFFE_ENFORCE_EQ(a.dim(0), D, a.dim());




  auto* g_X = Output(0, X.sizes(), at::dtype<float>());
  auto * g_a = Output(1, a.sizes(), at::dtype<float>());
  auto * g_b = Output(2, a.sizes(), at::dtype<float>());

  float* g_a_data = g_a->template mutable_data<float>();
  float* g_b_data = g_b->template mutable_data<float>();

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
      g_X->template mutable_data<float>(),
      g_a_data,
      g_b_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

REGISTER_CUDA_OPERATOR(ElementwiseLinear,
                       ElementwiseLinearOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(ElementwiseLinearGradient,
                       ElementwiseLinearGradientOp<float, CUDAContext>);

}  // namespace caffe2
