#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/softmax_op.h"


namespace caffe2 {

#define SOFTMAX_NUM_THREADS 128

namespace {
// The softmax kernel. This kernel has to be called with the number of threads
// per block being no more than SOFTMAX_NUM_THREADS.
__global__ void softmax_kernel(
    const int dim, const float* data, float* out) {
  // For the softmax kernel, each block is a data example.
  data += blockIdx.x * dim;
  out += blockIdx.x * dim;
  const int idx = threadIdx.x;
  __shared__ float reduction_buffer[SOFTMAX_NUM_THREADS];
  float tmp;

  // A two-level reduction to get the max.
  tmp = -FLT_MAX;
  for (int i = idx; i < dim; i += blockDim.x) {
    tmp = fmaxf(data[i], tmp);
  }
  reduction_buffer[idx] = tmp;
  __syncthreads();
  if (idx == 0) {
    tmp = reduction_buffer[0];
    for (int i = 1; i < blockDim.x; ++i) {
      tmp = fmaxf(reduction_buffer[i], tmp);
    }
    reduction_buffer[0] = tmp;
  }
  __syncthreads();
  // compute sum with a two-level reduction.
  float maxval = reduction_buffer[0];
  reduction_buffer[idx] = 0;
  for (int i = idx; i < dim; i += blockDim.x) {
    tmp = __expf(data[i] - maxval);
    reduction_buffer[idx] += tmp;
    out[i] = tmp;
  }
  __syncthreads();
  if (idx == 0) {
    tmp = reduction_buffer[0];
    for (int i = 1; i < blockDim.x; ++i) {
      tmp += reduction_buffer[i];
    }
    reduction_buffer[0] = tmp;
  }
  __syncthreads();
  // Compute the softmax;
  tmp = reduction_buffer[0];
  for (int i = idx; i < dim; i += blockDim.x) {
    out[i] /= tmp;
  }
}

// The softmax gradient kernel. This kernel has to be called with the number of
// threads per block being no more than SOFTMAX_NUM_THREADS.
__global__ void softmax_gradient_kernel(
    const int dim, const float* Y, const float* dY, float* dX) {
  Y += blockIdx.x * dim;
  dY += blockIdx.x * dim;
  dX += blockIdx.x * dim;
  const int idx = threadIdx.x;
  __shared__ float reduction_buffer[SOFTMAX_NUM_THREADS];
  float tmp;

  // A two-level reduction to compute the inner products.
  tmp = 0;
  for (int i = idx; i < dim; i += blockDim.x) {
    tmp += dY[i] * Y[i];
  }
  reduction_buffer[idx] = tmp;
  __syncthreads();
  if (idx == 0) {
    tmp = reduction_buffer[0];
    for (int i = 1; i < blockDim.x; ++i) tmp += reduction_buffer[i];
    reduction_buffer[0] = tmp;
  }
  __syncthreads();
  // Compute gradient.
  tmp = reduction_buffer[0];
  for (int i = idx; i < dim; i += blockDim.x) {
    dX[i] = Y[i] * (dY[i] - tmp);
  }
}
}  // namespace

// Implementation for the CPU context.
template <>
bool SoftmaxOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  CAFFE_DCHECK_EQ(X.ndim(), 2);
  int N = X.dim(0);
  int D = X.dim(1);
  Y->ReshapeLike(X);
  softmax_kernel<<<N, SOFTMAX_NUM_THREADS, 0, device_context_.cuda_stream()>>>(
      D, X.data<float>(), Y->mutable_data<float>());
  return true;
}

// Implementation for the CPU context.
template <>
bool SoftmaxGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  CAFFE_DCHECK_EQ(Y.ndim(), 2);
  int N = Y.dim(0);
  int D = Y.dim(1);
  CAFFE_DCHECK_EQ(dY.dim(0), N);
  CAFFE_DCHECK_EQ(dY.dim(1), D);
  dX->ReshapeLike(Y);
  softmax_gradient_kernel<<<N, SOFTMAX_NUM_THREADS, 0,
                            device_context_.cuda_stream()>>>(
      D, Y.data<float>(), dY.data<float>(), dX->mutable_data<float>());
  return true;
}

REGISTER_CUDA_OPERATOR(Softmax, SoftmaxOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(SoftmaxGradient, SoftmaxGradientOp<float, CUDAContext>);
}  // namespace caffe2
