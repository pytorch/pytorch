#include "caffe2/operators/rms_norm_op.h"

#include <vector>

#include <thrust/tuple.h>

#include "c10/cuda/CUDAMathCompat.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/math/reduce.cuh"
#include "caffe2/utils/math/utils.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void RowwiseRMSCUDAKernel(int64_t N, T eps, const T* X, T* rrms) {
  __shared__ typename BlockReduce<T>::TempStorage rms_storage;
  const int64_t i = blockIdx.x;
  T sum = 0;
  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    const int64_t index = i * N + j;
    sum += X[index] * X[index];
  }
  sum = BlockReduce<T>(rms_storage).Sum(sum);
  if (threadIdx.x == 0) {
    rrms[i] =
        c10::cuda::compat::rsqrt(sum / static_cast<T>(N) + static_cast<T>(eps));
  }
}

template <typename T>
__global__ void RMSNormForwardCUDAKernel(
    int64_t N,
    const T* X,
    const T* gamma,
    const T* beta,
    const T* rrms,
    T* Y) {
  const int64_t i = blockIdx.x;
  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    const int64_t index = i * N + j;
    Y[index] = rrms[i] * X[index] * gamma[j] + beta[j];
  }
}

template <typename T>
__global__ void ComputeInternalGradientsCUDAKernel(
    int64_t N,
    const T* dY,
    const T* X,
    const T* gamma,
    const T* rrms,
    T* c2) {
  __shared__ typename BlockReduce<T>::TempStorage ds_storage;
  const int64_t i = blockIdx.x;
  T ds = 0;
  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    const int index = i * N + j;
    ds += dY[index] * X[index] * gamma[j];
  }
  ds = BlockReduce<T>(ds_storage).Sum(ds);
  if (threadIdx.x == 0) {
    c2[i] = -ds * math::utils::Cube<T>(rrms[i]) / static_cast<T>(N);
  }
}

template <typename T>
__global__ void RMSNormBackwardCUDAKernel(
    int64_t N,
    const T* dY,
    const T* X,
    const T* gamma,
    const T* c1,
    const T* c2,
    T* dX) {
  const int64_t i = blockIdx.x;
  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    const int64_t index = i * N + j;
    dX[index] = c1[i] * dY[index] * gamma[j] + c2[i] * X[index];
  }
}

// Assume the batch size will not be very large, direct implementation is the
// most efficient one.
template <typename T>
__global__ void GammaBetaBackwardCUDAKernel(
    int64_t M,
    int64_t N,
    const T* dY,
    const T* X,
    const T* rrms,
    T* dg,
    T* db) {
  const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < N) {
    T sum1 = 0;
    T sum2 = 0;
    for (int64_t i = 0; i < M; ++i) {
      const int64_t index = i * N + j;
      sum1 += dY[index] * X[index] * rrms[i];
      sum2 += dY[index];
    }
    dg[j] = sum1;
    db[j] = sum2;
  }
}

} // namespace

template <>
template <typename T>
bool RMSNormOp<CUDAContext>::DoRunWithType() {
  const auto& X = Input(0);
  const auto& gamma = Input(1);
  const auto& beta = Input(2);
  auto* Y = Output(0, X.sizes(), at::dtype<T>());
  CAFFE_ENFORCE_GE(X.dim(), 2, "RMSNorm requires input dim >= 2.");
  const int canonical_axis = X.canonical_axis_index(axis_);
  const std::vector<int64_t> rms_dims(
      X.sizes().cbegin(), X.sizes().cbegin() + canonical_axis);
  auto* rrms = Output(1, rms_dims, at::dtype<T>());
  const int64_t M = X.size_to_dim(canonical_axis);
  const int64_t N = X.size_from_dim(canonical_axis);
  CAFFE_ENFORCE_EQ(gamma.numel(), N);
  CAFFE_ENFORCE_EQ(beta.numel(), N);

  const T* X_data = X.template data<T>();
  const T* gamma_data = gamma.template data<T>();
  const T* beta_data = beta.template data<T>();
  T* Y_data = Y->template data<T>();
  T* rrms_data = rrms->template data<T>();

  if (M > 0) {
    RowwiseRMSCUDAKernel<T>
        <<<M, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
            N, static_cast<T>(eps_), X_data, rrms_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    RMSNormForwardCUDAKernel<T>
        <<<M, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
            N, X_data, gamma_data, beta_data, rrms_data, Y_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return true;
}

template <>
template <typename T>
void RMSNormGradientOp<CUDAContext>::RMSNormBackward(
    int64_t M,
    int64_t N,
    const T* dY,
    const T* X,
    const T* gamma,
    const T* rrms,
    T* dX) {
  ReinitializeTensor(
      &c2_, {M}, at::dtype<T>().device(CUDAContext::GetDeviceType()));
  T* c2_data = c2_.mutable_data<T>();
  ComputeInternalGradientsCUDAKernel<T>
      <<<M, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          N, dY, X, gamma, rrms, c2_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  RMSNormBackwardCUDAKernel<T>
      <<<M, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          N, dY, X, gamma, rrms, c2_data, dX);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <>
template <typename T>
void RMSNormGradientOp<CUDAContext>::GammaBetaBackward(
    int64_t M,
    int64_t N,
    const T* dY,
    const T* X,
    const T* rrms,
    T* dgamma,
    T* dbeta) {
  const int64_t B = math::DivUp<int64_t>(N, CAFFE_CUDA_NUM_THREADS);
  GammaBetaBackwardCUDAKernel<T>
      <<<B, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          M, N, dY, X, rrms, dgamma, dbeta);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

REGISTER_CUDA_OPERATOR(RMSNorm, RMSNormOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(RMSNormGradient, RMSNormGradientOp<CUDAContext>);

} // namespace caffe2
