#include "caffe2/operators/softsign_op.h"

#include <algorithm>
#include <functional>

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {

using c10::cuda::compat::abs;

template <typename T>
inline __host__ __device__ T SquareCUDA(const T x) {
  return x * x;
}

template <typename T>
__global__ void SoftsignCUDAKernel(const int N, const T* X, T* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    Y[i] = __ldg(X + i) / (T(1) + abs(__ldg(X + i)));
#else
    Y[i] = X[i] / (T(1) + abs(X[i]));
#endif
  }
}

template <typename T>
__global__ void
SoftsignGradientCUDAKernel(const int N, const T* dY, const T* X, T* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    dX[i] = __ldg(dY + i) / SquareCUDA(T(1) + abs(__ldg(X + i)));
#else
    dX[i] = dY[i] / SquareCUDA(T(1) + abs(X[i]));
#endif
  }
}

} // namespace

template <>
template <typename T>
bool SoftsignFunctor<CUDAContext>::
operator()(const int N, const T* X, T* Y, CUDAContext* context) const {
  SoftsignCUDAKernel<T>
      <<<CAFFE_GET_BLOCKS(N),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(N, X, Y);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

template <>
template <typename T>
bool SoftsignGradientFunctor<CUDAContext>::Forward(
    const std::vector<int>& X_dims,
    const std::vector<int>& /* dY_dims */,
    const T* X,
    const T* dY,
    T* dX,
    CUDAContext* context) const {
  const int size = std::accumulate(
      X_dims.cbegin(), X_dims.cend(), 1, std::multiplies<int>());
  SoftsignGradientCUDAKernel<T>
      <<<CAFFE_GET_BLOCKS(size),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(size, dY, X, dX);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

REGISTER_CUDA_OPERATOR(
    Softsign,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        SoftsignFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    SoftsignGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        SoftsignGradientFunctor<CUDAContext>>);

} // namespace caffe2
