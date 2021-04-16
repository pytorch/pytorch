#include "caffe2/operators/elu_op.h"

#include <algorithm>
#include <functional>

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void EluCUDAKernel(const int N, const T alpha, const T* X, T* Y);

template <>
__global__ void
EluCUDAKernel<float>(const int N, const float alpha, const float* X, float* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    Y[i] =
        __ldg(X + i) < 0 ? alpha * (expf(__ldg(X + i)) - 1.0f) : __ldg(X + i);
#else
    Y[i] = X[i] < 0 ? alpha * (expf(X[i]) - 1.0f) : X[i];
#endif
  }
}

template <typename T>
__global__ void EluGradientCUDAKernel(
    const int N,
    const T alpha,
    const T* dY,
    const T* Y,
    T* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    dX[i] = __ldg(Y + i) < 0 ? __ldg(dY + i) * (__ldg(Y + i) + alpha)
                             : __ldg(dY + i);
#else
    dX[i] = Y[i] < 0 ? dY[i] * (Y[i] + alpha) : dY[i];
#endif
  }
}

} // namespace

template <>
template <typename T>
bool EluFunctor<CUDAContext>::
operator()(const int N, const T* X, T* Y, CUDAContext* context) const {
  EluCUDAKernel<T>
      <<<CAFFE_GET_BLOCKS(N),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(N, alpha, X, Y);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

template <>
template <typename T>
bool EluGradientFunctor<CUDAContext>::Forward(
    const std::vector<int>& Y_dims,
    const std::vector<int>& /* dY_dims */,
    const T* Y,
    const T* dY,
    T* dX,
    CUDAContext* context) const {
  const int size = std::accumulate(
      Y_dims.cbegin(), Y_dims.cend(), 1, std::multiplies<int>());
  EluGradientCUDAKernel<T>
      <<<CAFFE_GET_BLOCKS(size),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(size, alpha, dY, Y, dX);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

REGISTER_CUDA_OPERATOR(
    Elu,
    UnaryElementwiseWithArgsOp<
        TensorTypes<float>,
        CUDAContext,
        EluFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    EluGradient,
    BinaryElementwiseWithArgsOp<
        TensorTypes<float>,
        CUDAContext,
        EluGradientFunctor<CUDAContext>>);

} // namespace caffe2
