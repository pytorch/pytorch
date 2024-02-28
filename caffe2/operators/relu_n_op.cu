#include "caffe2/operators/relu_n_op.h"

#include <algorithm>
#include <functional>

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void
ReluNCUDAKernel(const int N, const T threshold, const T* X, T* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    Y[i] = __ldg(X + i) > 0
        ? (__ldg(X + i) < threshold ? __ldg(X + i) : threshold)
        : T(0);
#else
    Y[i] = X[i] > 0 ? (X[i] < threshold ? X[i] : threshold) : T(0);
#endif
  }
}

template <typename T>
__global__ void ReluNGradientCUDAKernel(
    const int N,
    const T threshold,
    const T* dY,
    const T* Y,
    T* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    dX[i] = (__ldg(Y + i) > 0 && __ldg(Y + i) < threshold) ? dY[i] : T(0);
#else
    dX[i] = (Y[i] > 0 && Y[i] < threshold) ? dY[i] : T(0);
#endif
  }
}

} // namespace

template <>
template <typename T>
bool ReluNFunctor<CUDAContext>::
operator()(const int N, const T* X, T* Y, CUDAContext* context) const {
  ReluNCUDAKernel<T>
      <<<CAFFE_GET_BLOCKS(N),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(N, n, X, Y);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

template <>
template <typename T>
bool ReluNGradientFunctor<CUDAContext>::Forward(
    const std::vector<int>& Y_dims,
    const std::vector<int>& /* dY_dims */,
    const T* Y,
    const T* dY,
    T* dX,
    CUDAContext* context) const {
  const int size = std::accumulate(
      Y_dims.cbegin(), Y_dims.cend(), 1, std::multiplies<int>());
  ReluNGradientCUDAKernel<T>
      <<<CAFFE_GET_BLOCKS(size),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(size, n, dY, Y, dX);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

REGISTER_CUDA_OPERATOR(
    ReluN,
    UnaryElementwiseWithArgsOp<
        TensorTypes<float>,
        CUDAContext,
        ReluNFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    ReluNGradient,
    BinaryElementwiseWithArgsOp<
        TensorTypes<float>,
        CUDAContext,
        ReluNGradientFunctor<CUDAContext>>);

} // namespace caffe2
