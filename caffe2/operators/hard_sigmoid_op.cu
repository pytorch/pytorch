#include "caffe2/operators/hard_sigmoid_op.h"

#include <algorithm>
#include <functional>

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void HardSigmoidCUDAKernel(
    const int N,
    const T alpha,
    const T beta,
    const T* X,
    T* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    Y[i] = max(T(0), min(T(1), alpha * __ldg(X + i) + beta));
#else
    Y[i] = max(T(0), min(T(1), alpha * X[i] + beta));
#endif
  }
}

template <typename T>
__global__ void HardSigmoidGradientCUDAKernel(
    const int N,
    const T alpha,
    const T* dY,
    const T* Y,
    T* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    dX[i] = (__ldg(Y + i) > T(0) && __ldg(Y + i) < T(1)) ? __ldg(dY + i) * alpha
                                                         : T(0);
#else
    dX[i] = (Y[i] > T(0) && Y[i] < T(1)) ? dY[i] * alpha : T(0);
#endif
  }
}

} // namespace

template <>
template <typename T>
bool HardSigmoidFunctor<CUDAContext>::
operator()(const int N, const T* X, T* Y, CUDAContext* context) const {
  HardSigmoidCUDAKernel<T>
      <<<CAFFE_GET_BLOCKS(N),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(N, alpha, beta, X, Y);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

template <>
template <typename T>
bool HardSigmoidGradientFunctor<CUDAContext>::Forward(
    const std::vector<int>& Y_dims,
    const std::vector<int>& /* dY_dims */,
    const T* Y,
    const T* dY,
    T* dX,
    CUDAContext* context) const {
  const int size = std::accumulate(
      Y_dims.cbegin(), Y_dims.cend(), 1, std::multiplies<int>());
  HardSigmoidGradientCUDAKernel<T>
      <<<CAFFE_GET_BLOCKS(size),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(size, alpha, dY, Y, dX);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

REGISTER_CUDA_OPERATOR(
    HardSigmoid,
    UnaryElementwiseWithArgsOp<
        TensorTypes<float>,
        CUDAContext,
        HardSigmoidFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    HardSigmoidGradient,
    BinaryElementwiseWithArgsOp<
        TensorTypes<float>,
        CUDAContext,
        HardSigmoidGradientFunctor<CUDAContext>>);

} // namespace caffe2
