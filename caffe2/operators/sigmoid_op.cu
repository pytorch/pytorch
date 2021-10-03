#include "caffe2/operators/sigmoid_op.h"

#include <algorithm>
#include <functional>

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void SigmoidCUDAKernel(const int N, const T* X, T* Y);

template <>
__global__ void
SigmoidCUDAKernel<float>(const int N, const float* X, float* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    Y[i] = 1.0f / (1.0f + expf(-__ldg(X + i)));
#else
    Y[i] = 1.0f / (1.0f + expf(-X[i]));
#endif
  }
}

template <typename T>
__global__ void
SigmoidGradientCUDAKernel(const int N, const T* dY, const T* Y, T* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
#if __CUDA_ARCH__ >= 350
    dX[i] = __ldg(dY + i) * __ldg(Y + i) * (T(1) - __ldg(Y + i));
#else
    dX[i] = dY[i] * Y[i] * (T(1) - Y[i]);
#endif
  }
}

} // namespace

template <>
template <typename T>
bool SigmoidFunctor<CUDAContext>::
operator()(const int N, const T* X, T* Y, CUDAContext* context) const {
  SigmoidCUDAKernel<T>
      <<<CAFFE_GET_BLOCKS(N),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(N, X, Y);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

template <>
template <typename T>
bool SigmoidGradientFunctor<CUDAContext>::Forward(
    const std::vector<int>& Y_dims,
    const std::vector<int>& /* dY_dims */,
    const T* Y,
    const T* dY,
    T* dX,
    CUDAContext* context) const {
  const int size = std::accumulate(
      Y_dims.cbegin(), Y_dims.cend(), 1, std::multiplies<int>());
  SigmoidGradientCUDAKernel<T>
      <<<CAFFE_GET_BLOCKS(size),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(size, dY, Y, dX);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

REGISTER_CUDA_OPERATOR(
    Sigmoid,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        SigmoidFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    SigmoidGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        SigmoidGradientFunctor<CUDAContext>>);

} // namespace caffe2
