#include "caffe2/operators/swish_op.h"

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void SwishGradientCUDAKernel(
    const int N,
    const T* dY,
    const T* X,
    const T* Y,
    T* dX);

#define DELEGATE_SWISH_GRADIENT_CUDA_KERNEL(T, DeviceExpFunc)                 \
  template <>                                                                 \
  __global__ void SwishGradientCUDAKernel<T>(                                 \
      const int N, const T* dY, const T* X, const T* Y, T* dX) {              \
    const int i = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;          \
    if (i < N) {                                                              \
      dX[i] = dY[i] * (Y[i] + (T(1) - Y[i]) / (T(1) + DeviceExpFunc(-X[i]))); \
    }                                                                         \
  }
DELEGATE_SWISH_GRADIENT_CUDA_KERNEL(float, expf)
DELEGATE_SWISH_GRADIENT_CUDA_KERNEL(double, exp)
#undef DELEGATE_SWISH_GRADIENT_CUDA_KERNEL

} // namespace

#define DELEGATE_SWISH_FUNCTOR(T, DeviceExpFunc)                   \
  template <>                                                      \
  template <>                                                      \
  bool SwishFunctor<CUDAContext>::operator()<T>(                   \
      const int N, const T* X, T* Y, CUDAContext* context) const { \
    if (N > 0) {                                                   \
      thrust::transform(                                           \
          thrust::cuda::par.on(context->cuda_stream()),            \
          X,                                                       \
          X + N,                                                   \
          Y,                                                       \
          [] __device__(const T x) {                               \
            return x / (T(1) + DeviceExpFunc(-x));                 \
          });                                                      \
    }                                                              \
    return true;                                                   \
  }
DELEGATE_SWISH_FUNCTOR(float, expf)
DELEGATE_SWISH_FUNCTOR(double, exp)
#undef DELEGATE_SWISH_FUNCTOR

template <>
template <typename T>
bool SwishGradientOp<CUDAContext>::SwishBackward(
    const int N,
    const T* dY,
    const T* X,
    const T* Y,
    T* dX) {
  if (N > 0) {
    const int M = math::DivUp(N, CAFFE_CUDA_NUM_THREADS);
    SwishGradientCUDAKernel<T>
        <<<M, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
            N, dY, X, Y, dX);
  }
  return true;
}

REGISTER_CUDA_OPERATOR(
    Swish,
    UnaryElementwiseOp<
        TensorTypes<float, double>,
        CUDAContext,
        SwishFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(SwishGradient, SwishGradientOp<CUDAContext>);

} // namespace caffe2
