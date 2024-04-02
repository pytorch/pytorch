#include "caffe2/operators/gelu_op.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif // _MSC_VER

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

// y = x * P(X <= x) where X ~ N(0, 1)
template <typename T>
__global__ void GeluCUDAKernel(const int N, const T* X, T* Y);

#define DELEGATE_GELU_CUDA_KERNEL(T, CdfNormFunc)                        \
  template <>                                                            \
  __global__ void GeluCUDAKernel<T>(const int N, const T* X, T* Y) {     \
    const int index = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x; \
    if (index < N) {                                                     \
      Y[index] = X[index] * CdfNormFunc(X[index]);                       \
    }                                                                    \
  }
DELEGATE_GELU_CUDA_KERNEL(float, normcdff)
#undef DELEGATE_GELU_CUDA_KERNEL

// y = 0.5x * (1 + tanh(sqrt(2/Pi) * (x + 0.044715x^3)))
template <typename T>
__global__ void FastGeluCUDAKernel(const int N, const T* X, T* Y);

#define DELEGATE_FAST_GELU_CUDA_KERNEL(T, FMAFunc, TanhFunc)             \
  template <>                                                            \
  __global__ void FastGeluCUDAKernel(const int N, const T* X, T* Y) {    \
    constexpr T kAlpha = M_2_SQRTPI * M_SQRT1_2;                         \
    const int index = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x; \
    if (index < N) {                                                     \
      Y[index] = static_cast<T>(0.5) *                                   \
          FMAFunc(X[index],                                              \
                  TanhFunc(                                              \
                      kAlpha *                                           \
                      FMAFunc(                                           \
                          gelu_utils::kFastCoeff,                        \
                          math::utils::Cube<T>(X[index]),                \
                          X[index])),                                    \
                  X[index]);                                             \
    }                                                                    \
  }
DELEGATE_FAST_GELU_CUDA_KERNEL(float, fmaf, tanhf)
#undef DELEGATE_FAST_GELU_CUDA_KERNEL

template <typename T>
__global__ void
GeluGradientCUDAKernel(const int N, const T* dY, const T* X, T* dX);

#define DELEGATE_GELU_GRADIENT_CUDA_KERNEL(T, FMAFunc, CdfNormFunc, ExpFunc) \
  template <>                                                                \
  __global__ void GeluGradientCUDAKernel<T>(                                 \
      const int N, const T* dY, const T* X, T* dX) {                         \
    constexpr T kAlpha = M_2_SQRTPI * M_SQRT1_2 * T(0.5);                    \
    const int index = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;     \
    if (index < N) {                                                         \
      dX[index] = dY[index] *                                                \
          FMAFunc(kAlpha * X[index],                                         \
                  ExpFunc(-X[index] * X[index] * static_cast<T>(0.5)),       \
                  CdfNormFunc(X[index]));                                    \
    }                                                                        \
  }
DELEGATE_GELU_GRADIENT_CUDA_KERNEL(float, fmaf, normcdff, expf)
#undef DELEGATE_GELU_GRADIENT_CUDA_KERNEL

template <typename T>
__global__ void
FastGeluGradientCUDAKernel(const int N, const T* dY, const T* X, T* dX);

#define DELEGATE_FAST_GELU_GRADIENT_CUDA_KERNEL(T, FMAFunc, TanhFunc)    \
  template <>                                                            \
  __global__ void FastGeluGradientCUDAKernel<T>(                         \
      const int N, const T* dY, const T* X, T* dX) {                     \
    constexpr T kAlpha = M_2_SQRTPI * M_SQRT1_2;                         \
    constexpr T kBeta = kAlpha * gelu_utils::kFastCoeff * T(3);          \
    const int index = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x; \
    if (index < N) {                                                     \
      const T y = TanhFunc(                                              \
          kAlpha *                                                       \
          FMAFunc(                                                       \
              gelu_utils::kFastCoeff,                                    \
              math::utils::Cube<T>(X[index]),                            \
              X[index]));                                                \
      dX[index] = FMAFunc(                                               \
                      FMAFunc(-X[index], y * y, X[index]),               \
                      FMAFunc(kBeta, X[index] * X[index], kAlpha),       \
                      T(1) + y) *                                        \
          dY[index] * static_cast<T>(0.5);                               \
    }                                                                    \
  }
DELEGATE_FAST_GELU_GRADIENT_CUDA_KERNEL(float, fmaf, tanhf)
#undef DELEGATE_FAST_GELU_GRADIENT_CUDA_KERNEL

} // namespace

template <>
template <typename T>
bool GeluFunctor<CUDAContext>::
operator()(const int N, const T* X, T* Y, CUDAContext* context) const {
  const int M = math::DivUp(N, CAFFE_CUDA_NUM_THREADS);
  if (fast_gelu) {
    FastGeluCUDAKernel<T>
        <<<M, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(N, X, Y);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    GeluCUDAKernel<T>
        <<<M, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(N, X, Y);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return true;
}

template <>
template <typename T>
bool GeluGradientFunctor<CUDAContext>::Forward(
    const std::vector<int>& dY_dims,
    const std::vector<int>& /* X_dims */,
    const T* dY,
    const T* X,
    T* dX,
    CUDAContext* context) const {
  const int N = std::accumulate(
      dY_dims.cbegin(), dY_dims.cend(), 1, std::multiplies<int>());
  const int M = math::DivUp(N, CAFFE_CUDA_NUM_THREADS);
  if (fast_gelu) {
    // y = 0.5x * (1 + tanh(sqrt(2/Pi) * (x + 0.044715x^3)))
    FastGeluGradientCUDAKernel<T>
        <<<M, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
            N, dY, X, dX);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    // y = x * P(X <= x) where X ~ N(0, 1)
    GeluGradientCUDAKernel<T>
        <<<M, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
            N, dY, X, dX);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return true;
}

REGISTER_CUDA_OPERATOR(Gelu, GeluOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(GeluGradient, GeluGradientOp<CUDAContext>);

} // namespace caffe2
