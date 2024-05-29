#include "caffe2/utils/math/elementwise.h"

#include <type_traits>

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/conversions.h"
#include "caffe2/utils/math/half_utils.h"
#include "caffe2/utils/math/utils.h"

namespace caffe2 {
namespace math {

namespace {

template <typename T>
__global__ void SinCosCUDAKernel(const int N, const T* X, T* S, T* C) {
  const int i = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (i < N) {
#if __CUDA_ARCH__ >= 350 || defined(USE_ROCM)
    c10::cuda::compat::sincos(__ldg(X + i), S + i, C + i);
#else
    c10::cuda::compat::sincos(X[i], S + i, C + i);
#endif
  }
}

#if defined(USE_ROCM)

template <typename TAlpha, typename TData>
__global__ void AxpyCUDAKernel(
    const std::int64_t N,
    const TAlpha alpha,
    const TData* X,
    TData* Y) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) *
          static_cast<int64_t>(CAFFE_CUDA_NUM_THREADS) +
      static_cast<int64_t>(threadIdx.x);
  if (index < N) {
    Y[index] += static_cast<TData>(alpha) * __ldg(X + index);
  }
}

template <typename TAlpha, typename TData>
__global__ void AxpyCUDAKernel(
    const std::int64_t N,
    const TAlpha* alpha,
    const TData* X,
    TData* Y) {
  __shared__ TData a;
  if (threadIdx.x == 0) {
    a = static_cast<TData>(__ldg(alpha));
  }
  __syncthreads();
  const int64_t index = static_cast<int64_t>(blockIdx.x) *
          static_cast<int64_t>(CAFFE_CUDA_NUM_THREADS) +
      static_cast<int64_t>(threadIdx.x);
  if (index < N) {
    Y[index] += a * __ldg(X + index);
  }
}

#define DELEGATE_HALF_AXPY_CUDA_KERNEL(TAlpha, FMAFunc)                \
  template <>                                                          \
  __global__ void AxpyCUDAKernel<TAlpha, at::Half>(                    \
      const std::int64_t N,                                            \
      const TAlpha alpha,                                              \
      const at::Half* X,                                               \
      at::Half* Y) {                                                   \
    const int64_t index = static_cast<int64_t>(blockIdx.x) *           \
            static_cast<int64_t>(CAFFE_CUDA_NUM_THREADS) +             \
        static_cast<int64_t>(threadIdx.x);                             \
    if (index < N) {                                                   \
      Y[index] = convert::To<TAlpha, at::Half>(FMAFunc(                \
          alpha,                                                       \
          convert::To<at::Half, TAlpha>(X[index]),                     \
          convert::To<at::Half, TAlpha>(Y[index])));                   \
    }                                                                  \
  }                                                                    \
  template <>                                                          \
  __global__ void AxpyCUDAKernel<TAlpha, at::Half>(                    \
      const std::int64_t N,                                            \
      const TAlpha* alpha,                                             \
      const at::Half* X,                                               \
      at::Half* Y) {                                                   \
    __shared__ TAlpha a;                                               \
    if (threadIdx.x == 0) {                                            \
      a = __ldg(alpha);                                                \
    }                                                                  \
    __syncthreads();                                                   \
    const int64_t index = static_cast<int64_t>(blockIdx.x) *           \
            static_cast<int64_t>(CAFFE_CUDA_NUM_THREADS) +             \
        static_cast<int64_t>(threadIdx.x);                             \
    if (index < N) {                                                   \
      Y[index] = convert::To<TAlpha, at::Half>(FMAFunc(                \
          a,                                                           \
          convert::To<at::Half, TAlpha>(X[index]),                     \
          convert::To<at::Half, TAlpha>(Y[index])));                   \
    }                                                                  \
  }
DELEGATE_HALF_AXPY_CUDA_KERNEL(float, fmaf)
#undef DELEGATE_HALF_AXPY_CUDA_KERNEL

#endif // USE_ROCM

template <typename TAlpha, typename TData>
__global__ void AxpbyCUDAKernel(
    const std::int64_t N,
    const TAlpha alpha,
    const TData* X,
    const TAlpha beta,
    TData* Y);

template <typename TAlpha, typename TData>
__global__ void AxpbyCUDAKernel(
    const std::int64_t N,
    const TAlpha* alpha,
    const TData* X,
    const TAlpha* beta,
    TData* Y);

#define DELEGATE_AXPBY_CUDA_KERNEL(TAlpha, TData, FMAFunc)             \
  template <>                                                          \
  __global__ void AxpbyCUDAKernel<TAlpha, TData>(                      \
      const std::int64_t N,                                            \
      const TAlpha alpha,                                              \
      const TData* X,                                                  \
      const TAlpha beta,                                               \
      TData* Y) {                                                      \
    const int64_t index = static_cast<int64_t>(blockIdx.x) *           \
            static_cast<int64_t>(CAFFE_CUDA_NUM_THREADS) +             \
        static_cast<int64_t>(threadIdx.x);                             \
    if (index < N) {                                                   \
      Y[index] = FMAFunc(                                              \
          static_cast<TData>(alpha),                                   \
          X[index],                                                    \
          static_cast<TData>(beta) * Y[index]);                        \
    }                                                                  \
  }                                                                    \
  template <>                                                          \
  __global__ void AxpbyCUDAKernel<TAlpha, TData>(                      \
      const std::int64_t N,                                            \
      const TAlpha* alpha,                                             \
      const TData* X,                                                  \
      const TAlpha* beta,                                              \
      TData* Y) {                                                      \
    __shared__ TData a;                                                \
    __shared__ TData b;                                                \
    if (threadIdx.x == 0) {                                            \
      a = static_cast<TData>(*alpha);                                  \
      b = static_cast<TData>(*beta);                                   \
    }                                                                  \
    __syncthreads();                                                   \
    const int64_t index = static_cast<int64_t>(blockIdx.x) *           \
            static_cast<int64_t>(CAFFE_CUDA_NUM_THREADS) +             \
        static_cast<int64_t>(threadIdx.x);                             \
    if (index < N) {                                                   \
      Y[index] = FMAFunc(a, X[index], b * Y[index]);                   \
    }                                                                  \
  }
DELEGATE_AXPBY_CUDA_KERNEL(float, float, fmaf)
DELEGATE_AXPBY_CUDA_KERNEL(float, double, fma)
#undef DELEGATE_AXPBY_CUDA_KERNEL

#define DELEGATE_HALF_AXPBY_CUDA_KERNEL(TAlpha, FMAFunc)               \
  template <>                                                          \
  __global__ void AxpbyCUDAKernel<TAlpha, at::Half>(                   \
      const std::int64_t N,                                            \
      const TAlpha alpha,                                              \
      const at::Half* X,                                               \
      const TAlpha beta,                                               \
      at::Half* Y) {                                                   \
    const int64_t index = static_cast<int64_t>(blockIdx.x) *           \
            static_cast<int64_t>(CAFFE_CUDA_NUM_THREADS) +             \
        static_cast<int64_t>(threadIdx.x);                             \
    if (index < N) {                                                   \
      Y[index] = convert::To<TAlpha, at::Half>(FMAFunc(                \
          alpha,                                                       \
          convert::To<at::Half, TAlpha>(X[index]),                     \
          beta * convert::To<at::Half, TAlpha>(Y[index])));            \
    }                                                                  \
  }                                                                    \
  template <>                                                          \
  __global__ void AxpbyCUDAKernel<TAlpha, at::Half>(                   \
      const std::int64_t N,                                            \
      const TAlpha* alpha,                                             \
      const at::Half* X,                                               \
      const TAlpha* beta,                                              \
      at::Half* Y) {                                                   \
    __shared__ TAlpha a;                                               \
    __shared__ TAlpha b;                                               \
    if (threadIdx.x == 0) {                                            \
      a = *alpha;                                                      \
      b = *beta;                                                       \
    }                                                                  \
    __syncthreads();                                                   \
    const int64_t index = static_cast<int64_t>(blockIdx.x) *           \
            static_cast<int64_t>(CAFFE_CUDA_NUM_THREADS) +             \
        static_cast<int64_t>(threadIdx.x);                             \
    if (index < N) {                                                   \
      Y[index] = convert::To<TAlpha, at::Half>(FMAFunc(                \
          a,                                                           \
          convert::To<at::Half, TAlpha>(X[index]),                     \
          b * convert::To<at::Half, TAlpha>(Y[index])));               \
    }                                                                  \
  }
DELEGATE_HALF_AXPBY_CUDA_KERNEL(float, fmaf)
#undef DELEGATE_HALF_AXPBY_CUDA_KERNEL

template <typename TAlpha, typename TData>
__global__ void ScaleCUDAKernel(
    const std::int64_t N,
    const TAlpha alpha,
    const TData* X,
    TData* Y);

template <typename TAlpha, typename TData>
__global__ void ScaleCUDAKernel(
    const std::int64_t N,
    const TAlpha* alpha,
    const TData* X,
    TData* Y);

#define CAFFE2_SPECIALIZED_SCALE_CUDA_KERNEL(TAlpha, TData)                  \
  template <>                                                                \
  __global__ void ScaleCUDAKernel<TAlpha, TData>(                            \
      const std::int64_t N, const TAlpha alpha, const TData* X, TData* Y) {  \
    const int64_t index = static_cast<int64_t>(blockIdx.x) *                 \
            static_cast<int64_t>(CAFFE_CUDA_NUM_THREADS) +                   \
        static_cast<int64_t>(threadIdx.x);                                   \
    if (index < N) {                                                         \
      Y[index] = static_cast<TData>(alpha) * X[index];                       \
    }                                                                        \
  }                                                                          \
  template <>                                                                \
  __global__ void ScaleCUDAKernel<TAlpha, TData>(                            \
      const std::int64_t N, const TAlpha* alpha, const TData* X, TData* Y) { \
    __shared__ TData a;                                                      \
    if (threadIdx.x == 0) {                                                  \
      a = static_cast<TData>(*alpha);                                        \
    }                                                                        \
    __syncthreads();                                                         \
    const int64_t index = static_cast<int64_t>(blockIdx.x) *                 \
            static_cast<int64_t>(CAFFE_CUDA_NUM_THREADS) +                   \
        static_cast<int64_t>(threadIdx.x);                                   \
    if (index < N) {                                                         \
      Y[index] = a * X[index];                                               \
    }                                                                        \
  }
CAFFE2_SPECIALIZED_SCALE_CUDA_KERNEL(float, float)
CAFFE2_SPECIALIZED_SCALE_CUDA_KERNEL(double, double)
CAFFE2_SPECIALIZED_SCALE_CUDA_KERNEL(float, double)
CAFFE2_SPECIALIZED_SCALE_CUDA_KERNEL(std::int32_t, std::int32_t)
CAFFE2_SPECIALIZED_SCALE_CUDA_KERNEL(std::int64_t, std::int64_t)
#undef CAFFE2_SPECIALIZED_SCALE_CUDA_KERNEL

#define CAFFE2_SPECIALIZED_HALF_SCALE_CUDA_KERNEL(TAlpha)              \
  template <>                                                          \
  __global__ void ScaleCUDAKernel<TAlpha, at::Half>(                   \
      const std::int64_t N,                                            \
      const TAlpha alpha,                                              \
      const at::Half* X,                                               \
      at::Half* Y) {                                                   \
    const int64_t index = static_cast<int64_t>(blockIdx.x) *           \
            static_cast<int64_t>(CAFFE_CUDA_NUM_THREADS) +             \
        static_cast<int64_t>(threadIdx.x);                             \
    if (index < N) {                                                   \
      Y[index] = convert::To<TAlpha, at::Half>(                        \
          alpha * convert::To<at::Half, TAlpha>(X[index]));            \
    }                                                                  \
  }                                                                    \
  template <>                                                          \
  __global__ void ScaleCUDAKernel<TAlpha, at::Half>(                   \
      const std::int64_t N,                                            \
      const TAlpha* alpha,                                             \
      const at::Half* X,                                               \
      at::Half* Y) {                                                   \
    __shared__ TAlpha a;                                               \
    if (threadIdx.x == 0) {                                            \
      a = *alpha;                                                      \
    }                                                                  \
    __syncthreads();                                                   \
    const int64_t index = static_cast<int64_t>(blockIdx.x) * \
            static_cast<int64_t>(CAFFE_CUDA_NUM_THREADS) +        \
        static_cast<int64_t>(threadIdx.x);                        \
    if (index < N) {                                                   \
      Y[index] = convert::To<TAlpha, at::Half>(                        \
          a * convert::To<at::Half, TAlpha>(X[index]));                \
    }                                                                  \
  }
CAFFE2_SPECIALIZED_HALF_SCALE_CUDA_KERNEL(float)
#undef CAFFE2_SPECIALIZED_HALF_SCALE_CUDA_KERNEL

} // namespace

#define CAFFE2_SPECIALIZED_CUDA_SET(T)                                    \
  template <>                                                             \
  CAFFE2_CUDA_EXPORT void Set<T, CUDAContext>(                            \
      const std::int64_t N, const T alpha, T* Y, CUDAContext* context) {  \
    if (N == 0) {                                                         \
      return;                                                             \
    }                                                                     \
    if (alpha == T(0)) {                                                  \
      C10_CUDA_CHECK(cudaMemsetAsync(Y, 0, sizeof(T) * N, context->cuda_stream()));       \
    } else {                                                              \
      thrust::fill(                                                       \
          thrust::cuda::par.on(context->cuda_stream()), Y, Y + N, alpha); \
    }                                                                     \
  }
CAFFE2_SPECIALIZED_CUDA_SET(bool)
CAFFE2_SPECIALIZED_CUDA_SET(char)
CAFFE2_SPECIALIZED_CUDA_SET(std::int8_t)
CAFFE2_SPECIALIZED_CUDA_SET(std::int16_t)
CAFFE2_SPECIALIZED_CUDA_SET(std::int32_t)
CAFFE2_SPECIALIZED_CUDA_SET(std::int64_t)
CAFFE2_SPECIALIZED_CUDA_SET(std::uint8_t)
CAFFE2_SPECIALIZED_CUDA_SET(std::uint16_t)
CAFFE2_SPECIALIZED_CUDA_SET(float)
CAFFE2_SPECIALIZED_CUDA_SET(double)
CAFFE2_SPECIALIZED_CUDA_SET(at::Half)
CAFFE2_SPECIALIZED_CUDA_SET(at::BFloat16)
#undef CAFFE2_SPECIALIZED_CUDA_SET

#define DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(T, Func, DeviceFunc) \
  template <>                                                    \
  CAFFE2_CUDA_EXPORT void Func<T, CUDAContext>(                  \
      const int N, const T* X, T* Y, CUDAContext* context) {     \
    if (N > 0) {                                                 \
      thrust::transform(                                         \
          thrust::cuda::par.on(context->cuda_stream()),          \
          X,                                                     \
          X + N,                                                 \
          Y,                                                     \
          [] __device__(const T x) { return DeviceFunc(x); });   \
    }                                                            \
  }
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Exp, expf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Log, logf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Log1p, log1pf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Sin, sinf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Asin, asinf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Cos, cosf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Acos, acosf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Tan, tanf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Atan, atanf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Sinh, sinhf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Cosh, coshf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Tanh, tanhf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Abs, fabsf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Inv, utils::Inv<float>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(double, Inv, utils::Inv<double>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Sqr, utils::Square<float>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Sqrt, sqrtf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Rsqrt, rsqrtf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(
    std::int32_t,
    Cube,
    utils::Cube<std::int32_t>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(
    std::int64_t,
    Cube,
    utils::Cube<std::int64_t>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Cube, utils::Cube<float>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(double, Cube, utils::Cube<double>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Cbrt, cbrtf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Erf, erff)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(double, Erf, erf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, CdfNorm, normcdff)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(double, CdfNorm, normcdf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(bool, Not, utils::Not<bool>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(
    std::int32_t,
    Neg,
    utils::Negate<std::int32_t>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(
    std::int64_t,
    Neg,
    utils::Negate<std::int64_t>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Neg, utils::Negate<float>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(double, Neg, utils::Negate<double>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(
    std::int32_t,
    Sign,
    utils::Sign<std::int32_t>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(
    std::int64_t,
    Sign,
    utils::Sign<std::int64_t>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Sign, utils::Sign<float>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(double, Sign, utils::Sign<double>)
#undef DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION

#define DELEGATE_CUDA_POWX(T, DeviceFunc)                               \
  template <>                                                           \
  CAFFE2_CUDA_EXPORT void Powx<T, CUDAContext>(                         \
      const int N, const T* A, const T b, T* Y, CUDAContext* context) { \
    thrust::transform(                                                  \
        thrust::cuda::par.on(context->cuda_stream()),                   \
        A,                                                              \
        A + N,                                                          \
        Y,                                                              \
        [b] __device__(const T x) { return DeviceFunc(x, b); });        \
  }
DELEGATE_CUDA_POWX(float, powf)
#undef DELEGATE_CUDA_POWX

#define CAFFE2_SPECIALIZED_CUDA_SINCOS(T)                             \
  template <>                                                         \
  CAFFE2_CUDA_EXPORT void SinCos<T, CUDAContext>(                     \
      const int N, const T* X, T* S, T* C, CUDAContext* context) {    \
    if (N > 0) {                                                      \
      const int K = DivUp(N, CAFFE_CUDA_NUM_THREADS);                 \
      SinCosCUDAKernel<T>                                             \
          <<<K, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>( \
              N, X, S, C);                                            \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                 \
    }                                                                 \
  }
CAFFE2_SPECIALIZED_CUDA_SINCOS(float)
CAFFE2_SPECIALIZED_CUDA_SINCOS(double)
#undef CAFFE2_SPECIALIZED_CUDA_SINCOS

#define DELEGATE_CUDA_SCALE(T, CuBLASFunc)                                   \
  template <>                                                                \
  CAFFE2_CUDA_EXPORT void Scale<T, T, CUDAContext>(                          \
      const std::int64_t N,                                                  \
      const T alpha,                                                         \
      const T* X,                                                            \
      T* Y,                                                                  \
      CUDAContext* context) {                                                \
    if (N == 0) {                                                            \
      return;                                                                \
    }                                                                        \
    if (Y == X) {                                                            \
      CUBLAS_ENFORCE(cublasSetPointerMode(                                   \
          context->cublas_handle(), CUBLAS_POINTER_MODE_HOST));              \
      CUBLAS_ENFORCE(CuBLASFunc(context->cublas_handle(), N, &alpha, Y, 1)); \
    } else {                                                                 \
      const std::int64_t M = DivUp<std::int64_t>(N, CAFFE_CUDA_NUM_THREADS); \
      ScaleCUDAKernel<T, T>                                                  \
          <<<M, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(        \
              N, alpha, X, Y);                                               \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                        \
    }                                                                        \
  }                                                                          \
  template <>                                                                \
  CAFFE2_CUDA_EXPORT void Scale<T, T, CUDAContext>(                          \
      const std::int64_t N,                                                  \
      const T* alpha,                                                        \
      const T* X,                                                            \
      T* Y,                                                                  \
      CUDAContext* context) {                                                \
    if (N == 0) {                                                            \
      return;                                                                \
    }                                                                        \
    if (Y == X) {                                                            \
      CUBLAS_ENFORCE(cublasSetPointerMode(                                   \
          context->cublas_handle(), CUBLAS_POINTER_MODE_DEVICE));            \
      CUBLAS_ENFORCE(CuBLASFunc(context->cublas_handle(), N, alpha, Y, 1));  \
    } else {                                                                 \
      const std::int64_t M = DivUp<std::int64_t>(N, CAFFE_CUDA_NUM_THREADS); \
      ScaleCUDAKernel<T, T>                                                  \
          <<<M, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(        \
              N, alpha, X, Y);                                               \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                        \
    }                                                                        \
  }
DELEGATE_CUDA_SCALE(float, cublasSscal)
DELEGATE_CUDA_SCALE(double, cublasDscal)
#undef DELEGATE_CUDA_SCALE

#if !defined(USE_ROCM)

#define DELEGATE_CUDA_SCALE_EX(                                              \
    TAlpha, TData, kAlphaType, kDataType, kExecutionType)                    \
  template <>                                                                \
  CAFFE2_CUDA_EXPORT void Scale<TAlpha, TData, CUDAContext>(                 \
      const std::int64_t N,                                                  \
      const TAlpha alpha,                                                    \
      const TData* X,                                                        \
      TData* Y,                                                              \
      CUDAContext* context) {                                                \
    if (N == 0) {                                                            \
      return;                                                                \
    }                                                                        \
    if (Y == X) {                                                            \
      CUBLAS_ENFORCE(cublasSetPointerMode(                                   \
          context->cublas_handle(), CUBLAS_POINTER_MODE_HOST));              \
      CUBLAS_ENFORCE(cublasScalEx(                                           \
          context->cublas_handle(),                                          \
          N,                                                                 \
          &alpha,                                                            \
          kAlphaType,                                                        \
          Y,                                                                 \
          kDataType,                                                         \
          1,                                                                 \
          kExecutionType));                                                  \
    } else {                                                                 \
      const std::int64_t M = DivUp<std::int64_t>(N, CAFFE_CUDA_NUM_THREADS); \
      ScaleCUDAKernel<TAlpha, TData>                                         \
          <<<M, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(        \
              N, alpha, X, Y);                                               \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                        \
    }                                                                        \
  }                                                                          \
  template <>                                                                \
  CAFFE2_CUDA_EXPORT void Scale<TAlpha, TData, CUDAContext>(                 \
      const std::int64_t N,                                                  \
      const TAlpha* alpha,                                                   \
      const TData* X,                                                        \
      TData* Y,                                                              \
      CUDAContext* context) {                                                \
    if (N == 0) {                                                            \
      return;                                                                \
    }                                                                        \
    if (Y == X) {                                                            \
      CUBLAS_ENFORCE(cublasSetPointerMode(                                   \
          context->cublas_handle(), CUBLAS_POINTER_MODE_DEVICE));            \
      CUBLAS_ENFORCE(cublasScalEx(                                           \
          context->cublas_handle(),                                          \
          N,                                                                 \
          alpha,                                                             \
          kAlphaType,                                                        \
          Y,                                                                 \
          kDataType,                                                         \
          1,                                                                 \
          kExecutionType));                                                  \
    } else {                                                                 \
      const std::int64_t M = DivUp<std::int64_t>(N, CAFFE_CUDA_NUM_THREADS); \
      ScaleCUDAKernel<TAlpha, TData>                                         \
          <<<M, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(        \
              N, alpha, X, Y);                                               \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                        \
    }                                                                        \
  }
DELEGATE_CUDA_SCALE_EX(float, double, CUDA_R_32F, CUDA_R_64F, CUDA_R_64F)
DELEGATE_CUDA_SCALE_EX(float, at::Half, CUDA_R_32F, CUDA_R_16F, CUDA_R_32F)
#undef DELEGATE_CUDA_SCALE_EX

#endif // USE_ROCM

#define CAFFE2_SPECIALIZED_CUDA_SCALE(TAlpha, TData)                         \
  template <>                                                                \
  CAFFE2_CUDA_EXPORT void Scale<TAlpha, TData, CUDAContext>(                 \
      const std::int64_t N,                                                  \
      const TAlpha alpha,                                                    \
      const TData* X,                                                        \
      TData* Y,                                                              \
      CUDAContext* context) {                                                \
    if (N > 0) {                                                             \
      const std::int64_t M = DivUp<std::int64_t>(N, CAFFE_CUDA_NUM_THREADS); \
      ScaleCUDAKernel<TAlpha, TData>                                         \
          <<<M, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(        \
              N, alpha, X, Y);                                               \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                        \
    }                                                                        \
  }                                                                          \
  template <>                                                                \
  CAFFE2_CUDA_EXPORT void Scale<TAlpha, TData, CUDAContext>(                 \
      const std::int64_t N,                                                  \
      const TAlpha* alpha,                                                   \
      const TData* X,                                                        \
      TData* Y,                                                              \
      CUDAContext* context) {                                                \
    if (N > 0) {                                                             \
      const std::int64_t M = DivUp<std::int64_t>(N, CAFFE_CUDA_NUM_THREADS); \
      ScaleCUDAKernel<TAlpha, TData>                                         \
          <<<M, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(        \
              N, *alpha, X, Y);                                              \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                        \
    }                                                                        \
  }
CAFFE2_SPECIALIZED_CUDA_SCALE(std::int32_t, std::int32_t)
CAFFE2_SPECIALIZED_CUDA_SCALE(std::int64_t, std::int64_t)

#if defined(USE_ROCM)
CAFFE2_SPECIALIZED_CUDA_SCALE(float, double)
CAFFE2_SPECIALIZED_CUDA_SCALE(float, at::Half)
#endif // USE_ROCM
#undef CAFFE2_SPECIALIZED_CUDA_SCALE

#define DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(T, Func, DeviceFunc)        \
  template <>                                                            \
  CAFFE2_CUDA_EXPORT void Func<T, CUDAContext>(                          \
      const int N, const T* A, const T* B, T* C, CUDAContext* context) { \
    if (N > 0) {                                                         \
      thrust::transform(                                                 \
          thrust::cuda::par.on(context->cuda_stream()),                  \
          A,                                                             \
          A + N,                                                         \
          B,                                                             \
          C,                                                             \
          DeviceFunc);                                                   \
    }                                                                    \
  }
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(
    std::int32_t,
    Add,
    thrust::plus<std::int32_t>())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(
    std::int64_t,
    Add,
    thrust::plus<std::int64_t>())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(float, Add, thrust::plus<float>())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(double, Add, thrust::plus<double>())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(at::Half, Add, utils::HalfAddFunctor())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(
    std::int32_t,
    Sub,
    thrust::minus<std::int32_t>())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(
    std::int64_t,
    Sub,
    thrust::minus<std::int64_t>())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(float, Sub, thrust::minus<float>())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(double, Sub, thrust::minus<double>())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(at::Half, Sub, utils::HalfSubFunctor())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(
    std::int32_t,
    Mul,
    thrust::multiplies<std::int32_t>())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(
    std::int64_t,
    Mul,
    thrust::multiplies<std::int64_t>())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(float, Mul, thrust::multiplies<float>())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(double, Mul, thrust::multiplies<double>())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(at::Half, Mul, utils::HalfMulFunctor())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(
    std::int32_t,
    Div,
    thrust::divides<std::int32_t>())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(
    std::int64_t,
    Div,
    thrust::divides<std::int64_t>())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(float, Div, thrust::divides<float>())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(double, Div, thrust::divides<double>())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(at::Half, Div, utils::HalfDivFunctor())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(float, Min, thrust::minimum<float>())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(double, Min, thrust::minimum<double>())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(float, Max, thrust::maximum<float>())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(double, Max, thrust::maximum<double>())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(bool, And, thrust::logical_and<bool>())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(bool, Or, thrust::logical_or<bool>())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(bool, Xor, thrust::bit_xor<bool>())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(bool, BitwiseAnd, thrust::bit_and<bool>())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(
    std::int32_t,
    BitwiseAnd,
    thrust::bit_and<std::int32_t>())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(
    std::int64_t,
    BitwiseAnd,
    thrust::bit_and<std::int64_t>())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(bool, BitwiseOr, thrust::bit_or<bool>())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(
    std::int32_t,
    BitwiseOr,
    thrust::bit_or<std::int32_t>())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(
    std::int64_t,
    BitwiseOr,
    thrust::bit_or<std::int64_t>())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(bool, BitwiseXor, thrust::bit_xor<bool>())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(
    std::int32_t,
    BitwiseXor,
    thrust::bit_xor<std::int32_t>())
DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION(
    std::int64_t,
    BitwiseXor,
    thrust::bit_xor<std::int64_t>())
#undef DELEGATE_SIMPLE_CUDA_BINARY_FUNCTION

#define DELEGATE_SIMPLE_CUDA_COMPARE_FUNCTION(T, Func, DeviceComp)          \
  template <>                                                               \
  CAFFE2_CUDA_EXPORT void Func<T, CUDAContext>(                             \
      const int N, const T* A, const T* B, bool* C, CUDAContext* context) { \
    if (N > 0) {                                                            \
      thrust::transform(                                                    \
          thrust::cuda::par.on(context->cuda_stream()),                     \
          A,                                                                \
          A + N,                                                            \
          B,                                                                \
          C,                                                                \
          DeviceComp);                                                      \
    }                                                                       \
  }
DELEGATE_SIMPLE_CUDA_COMPARE_FUNCTION(bool, EQ, thrust::equal_to<bool>())
DELEGATE_SIMPLE_CUDA_COMPARE_FUNCTION(
    std::int32_t,
    EQ,
    thrust::equal_to<std::int32_t>())
DELEGATE_SIMPLE_CUDA_COMPARE_FUNCTION(
    std::int64_t,
    EQ,
    thrust::equal_to<std::int64_t>())
DELEGATE_SIMPLE_CUDA_COMPARE_FUNCTION(float, EQ, thrust::equal_to<float>())
DELEGATE_SIMPLE_CUDA_COMPARE_FUNCTION(double, EQ, thrust::equal_to<double>())
DELEGATE_SIMPLE_CUDA_COMPARE_FUNCTION(bool, NE, thrust::not_equal_to<bool>())
DELEGATE_SIMPLE_CUDA_COMPARE_FUNCTION(
    std::int32_t,
    NE,
    thrust::not_equal_to<std::int32_t>())
DELEGATE_SIMPLE_CUDA_COMPARE_FUNCTION(
    std::int64_t,
    NE,
    thrust::not_equal_to<std::int64_t>())
DELEGATE_SIMPLE_CUDA_COMPARE_FUNCTION(float, NE, thrust::not_equal_to<float>())
DELEGATE_SIMPLE_CUDA_COMPARE_FUNCTION(
    double,
    NE,
    thrust::not_equal_to<double>())
DELEGATE_SIMPLE_CUDA_COMPARE_FUNCTION(bool, LT, thrust::less<bool>())
DELEGATE_SIMPLE_CUDA_COMPARE_FUNCTION(
    std::int32_t,
    LT,
    thrust::less<std::int32_t>())
DELEGATE_SIMPLE_CUDA_COMPARE_FUNCTION(
    std::int64_t,
    LT,
    thrust::less<std::int64_t>())
DELEGATE_SIMPLE_CUDA_COMPARE_FUNCTION(float, LT, thrust::less<float>())
DELEGATE_SIMPLE_CUDA_COMPARE_FUNCTION(double, LT, thrust::less<double>())
DELEGATE_SIMPLE_CUDA_COMPARE_FUNCTION(bool, LE, thrust::less_equal<bool>())
DELEGATE_SIMPLE_CUDA_COMPARE_FUNCTION(
    std::int32_t,
    LE,
    thrust::less_equal<std::int32_t>())
DELEGATE_SIMPLE_CUDA_COMPARE_FUNCTION(
    std::int64_t,
    LE,
    thrust::less_equal<std::int64_t>())
DELEGATE_SIMPLE_CUDA_COMPARE_FUNCTION(float, LE, thrust::less_equal<float>())
DELEGATE_SIMPLE_CUDA_COMPARE_FUNCTION(double, LE, thrust::less_equal<double>())
DELEGATE_SIMPLE_CUDA_COMPARE_FUNCTION(bool, GT, thrust::greater<bool>())
DELEGATE_SIMPLE_CUDA_COMPARE_FUNCTION(
    std::int32_t,
    GT,
    thrust::greater<std::int32_t>())
DELEGATE_SIMPLE_CUDA_COMPARE_FUNCTION(
    std::int64_t,
    GT,
    thrust::greater<std::int64_t>())
DELEGATE_SIMPLE_CUDA_COMPARE_FUNCTION(float, GT, thrust::greater<float>())
DELEGATE_SIMPLE_CUDA_COMPARE_FUNCTION(double, GT, thrust::greater<double>())
DELEGATE_SIMPLE_CUDA_COMPARE_FUNCTION(bool, GE, thrust::greater_equal<bool>())
DELEGATE_SIMPLE_CUDA_COMPARE_FUNCTION(
    std::int32_t,
    GE,
    thrust::greater_equal<std::int32_t>())
DELEGATE_SIMPLE_CUDA_COMPARE_FUNCTION(
    std::int64_t,
    GE,
    thrust::greater_equal<std::int64_t>())
DELEGATE_SIMPLE_CUDA_COMPARE_FUNCTION(float, GE, thrust::greater_equal<float>())
DELEGATE_SIMPLE_CUDA_COMPARE_FUNCTION(
    double,
    GE,
    thrust::greater_equal<double>())
#undef DELEGATE_SIMPLE_CUDA_COMPARE_FUNCTION

#define DELEGATE_CUDA_AXPY(T, CuBLASFunc)                             \
  template <>                                                         \
  CAFFE2_CUDA_EXPORT void Axpy<T, T, CUDAContext>(                    \
      const std::int64_t N,                                           \
      const T alpha,                                                  \
      const T* X,                                                     \
      T* Y,                                                           \
      CUDAContext* context) {                                         \
    CUBLAS_ENFORCE(cublasSetPointerMode(                              \
        context->cublas_handle(), CUBLAS_POINTER_MODE_HOST));         \
    CUBLAS_ENFORCE(                                                   \
        CuBLASFunc(context->cublas_handle(), N, &alpha, X, 1, Y, 1)); \
  }                                                                   \
  template <>                                                         \
  CAFFE2_CUDA_EXPORT void Axpy<T, T, CUDAContext>(                    \
      const std::int64_t N,                                           \
      const T* alpha,                                                 \
      const T* X,                                                     \
      T* Y,                                                           \
      CUDAContext* context) {                                         \
    CUBLAS_ENFORCE(cublasSetPointerMode(                              \
        context->cublas_handle(), CUBLAS_POINTER_MODE_DEVICE));       \
    CUBLAS_ENFORCE(                                                   \
        cublasSaxpy(context->cublas_handle(), N, alpha, X, 1, Y, 1)); \
  }
DELEGATE_CUDA_AXPY(float, cublasSaxpy)
#undef DELEGATE_CUDA_AXPY

#if !defined(USE_ROCM)

#define DELEGATE_CUDA_AXPY_EX(                                  \
    TAlpha, TData, kAlphaType, kDataType, kExecutionType)       \
  template <>                                                   \
  CAFFE2_CUDA_EXPORT void Axpy<TAlpha, TData, CUDAContext>(     \
      const std::int64_t N,                                     \
      const TAlpha alpha,                                       \
      const TData* X,                                           \
      TData* Y,                                                 \
      CUDAContext* context) {                                   \
    CUBLAS_ENFORCE(cublasSetPointerMode(                        \
        context->cublas_handle(), CUBLAS_POINTER_MODE_HOST));   \
    CUBLAS_ENFORCE(cublasAxpyEx(                                \
        context->cublas_handle(),                               \
        N,                                                      \
        &alpha,                                                 \
        kAlphaType,                                             \
        X,                                                      \
        kDataType,                                              \
        1,                                                      \
        Y,                                                      \
        kDataType,                                              \
        1,                                                      \
        kExecutionType));                                       \
  }                                                             \
  template <>                                                   \
  CAFFE2_CUDA_EXPORT void Axpy<TAlpha, TData, CUDAContext>(     \
      const std::int64_t N,                                     \
      const TAlpha* alpha,                                      \
      const TData* X,                                           \
      TData* Y,                                                 \
      CUDAContext* context) {                                   \
    CUBLAS_ENFORCE(cublasSetPointerMode(                        \
        context->cublas_handle(), CUBLAS_POINTER_MODE_DEVICE)); \
    CUBLAS_ENFORCE(cublasAxpyEx(                                \
        context->cublas_handle(),                               \
        N,                                                      \
        alpha,                                                  \
        kAlphaType,                                             \
        X,                                                      \
        kDataType,                                              \
        1,                                                      \
        Y,                                                      \
        kDataType,                                              \
        1,                                                      \
        kExecutionType));                                       \
  }
DELEGATE_CUDA_AXPY_EX(float, double, CUDA_R_32F, CUDA_R_64F, CUDA_R_64F)
DELEGATE_CUDA_AXPY_EX(float, at::Half, CUDA_R_32F, CUDA_R_16F, CUDA_R_32F)
#undef DELEGATE_CUDA_AXPY_EX

#else // USE_ROCM

#define CAFFE2_SPECIALIZED_CUDA_AXPY(TAlpha, TData)                        \
  template <>                                                              \
  CAFFE2_CUDA_EXPORT void Axpy<TAlpha, TData, CUDAContext>(                \
      const std::int64_t N,                                                \
      const TAlpha alpha,                                                  \
      const TData* X,                                                      \
      TData* Y,                                                            \
      CUDAContext* context) {                                              \
    const std::int64_t M = DivUp<std::int64_t>(N, CAFFE_CUDA_NUM_THREADS); \
    AxpyCUDAKernel<TAlpha, TData>                                          \
        <<<M, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(        \
            N, alpha, X, Y);                                               \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                        \
  }                                                                        \
  template <>                                                              \
  CAFFE2_CUDA_EXPORT void Axpy<TAlpha, TData, CUDAContext>(                \
      const std::int64_t N,                                                \
      const TAlpha* alpha,                                                 \
      const TData* X,                                                      \
      TData* Y,                                                            \
      CUDAContext* context) {                                              \
    const std::int64_t M = DivUp<std::int64_t>(N, CAFFE_CUDA_NUM_THREADS); \
    AxpyCUDAKernel<TAlpha, TData>                                          \
        <<<M, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(        \
            N, alpha, X, Y);                                               \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                        \
  }
CAFFE2_SPECIALIZED_CUDA_AXPY(float, double)
CAFFE2_SPECIALIZED_CUDA_AXPY(float, at::Half)
#undef CAFFE2_SPECIALIZED_CUDA_AXPY

#endif // USE_ROCM

#define CAFFE2_SPECIALIZED_CUDA_AXPBY(TAlpha, TData)                       \
  template <>                                                              \
  CAFFE2_CUDA_EXPORT void Axpby<TAlpha, TData, CUDAContext>(               \
      const std::int64_t N,                                                \
      const TAlpha alpha,                                                  \
      const TData* X,                                                      \
      const TAlpha beta,                                                   \
      TData* Y,                                                            \
      CUDAContext* context) {                                              \
    const std::int64_t M = DivUp<std::int64_t>(N, CAFFE_CUDA_NUM_THREADS); \
    AxpbyCUDAKernel<TAlpha, TData>                                         \
        <<<M, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(        \
            N, alpha, X, beta, Y);                                         \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                        \
  }                                                                        \
  template <>                                                              \
  CAFFE2_CUDA_EXPORT void Axpby<TAlpha, TData, CUDAContext>(               \
      const std::int64_t N,                                                \
      const TAlpha* alpha,                                                 \
      const TData* X,                                                      \
      const TAlpha* beta,                                                  \
      TData* Y,                                                            \
      CUDAContext* context) {                                              \
    const std::int64_t M = DivUp<std::int64_t>(N, CAFFE_CUDA_NUM_THREADS); \
    AxpbyCUDAKernel<TAlpha, TData>                                         \
        <<<M, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(        \
            N, alpha, X, beta, Y);                                         \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                        \
  }
CAFFE2_SPECIALIZED_CUDA_AXPBY(float, float)
CAFFE2_SPECIALIZED_CUDA_AXPBY(float, double)
CAFFE2_SPECIALIZED_CUDA_AXPBY(float, at::Half)
#undef CAFFE2_SPECIALIZED_CUDA_AXPBY

} // namespace math
} // namespace caffe2
