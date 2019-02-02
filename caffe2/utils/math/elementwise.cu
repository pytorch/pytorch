#include "caffe2/utils/math/elementwise.h"

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math_utils.h"

namespace caffe2 {
namespace math {

namespace {

template <typename T>
__global__ void SinCosCUDAKernel(const int N, const T* X, T* S, T* C) {
  const int i = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (i < N) {
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    c10::cuda::compat::sincos(__ldg(X + i), S + i, C + i);
#else
    c10::cuda::compat::sincos(X[i], S + i, C + i);
#endif
  }
}

template <typename T>
__global__ void AffineChannelNCHWCUDAKernel(
    const int C,
    const int HxW,
    const int K,
    const T* X,
    const T* scale,
    const T* bias,
    T* Y);

template <>
__global__ void AffineChannelNCHWCUDAKernel<float>(
    const int C,
    const int HxW,
    const int K,
    const float* X,
    const float* scale,
    const float* bias,
    float* Y) {
  const int nc = blockIdx.x / K;
  const int c = nc % C;
  const int w = blockIdx.x % K * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (w < HxW) {
    const int index = nc * HxW + w;
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    Y[index] = fmaf(__ldg(X + index), __ldg(scale + c), __ldg(bias + c));
#else
    Y[index] = fmaf(X[index], scale[c], bias[c]);
#endif
  }
}

template <typename T>
__global__ void AffineChannelNHWCCUDAKernel(
    const int C,
    const T* X,
    const T* scale,
    const T* bias,
    T* Y);

template <>
__global__ void AffineChannelNHWCCUDAKernel<float>(
    const int C,
    const float* X,
    const float* scale,
    const float* bias,
    float* Y) {
  const int c = blockIdx.y * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (c < C) {
    const int index = blockIdx.x * C + c;
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    Y[index] = fmaf(__ldg(X + index), __ldg(scale + c), __ldg(bias + c));
#else
    Y[index] = fmaf(X[index], scale[c], bias[c]);
#endif
  }
}

} // namespace

#define DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(T, Func, KernelFunc)     \
  __global__ void Func##CUDAKernel(const int N, const T* X, T* Y) {  \
    const int i = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x; \
    if (i < N) {                                                     \
      Y[i] = KernelFunc(X[i]);                                       \
    }                                                                \
  }                                                                  \
  template <>                                                        \
  CAFFE2_CUDA_EXPORT void Func<T, CUDAContext>(                      \
      const int N, const T* X, T* Y, CUDAContext* context) {         \
    if (N > 0) {                                                     \
      const int K = DivUp(N, CAFFE_CUDA_NUM_THREADS);                \
      Func##CUDAKernel<<<                                            \
          K,                                                         \
          CAFFE_CUDA_NUM_THREADS,                                    \
          0,                                                         \
          context->cuda_stream()>>>(N, X, Y);                        \
    }                                                                \
  }
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Exp, expf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Log, logf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Cos, cosf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Acos, acosf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Sin, sinf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Asin, asinf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Tan, tanf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Atan, atanf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Sinh, sinhf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Cosh, coshf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Tanh, tanhf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Abs, fabsf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Sqr, utils::Square<float>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Sqrt, sqrtf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Rsqrt, rsqrtf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Cbrt, cbrtf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Erf, erff)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(double, Erf, erf)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Cube, utils::Cube<float>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(double, Cube, utils::Cube<double>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(
    std::int32_t,
    Cube,
    utils::Cube<std::int32_t>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(
    std::int64_t,
    Cube,
    utils::Cube<std::int64_t>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(bool, Not, utils::Not)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Neg, utils::Negate<float>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(double, Neg, utils::Negate<double>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(
    std::int32_t,
    Neg,
    utils::Negate<std::int32_t>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(
    std::int64_t,
    Neg,
    utils::Negate<std::int64_t>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Sign, utils::Sign<float>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(double, Sign, utils::Sign<double>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(
    std::int32_t,
    Sign,
    utils::Sign<std::int32_t>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(
    std::int64_t,
    Sign,
    utils::Sign<std::int64_t>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(float, Inv, utils::Inv<float>)
DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION(double, Inv, utils::Inv<double>)
#undef DELEGATE_SIMPLE_CUDA_UNARY_FUNCTION

#define CAFFE2_SPECIALIZED_CUDA_SINCOS(T)                          \
  template <>                                                      \
  CAFFE2_CUDA_EXPORT void SinCos<T, CUDAContext>(                  \
      const int N, const T* X, T* S, T* C, CUDAContext* context) { \
    if (N > 0) {                                                   \
      const int K = DivUp(N, CAFFE_CUDA_NUM_THREADS);              \
      SinCosCUDAKernel<<<                                          \
          K,                                                       \
          CAFFE_CUDA_NUM_THREADS,                                  \
          0,                                                       \
          context->cuda_stream()>>>(N, X, S, C);                   \
    }                                                              \
  }
CAFFE2_SPECIALIZED_CUDA_SINCOS(float)
CAFFE2_SPECIALIZED_CUDA_SINCOS(double)
#undef CAFFE2_SPECIALIZED_CUDA_SINCOS

#define CAFFE2_SPECIALIZED_CUDA_AFFINE_CHANNEL(T)                            \
  template <>                                                                \
  CAFFE2_CUDA_EXPORT void AffineChannel<T, CUDAContext, StorageOrder::NCHW>( \
      const int N,                                                           \
      const int C,                                                           \
      const int HxW,                                                         \
      const T* X,                                                            \
      const T* scale,                                                        \
      const T* bias,                                                         \
      T* Y,                                                                  \
      CUDAContext* context) {                                                \
    const int K = DivUp(HxW, CAFFE_CUDA_NUM_THREADS);                        \
    AffineChannelNCHWCUDAKernel<T>                                           \
        <<<N * C * K, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(  \
            C, HxW, K, X, scale, bias, Y);                                   \
  }                                                                          \
  template <>                                                                \
  CAFFE2_CUDA_EXPORT void AffineChannel<T, CUDAContext, StorageOrder::NHWC>( \
      const int N,                                                           \
      const int C,                                                           \
      const int HxW,                                                         \
      const T* X,                                                            \
      const T* scale,                                                        \
      const T* bias,                                                         \
      T* Y,                                                                  \
      CUDAContext* context) {                                                \
    const int K = DivUp(C, CAFFE_CUDA_NUM_THREADS);                          \
    AffineChannelNHWCCUDAKernel<T>                                           \
        <<<dim3(N* HxW, K),                                                  \
           CAFFE_CUDA_NUM_THREADS,                                           \
           0,                                                                \
           context->cuda_stream()>>>(C, X, scale, bias, Y);                  \
  }
CAFFE2_SPECIALIZED_CUDA_AFFINE_CHANNEL(float)
#undef CAFFE2_SPECIALIZED_CUDA_AFFINE_CHANNEL

} // namespace math
} // namespace caffe2
