#include "caffe2/operators/relu_op.h"

#include <algorithm>
#include <functional>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

#ifdef __HIPCC__
using half2 = __half2;
#endif // __HIPCC__

template <typename T>
__global__ void ReluCUDAKernel(const int N, const T* X, T* Y);

#define DELEGATE_RELU_CUDA_KERNEL(T, MaxFunc)                        \
  template <>                                                        \
  __global__ void ReluCUDAKernel<T>(const int N, const T* X, T* Y) { \
    const int i = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x; \
    if (i < N) {                                                     \
      Y[i] = MaxFunc(X[i], T(0));                                    \
    }                                                                \
  }
DELEGATE_RELU_CUDA_KERNEL(float, fmaxf)
#undef DELEGATE_RELU_CUDA_KERNEL

template <>
__global__ void ReluCUDAKernel<half>(const int N, const half* X, half* Y) {
  const int i = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (i < N) {
    const half kZero = __float2half(0.0f);
#if __CUDA_ARCH__ >= 530 || TORCH_HIP_VERSION >= 300
    Y[i] = __hgt(__ldg(X + i), kZero) ? __ldg(X + i) : kZero;
#else
    Y[i] = (__half2float(X[i]) > 0) ? X[i] : kZero;
#endif
  }
}

template <>
__global__ void ReluCUDAKernel<half2>(const int N, const half2* X, half2* Y) {
  const int i = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (i < N) {
    const half2 kZero = __float2half2_rn(0.0f);
#if __CUDA_ARCH__ >= 530 || TORCH_HIP_VERSION >= 300
    Y[i] = __hmul2(__hgt2(__ldg(X + i), kZero), __ldg(X + i));
#else
    const float2 xx = __half22float2(X[i]);
    // There are explicit cast to float here, because it may otherwise cause ambiguity on ROCm and can be triggered
    // sometimes:
    //
    //   error: conditional expression is ambiguous; 'const hip_impl::Scalar_accessor<float, Native_vec_, 0>' can be
    //   converted to 'float' and vice versa
    Y[i] = __floats2half2_rn(xx.x > 0.0f ? static_cast<float>(xx.x) : 0.0f,
                             xx.y > 0.0f ? static_cast<float>(xx.y) : 0.0f);
#endif
  }
}

template <typename T>
__global__ void
ReluGradientCUDAKernel(const int N, const T* dY, const T* Y, T* dX) {
  const int i = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (i < N) {
#if __CUDA_ARCH__ >= 350 || TORCH_HIP_VERSION >= 300
    dX[i] = __ldg(Y + i) > T(0) ? __ldg(dY + i) : T(0);
#else
    dX[i] = Y[i] > T(0) ? dY[i] : T(0);
#endif
  }
}

template <>
__global__ void ReluGradientCUDAKernel<half>(
    const int N,
    const half* dY,
    const half* Y,
    half* dX) {
  const int i = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (i < N) {
    const half kZero = __float2half(0.0f);
#if __CUDA_ARCH__ >= 530 || TORCH_HIP_VERSION >= 300
    dX[i] = __hgt(__ldg(Y + i), kZero) ? __ldg(dY + i) : kZero;
#else
    dX[i] = (__half2float(Y[i]) > 0) ? dY[i] : kZero;
#endif
  }
}

template <>
__global__ void ReluGradientCUDAKernel<half2>(
    const int N,
    const half2* dY,
    const half2* Y,
    half2* dX) {
  const int i = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (i < N) {
    const half2 kZero = __float2half2_rn(0.0f);
#if __CUDA_ARCH__ >= 530 || TORCH_HIP_VERSION >= 300
    dX[i] = __hmul2(__hgt2(__ldg(Y + i), kZero), __ldg(dY + i));
#else
    const float2 dy = __half22float2(dY[i]);
    const float2 yy = __half22float2(Y[i]);
    // There are explicit cast to float here, because it may otherwise cause ambiguity on ROCm and can be triggered
    // sometimes:
    //
    //   error: conditional expression is ambiguous; 'const hip_impl::Scalar_accessor<float, Native_vec_, 1>' can be
    //   converted to 'float' and vice versa

     dX[i] = __floats2half2_rn(yy.x > 0.0f ? static_cast<float>(dy.x) : 0.0f,
                               yy.y > 0.0f ? static_cast<float>(dy.y) : 0.0f);
#endif
  }
}

} // namespace

template <>
template <typename T>
bool ReluFunctor<CUDAContext>::
operator()(const int N, const T* X, T* Y, CUDAContext* context) const {
  if (N > 0) {
    const int M = math::DivUp(N, CAFFE_CUDA_NUM_THREADS);
    ReluCUDAKernel<T>
        <<<M, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(N, X, Y);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return true;
}

template <>
template <>
bool ReluFunctor<CUDAContext>::operator()<at::Half>(
    const int N,
    const at::Half* X,
    at::Half* Y,
    CUDAContext* context) const {
  if (N == 0) {
    return true;
  }
  if (N % 2 == 0) {
    const int M = math::DivUp(N / 2, CAFFE_CUDA_NUM_THREADS);
    ReluCUDAKernel<half2>
        <<<M, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
            N / 2,
            reinterpret_cast<const half2*>(X),
            reinterpret_cast<half2*>(Y));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    const int M = math::DivUp(N, CAFFE_CUDA_NUM_THREADS);
    ReluCUDAKernel<half>
        <<<M, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
            N, reinterpret_cast<const half*>(X), reinterpret_cast<half*>(Y));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return true;
}

template <>
template <typename T>
bool ReluGradientFunctor<CUDAContext>::Forward(
    const std::vector<int>& Y_dims,
    const std::vector<int>& /* dY_dims */,
    const T* Y,
    const T* dY,
    T* dX,
    CUDAContext* context) const {
  const int N = std::accumulate(
      Y_dims.cbegin(), Y_dims.cend(), 1, std::multiplies<int>());
  if (N > 0) {
    const int M = math::DivUp(N, CAFFE_CUDA_NUM_THREADS);
    ReluGradientCUDAKernel<T>
        <<<M, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
            N, dY, Y, dX);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return true;
}

template <>
template <>
bool ReluGradientFunctor<CUDAContext>::Forward<at::Half>(
    const std::vector<int>& Y_dims,
    const std::vector<int>& /* dY_dims */,
    const at::Half* Y,
    const at::Half* dY,
    at::Half* dX,
    CUDAContext* context) const {
  const int N = std::accumulate(
      Y_dims.cbegin(), Y_dims.cend(), 1, std::multiplies<int>());
  if (N == 0) {
    return true;
  }
  if (N % 2 == 0) {
    const int M = math::DivUp(N / 2, CAFFE_CUDA_NUM_THREADS);
    ReluGradientCUDAKernel<half2>
        <<<M, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
            N / 2,
            reinterpret_cast<const half2*>(dY),
            reinterpret_cast<const half2*>(Y),
            reinterpret_cast<half2*>(dX));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    const int M = math::DivUp(N, CAFFE_CUDA_NUM_THREADS);
    ReluGradientCUDAKernel<half>
        <<<M, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
            N,
            reinterpret_cast<const half*>(dY),
            reinterpret_cast<const half*>(Y),
            reinterpret_cast<half*>(dX));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return true;
}

REGISTER_CUDA_OPERATOR(
    Relu,
    UnaryElementwiseOp<
        TensorTypes<float, at::Half>,
        CUDAContext,
        ReluFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    ReluGradient,
    BinaryElementwiseOp<
        TensorTypes<float, at::Half>,
        CUDAContext,
        ReluGradientFunctor<CUDAContext>>);

} // namespace caffe2
