#include "caffe2/utils/math/broadcast.h"

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math/utils.h"

namespace caffe2 {
namespace math {

namespace {

template <typename T>
__global__ void AffineChannelNCHWCUDAKernel(
    const int C,
    const int M,
    const int HxW,
    const T* X,
    const T* scale,
    const T* bias,
    T* Y);

template <>
__global__ void AffineChannelNCHWCUDAKernel<float>(
    const int C,
    const int M,
    const int HxW,
    const float* X,
    const float* scale,
    const float* bias,
    float* Y) {
  const int nc = blockIdx.x / M;
  const int c = nc % C;
  const int w = blockIdx.x % M * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
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
    const int M = DivUp(HxW, CAFFE_CUDA_NUM_THREADS);                        \
    AffineChannelNCHWCUDAKernel<T>                                           \
        <<<N * C * M, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(  \
            C, M, HxW, X, scale, bias, Y);                                   \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                          \
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
    const int M = DivUp(C, CAFFE_CUDA_NUM_THREADS);                          \
    AffineChannelNHWCCUDAKernel<T>                                           \
        <<<dim3(N* HxW, M),                                                  \
           CAFFE_CUDA_NUM_THREADS,                                           \
           0,                                                                \
           context->cuda_stream()>>>(C, X, scale, bias, Y);                  \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                          \
  }
CAFFE2_SPECIALIZED_CUDA_AFFINE_CHANNEL(float)
#undef CAFFE2_SPECIALIZED_CUDA_AFFINE_CHANNEL

} // namespace math
} // namespace caffe2
