#include "caffe2/utils/math/elementwise.h"

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math_utils.h"

namespace caffe2 {
namespace math {

namespace {

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
  const int block = blockIdx.x % K;
  const int c = nc % C;
  const int w = block * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
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
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
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
    AffineChannelNHWCCUDAKernel<T>                                           \
        <<<N * HxW, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(    \
            C, X, scale, bias, Y);                                           \
  }
CAFFE2_SPECIALIZED_CUDA_AFFINE_CHANNEL(float)
#undef CAFFE2_SPECIALIZED_CUDA_AFFINE_CHANNEL

} // namespace math
} // namespace caffe2
