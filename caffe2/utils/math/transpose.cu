#include "caffe2/utils/math/transpose.h"

#include <algorithm>
#include <functional>
#include <numeric>

#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math/utils.h"

namespace caffe2 {
namespace math {

namespace {

constexpr int kTileDim = 32;
constexpr int kBlockRows = 8;

// Splits the original matrix into submatrices with size 32 * 32.
// Each block transposes one submatrix by loading it into shared memory.
// Reference https://devblogs.nvidia.com/efficient-matrix-transpose-cuda-cc/
template <typename TIndex, typename TData>
__global__ void BatchTranspose2DCUDAKernel(
    const TIndex H,
    const TIndex W,
    const TIndex dh,
    const TIndex dw,
    const TData* X,
    TData* Y) {
  __shared__ TData tile[kTileDim][kTileDim + 1];
  const TIndex n = blockIdx.x / (dh * dw);
  const TIndex k = blockIdx.x % (dh * dw);
  const TIndex r = k / dw;
  const TIndex c = k % dw;
  const TIndex offset = n * H * W;
  int x = c * kTileDim + threadIdx.x;
  int y = r * kTileDim + threadIdx.y;
  if (x < W) {
    for (int i = 0; threadIdx.y + i < kTileDim && y + i < H; i += kBlockRows) {
#if __CUDA_ARCH__ >= 350 || defined(USE_ROCM)
      tile[threadIdx.y + i][threadIdx.x] = __ldg(X + offset + (y + i) * W + x);
#else
      tile[threadIdx.y + i][threadIdx.x] = X[offset + (y + i) * W + x];
#endif
    }
  }
  __syncthreads();
  x = r * kTileDim + threadIdx.x;
  y = c * kTileDim + threadIdx.y;
  if (x < H) {
    for (int i = 0; threadIdx.y + i < kTileDim && y + i < W; i += kBlockRows) {
      Y[offset + (y + i) * H + x] = tile[threadIdx.x][threadIdx.y + i];
    }
  }
}

template <typename TIndex, typename TData>
void BatchTranspose2DCUDAImpl(
    const TIndex N,
    const TIndex H,
    const TIndex W,
    const TData* X,
    TData* Y,
    CUDAContext* context) {
  const TIndex dh = DivUp<TIndex>(H, kTileDim);
  const TIndex dw = DivUp<TIndex>(W, kTileDim);
  BatchTranspose2DCUDAKernel<TIndex, TData>
      <<<N * dh * dw, dim3(kTileDim, kBlockRows), 0, context->cuda_stream()>>>(
          H, W, dh, dw, X, Y);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#define DELEGATE_TRANSPOSE_2D_CUDA_IMPL(TIndex, TData, CuBLASFunc) \
  template <>                                                      \
  void BatchTranspose2DCUDAImpl<TIndex, TData>(                    \
      const TIndex N,                                              \
      const TIndex H,                                              \
      const TIndex W,                                              \
      const TData* X,                                              \
      TData* Y,                                                    \
      CUDAContext* context) {                                      \
    if (N == 1) {                                                  \
      const TData kAlpha = TData(1);                               \
      const TData kBeta = TData(0);                                \
      CUBLAS_ENFORCE(cublasSetPointerMode(                         \
          context->cublas_handle(), CUBLAS_POINTER_MODE_HOST));    \
      CUBLAS_ENFORCE(CuBLASFunc(                                   \
          context->cublas_handle(),                                \
          CUBLAS_OP_T,                                             \
          CUBLAS_OP_N,                                             \
          H,                                                       \
          W,                                                       \
          &kAlpha,                                                 \
          X,                                                       \
          W,                                                       \
          &kBeta,                                                  \
          Y,                                                       \
          H,                                                       \
          Y,                                                       \
          H));                                                     \
    } else {                                                       \
      const TIndex dh = DivUp<TIndex>(H, kTileDim);                \
      const TIndex dw = DivUp<TIndex>(W, kTileDim);                \
      BatchTranspose2DCUDAKernel<TIndex, TData>                    \
          <<<N * dh * dw,                                          \
             dim3(kTileDim, kBlockRows),                           \
             0,                                                    \
             context->cuda_stream()>>>(H, W, dh, dw, X, Y);        \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                              \
    }                                                              \
  }
DELEGATE_TRANSPOSE_2D_CUDA_IMPL(std::int32_t, float, cublasSgeam)
DELEGATE_TRANSPOSE_2D_CUDA_IMPL(std::int64_t, float, cublasSgeam)
DELEGATE_TRANSPOSE_2D_CUDA_IMPL(std::int32_t, double, cublasDgeam)
DELEGATE_TRANSPOSE_2D_CUDA_IMPL(std::int64_t, double, cublasDgeam)
#undef DELEGATE_TRANSPOSE_2D_CUDA_IMPL

template <typename TIndex, typename TData, int D>
__global__ void TransposeCUDAKernel(
    const TIndex size,
    const SimpleArray<TIndex, D> X_strides,
    const SimpleArray<TIndex, D> Y_dims,
    const TData* X,
    TData* Y) {
  const int Y_index = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (Y_index < size) {
    TIndex X_index = 0;
    TIndex v = Y_index;
#pragma unroll
    for (int i = D - 1; i >= 0; --i) {
      X_index += v % Y_dims.data[i] * X_strides.data[i];
      v /= Y_dims.data[i];
    }
#if __CUDA_ARCH__ >= 350 || defined(USE_ROCM)
    Y[Y_index] = __ldg(X + X_index);
#else
    Y[Y_index] = X[X_index];
#endif
  }
}

template <typename TIndex, typename TData, int D>
void TransposeCUDAImpl(
    const TIndex* dims,
    const int* axes,
    const TData* X,
    TData* Y,
    CUDAContext* context) {
  SimpleArray<TIndex, D> X_strides;
  SimpleArray<TIndex, D> Y_dims;
  utils::ComputeTransposedStrides<TIndex>(D, dims, axes, X_strides.data);
  TIndex size = 1;
  for (int i = 0; i < D; ++i) {
    Y_dims.data[i] = dims[axes[i]];
    size *= dims[i];
  }
  const TIndex M = DivUp<TIndex>(size, CAFFE_CUDA_NUM_THREADS);
  TransposeCUDAKernel<TIndex, TData, D>
      <<<M, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
          size, X_strides, Y_dims, X, Y);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

#define CAFFE2_SPECIALIZED_CUDA_TRANSPOSE(TIndex, TData)                    \
  template <>                                                               \
  CAFFE2_CUDA_EXPORT void Transpose<TIndex, TData, CUDAContext>(            \
      const int ndim,                                                       \
      const TIndex* dims,                                                   \
      const int* axes,                                                      \
      const TData* X,                                                       \
      TData* Y,                                                             \
      CUDAContext* context) {                                               \
    const TIndex size = std::accumulate(                                    \
        dims, dims + ndim, TIndex(1), std::multiplies<TIndex>());           \
    if (size == 0) {                                                        \
      return;                                                               \
    }                                                                       \
    if (utils::IsIdentityPermutation(ndim, axes)) {                         \
      context->template CopySameDevice<TData>(size, X, Y);                  \
      return;                                                               \
    }                                                                       \
    if (utils::IsBatchTranspose2D(ndim, axes)) {                            \
      const int H = dims[ndim - 2];                                         \
      const int W = dims[ndim - 1];                                         \
      const int N = size / (H * W);                                         \
      BatchTranspose2DCUDAImpl<TIndex, TData>(N, H, W, X, Y, context);      \
      return;                                                               \
    }                                                                       \
    DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_2(                                 \
        ndim, TransposeCUDAImpl, TIndex, TData, dims, axes, X, Y, context); \
  }
CAFFE2_SPECIALIZED_CUDA_TRANSPOSE(std::int32_t, float)
CAFFE2_SPECIALIZED_CUDA_TRANSPOSE(std::int64_t, float)
CAFFE2_SPECIALIZED_CUDA_TRANSPOSE(std::int32_t, double)
CAFFE2_SPECIALIZED_CUDA_TRANSPOSE(std::int64_t, double)
CAFFE2_SPECIALIZED_CUDA_TRANSPOSE(std::int32_t, std::int32_t)
CAFFE2_SPECIALIZED_CUDA_TRANSPOSE(std::int64_t, std::int32_t)
CAFFE2_SPECIALIZED_CUDA_TRANSPOSE(std::int32_t, std::int64_t)
CAFFE2_SPECIALIZED_CUDA_TRANSPOSE(std::int64_t, std::int64_t)
#undef CAFFE2_SPECIALIZED_CUDA_TRANSPOSE

#define CAFFE2_SPECIALIZED_CUDA_NCHW2NHWC(T)                    \
  template <>                                                   \
  CAFFE2_CUDA_EXPORT void NCHW2NHWC<T, CUDAContext>(            \
      const int N,                                              \
      const int C,                                              \
      const int HxW,                                            \
      const T* X,                                               \
      T* Y,                                                     \
      CUDAContext* context) {                                   \
    BatchTranspose2DCUDAImpl<int, T>(N, C, HxW, X, Y, context); \
  }
CAFFE2_SPECIALIZED_CUDA_NCHW2NHWC(float)
#undef CAFFE2_SPECIALIZED_CUDA_NCHW2NHWC

#define CAFFE2_SPECIALIZED_CUDA_NHWC2NCHW(T)                    \
  template <>                                                   \
  CAFFE2_CUDA_EXPORT void NHWC2NCHW<T, CUDAContext>(            \
      const int N,                                              \
      const int C,                                              \
      const int HxW,                                            \
      const T* X,                                               \
      T* Y,                                                     \
      CUDAContext* context) {                                   \
    BatchTranspose2DCUDAImpl<int, T>(N, HxW, C, X, Y, context); \
  }
CAFFE2_SPECIALIZED_CUDA_NHWC2NCHW(float)
#undef CAFFE2_SPECIALIZED_CUDA_NHWC2NCHW

} // namespace math
} // namespace caffe2
