#include "caffe2/operators/channel_backprop_stats_op.h"

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math/reduce.cuh"

namespace caffe2 {

namespace {

template <typename T, int kBlockDimX, int kBlockDimY>
__global__ void ChannelStatsBackwardNCHWCUDAKernel(
    const int N,
    const int C,
    const int HxW,
    const T* dY,
    const T* X,
    const T* mean,
    const T* rstd,
    T* dscale,
    T* dbias) {
  __shared__
      typename BlockReduce2D<T, kBlockDimX, kBlockDimY>::TempStorage ds_storage;
  __shared__
      typename BlockReduce2D<T, kBlockDimX, kBlockDimY>::TempStorage db_storage;
  const int c = blockIdx.x;
  T ds_val = 0;
  T db_val = 0;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    for (int j = threadIdx.y; j < HxW; j += blockDim.y) {
      const int index = (i * C + c) * HxW + j;
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
      ds_val += __ldg(dY + index) * __ldg(X + index);
      db_val += __ldg(dY + index);
#else
      ds_val += dY[index] * X[index];
      db_val += dY[index];
#endif
    }
  }
  ds_val = BlockReduce2D<T, kBlockDimX, kBlockDimY>(ds_storage).Sum(ds_val);
  db_val = BlockReduce2D<T, kBlockDimX, kBlockDimY>(db_storage).Sum(db_val);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    dscale[c] = (ds_val - __ldg(mean + c) * db_val) * __ldg(rstd + c);
#else
    dscale[c] = (ds_val - mean[c] * db_val) * rstd[c];
#endif
    dbias[c] = db_val;
  }
}

template <typename T>
__global__ void ChannelStatsBackwardNHWCCUDAKernel(
    const int N,
    const int C,
    const int HxW,
    const T* dY,
    const T* X,
    const T* mean,
    const T* rstd,
    T* dscale,
    T* dbias) {
  __shared__ typename BlockReduce<T>::TempStorage ds_storage;
  __shared__ typename BlockReduce<T>::TempStorage db_storage;
  const int c = blockIdx.x;
  const int inner_size = N * HxW;
  T ds_val = 0;
  T db_val = 0;
  for (int i = threadIdx.x; i < inner_size; i += blockDim.x) {
    const int index = i * C + c;
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    ds_val += __ldg(dY + index) * __ldg(X + index);
    db_val += __ldg(dY + index);
#else
    ds_val += dY[index] * X[index];
    db_val += dY[index];
#endif
  }
  ds_val = BlockReduce<T>(ds_storage).Sum(ds_val);
  db_val = BlockReduce<T>(db_storage).Sum(db_val);
  if (threadIdx.x == 0) {
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    dscale[c] = (ds_val - __ldg(mean + c) * db_val) * __ldg(rstd + c);
#else
    dscale[c] = (ds_val - mean[c] * db_val) * rstd[c];
#endif
    dbias[c] = db_val;
  }
}

} // namespace

template <>
template <>
bool ChannelBackpropStatsOp<CUDAContext>::ChannelStatsBackwardNCHW<float>(
    const int N,
    const int C,
    const int HxW,
    const float* dY,
    const float* X,
    const float* mean,
    const float* rstd,
    float* dscale,
    float* dbias) {
  DISPATCH_REDUCE_KERNEL_BY_2D_BLOCK_WITH_TYPE_1(
      HxW,
      ChannelStatsBackwardNCHWCUDAKernel,
      float,
      C,
      context_.cuda_stream(),
      N,
      C,
      HxW,
      dY,
      X,
      mean,
      rstd,
      dscale,
      dbias);
  return true;
}

template <>
template <>
bool ChannelBackpropStatsOp<CUDAContext>::ChannelStatsBackwardNHWC<float>(
    const int N,
    const int C,
    const int HxW,
    const float* dY,
    const float* X,
    const float* mean,
    const float* rstd,
    float* dscale,
    float* dbias) {
  ChannelStatsBackwardNHWCCUDAKernel<float>
      <<<C, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          N, C, HxW, dY, X, mean, rstd, dscale, dbias);
  return true;
}

REGISTER_CUDA_OPERATOR(
    ChannelBackpropStats,
    ChannelBackpropStatsOp<CUDAContext>);

} // namespace caffe2
