#include "caffe2/operators/channel_stats_op.h"

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math/reduce.cuh"

namespace caffe2 {

namespace {

template <typename T, int kBlockDimX, int kBlockDimY>
__global__ void ChannelStatsNCHWCUDAKernel(
    const int N,
    const int C,
    const int HxW,
    const T* X,
    T* sum,
    T* sumsq) {
  __shared__
      typename BlockReduce2D<T, kBlockDimX, kBlockDimY>::TempStorage m_storage;
  __shared__
      typename BlockReduce2D<T, kBlockDimX, kBlockDimY>::TempStorage v_storage;
  const int c = blockIdx.x;
  T m_val = 0;
  T v_val = 0;
  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    for (int hw = threadIdx.y; hw < HxW; hw += blockDim.y) {
      const int index = (n * C + c) * HxW + hw;
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
      m_val += __ldg(X + index);
      v_val += __ldg(X + index) * __ldg(X + index);
#else
      m_val += X[index];
      v_val += X[index] * X[index];
#endif
    }
  }
  m_val = BlockReduce2D<T, kBlockDimX, kBlockDimY>(m_storage).Sum(m_val);
  v_val = BlockReduce2D<T, kBlockDimX, kBlockDimY>(v_storage).Sum(v_val);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    sum[c] = m_val;
    sumsq[c] = v_val;
  }
}

template <typename T>
__global__ void ChannelStatsNHWCCUDAKernel(
    const int N,
    const int C,
    const int HxW,
    const T* X,
    T* sum,
    T* sumsq) {
  __shared__ typename BlockReduce<T>::TempStorage m_storage;
  __shared__ typename BlockReduce<T>::TempStorage v_storage;
  const int inner_size = N * HxW;
  const int c = blockIdx.x;
  T m_val = 0;
  T v_val = 0;
  for (int i = threadIdx.x; i < inner_size; i += blockDim.x) {
    const int index = i * C + c;
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    m_val += __ldg(X + index);
    v_val += __ldg(X + index) * __ldg(X + index);
#else
    m_val += X[index];
    v_val += X[index] * X[index];
#endif
  }
  m_val = BlockReduce<T>(m_storage).Sum(m_val);
  v_val = BlockReduce<T>(v_storage).Sum(v_val);
  if (threadIdx.x == 0) {
    sum[c] = m_val;
    sumsq[c] = v_val;
  }
}

} // namespace

template <>
template <>
bool ChannelStatsOp<CUDAContext>::ComputeChannelStatsNCHW<float>(
    const int N,
    const int C,
    const int HxW,
    const float* X,
    float* sum,
    float* sumsq) {
  DISPATCH_REDUCE_KERNEL_BY_2D_BLOCK_WITH_TYPE_1(
      HxW,
      ChannelStatsNCHWCUDAKernel,
      float,
      C,
      context_.cuda_stream(),
      N,
      C,
      HxW,
      X,
      sum,
      sumsq);
  return true;
}

template <>
template <>
bool ChannelStatsOp<CUDAContext>::ComputeChannelStatsNHWC<float>(
    const int N,
    const int C,
    const int HxW,
    const float* X,
    float* sum,
    float* sumsq) {
  ChannelStatsNHWCCUDAKernel<float>
      <<<C, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          N, C, HxW, X, sum, sumsq);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

REGISTER_CUDA_OPERATOR(ChannelStats, ChannelStatsOp<CUDAContext>);

} // namespace caffe2
