#include "caffe2/operators/batch_moments_op.h"

#include <cub/block/block_reduce.cuh>

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {

template <typename T>
using BlockReduce = cub::BlockReduce<T, CAFFE_CUDA_NUM_THREADS>;

template <typename T, StorageOrder kOrder>
__global__ void BatchMomentsCUDAKernel(
    const int N,
    const int C,
    const int HxW,
    const T* X,
    T* mu,
    T* var) {
  const int outer_size = C;
  const int inner_size = N * HxW;
  __shared__ typename BlockReduce<T>::TempStorage m_storage;
  __shared__ typename BlockReduce<T>::TempStorage v_storage;
  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    T m_sum = 0;
    T v_sum = 0;
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = kOrder == StorageOrder::NCHW
          ? (j / HxW * C + i) * HxW + j % HxW
          : j * outer_size + i;
#if __CUDA_ARCH__ >= 350
      m_sum += __ldg(X + index);
      v_sum += __ldg(X + index) * __ldg(X + index);
#else
      m_sum += X[index];
      v_sum += X[index] * X[index];
#endif
    }
    m_sum = BlockReduce<T>(m_storage).Reduce(m_sum, cub::Sum());
    v_sum = BlockReduce<T>(v_storage).Reduce(v_sum, cub::Sum());
    if (threadIdx.x == 0) {
      mu[i] = m_sum / static_cast<T>(N * HxW);
      var[i] = v_sum / static_cast<T>(N * HxW);
    }
    __syncthreads();
  }
}

template <typename T, StorageOrder kOrder>
__global__ void BatchMomentsGradientCUDAKernel(
    const int N,
    const int C,
    const int HxW,
    const T* dmu,
    const T* dvar,
    const T* X,
    T* dX) {
  const int size = N * C * HxW;
  const T scale = T(1) / static_cast<T>(N * HxW);
  CUDA_1D_KERNEL_LOOP(i, size) {
    const int i_mu = kOrder == StorageOrder::NCHW ? i / (HxW) % C : i % C;
#if __CUDA_ARCH__ >= 350
    dX[i] =
        (__ldg(dmu + i_mu) + __ldg(dvar + i_mu) * T(2) * __ldg(X + i)) * scale;
#else
    dX[i] = (dmu[i_mu] + dvar[i_mu] * T(2) * X[i]) * scale;
#endif
  }
}

} // namespace

template <>
bool BatchMomentsOp<float, CUDAContext>::ComputeBatchMomentsNCHW(
    const int N,
    const int C,
    const int HxW,
    const float* X,
    float* mu,
    float* var) {
  const int outer_size = N * HxW;
  BatchMomentsCUDAKernel<float, StorageOrder::NCHW>
      <<<std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(N, C, HxW, X, mu, var);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

template <>
bool BatchMomentsOp<float, CUDAContext>::ComputeBatchMomentsNHWC(
    const int N,
    const int C,
    const int HxW,
    const float* X,
    float* mu,
    float* var) {
  const int outer_size = N * HxW;
  BatchMomentsCUDAKernel<float, StorageOrder::NHWC>
      <<<std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(N, C, HxW, X, mu, var);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

template <>
bool BatchMomentsGradientOp<float, CUDAContext>::
    ComputeBatchMomentsGradientNCHW(
        const int N,
        const int C,
        const int HxW,
        const float* dmu,
        const float* dvar,
        const float* X,
        float* dX) {
  const int size = N * C * HxW;
  BatchMomentsGradientCUDAKernel<float, StorageOrder::NCHW>
      <<<CAFFE_GET_BLOCKS(size),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(N, C, HxW, dmu, dvar, X, dX);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

template <>
bool BatchMomentsGradientOp<float, CUDAContext>::
    ComputeBatchMomentsGradientNHWC(
        const int N,
        const int C,
        const int HxW,
        const float* dmu,
        const float* dvar,
        const float* X,
        float* dX) {
  const int size = N * C * HxW;
  BatchMomentsGradientCUDAKernel<float, StorageOrder::NHWC>
      <<<CAFFE_GET_BLOCKS(size),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(N, C, HxW, dmu, dvar, X, dX);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

REGISTER_CUDA_OPERATOR(BatchMoments, BatchMomentsOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    BatchMomentsGradient,
    BatchMomentsGradientOp<float, CUDAContext>);

} // namespace caffe2
