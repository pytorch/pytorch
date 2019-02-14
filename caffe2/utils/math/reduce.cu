#include "caffe2/utils/math.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

#include <cub/block/block_reduce.cuh>
#include <cub/cub.cuh>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/fixed_divisor.h"
#include "caffe2/utils/math/reduce.cuh"
#include "caffe2/utils/math/utils.h"

namespace caffe2 {
namespace math {

namespace {

template <typename T>
__global__ void
RowwiseMomentsCUDAKernel(const int cols, const T* X, T* mean, T* var) {
  __shared__ typename BlockReduce<T>::TempStorage m_storage;
  __shared__ typename BlockReduce<T>::TempStorage v_storage;
  const T scale = T(1) / static_cast<T>(cols);
  const int r = blockIdx.x;
  T m_val = 0;
  T v_val = 0;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const int X_index = r * cols + c;
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    m_val += __ldg(X + X_index);
    v_val += __ldg(X + X_index) * __ldg(X + X_index);
#else
    m_val += X[X_index];
    v_val += X[X_index] * X[X_index];
#endif
  }
  m_val = BlockReduce<T>(m_storage).Sum(m_val);
  v_val = BlockReduce<T>(v_storage).Sum(v_val);
  if (threadIdx.x == 0) {
    const T mu = m_val * scale;
    mean[r] = mu;
    var[r] = v_val * scale - mu * mu;
  }
}

template <typename T>
__global__ void ColwiseMomentsCUDAKernel(
    const int rows,
    const int cols,
    const T* X,
    T* mean,
    T* var) {
  __shared__ typename BlockReduce<T>::TempStorage m_storage;
  __shared__ typename BlockReduce<T>::TempStorage v_storage;
  const T scale = T(1) / static_cast<T>(rows);
  const int c = blockIdx.x;
  T m_val = 0;
  T v_val = 0;
  for (int r = threadIdx.x; r < rows; r += blockDim.x) {
    const int X_index = r * cols + c;
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    m_val += __ldg(X + X_index);
    v_val += __ldg(X + X_index) * __ldg(X + X_index);
#else
    m_val += X[X_index];
    v_val += X[X_index] * X[X_index];
#endif
  }
  m_val = BlockReduce<T>(m_storage).Sum(m_val);
  v_val = BlockReduce<T>(v_storage).Sum(v_val);
  if (threadIdx.x == 0) {
    const T mu = m_val * scale;
    mean[c] = mu;
    var[c] = v_val * scale - mu * mu;
  }
}

template <typename T, int kBlockDimX, int kBlockDimY>
__global__ void BothEndsMomentsCUDAKernel(
    const int M,
    const int N,
    const int K,
    const T* X,
    T* mean,
    T* var) {
  __shared__
      typename BlockReduce2D<T, kBlockDimX, kBlockDimY>::TempStorage m_storage;
  __shared__
      typename BlockReduce2D<T, kBlockDimX, kBlockDimY>::TempStorage v_storage;
  const T scale = T(1) / static_cast<T>(M * K);
  const int n = blockIdx.x;
  T m_val = 0;
  T v_val = 0;
  for (int m = threadIdx.x; m < M; m += blockDim.x) {
    for (int k = threadIdx.y; k < K; k += blockDim.y) {
      const int X_index = (m * N + n) * K + k;
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
      m_val += __ldg(X + X_index);
      v_val += __ldg(X + X_index) * __ldg(X + X_index);
#else
      m_val += X[X_index];
      v_val += X[X_index] * X[X_index];
#endif
    }
  }
  m_val = BlockReduce2D<T, kBlockDimX, kBlockDimY>(m_storage).Sum(m_val);
  v_val = BlockReduce2D<T, kBlockDimX, kBlockDimY>(v_storage).Sum(v_val);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    const T mu = m_val * scale;
    mean[n] = mu;
    var[n] = v_val * scale - mu * mu;
  }
}

template <typename T, int D>
__global__ void MomentsCUDAKernel(
    const int inner_size,
    const SimpleArray<int, D> X_strides,
    const SimpleArray<FixedDivisor<int>, D> Y_dims,
    const T* X,
    T* mean,
    T* var) {
  __shared__ typename BlockReduce<T>::TempStorage m_storage;
  __shared__ typename BlockReduce<T>::TempStorage v_storage;
  const T scale = T(1) / static_cast<T>(inner_size);
  const int x = blockIdx.x;
  T m_val = 0;
  T v_val = 0;
  for (int y = threadIdx.x; y < inner_size; y += blockDim.x) {
    int X_index = 0;
    int Y_index = x * inner_size + y;
#pragma unroll
    for (int d = D - 1; d >= 0; --d) {
      int r;
      Y_dims.data[d].DivMod(Y_index, &Y_index, &r);
      X_index += r * X_strides.data[d];
    }
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    m_val += __ldg(X + X_index);
    v_val += __ldg(X + X_index) * __ldg(X + X_index);
#else
    m_val += X[X_index];
    v_val += X[X_index] * X[X_index];
#endif
  }
  m_val = BlockReduce<T>(m_storage).Sum(m_val);
  v_val = BlockReduce<T>(v_storage).Sum(v_val);
  if (threadIdx.x == 0) {
    const T mu = m_val * scale;
    mean[x] = mu;
    var[x] = v_val * scale - mu * mu;
  }
}

template <typename T, int D>
CAFFE2_CUDA_EXPORT void MomentsCUDAImpl(
    const int outer_size,
    const int inner_size,
    const int* dims,
    const int* axes,
    const T* X,
    T* mean,
    T* var,
    CUDAContext* context) {
  SimpleArray<int, D> X_strides;
  SimpleArray<FixedDivisor<int>, D> Y_dims;
  utils::ComputeTransposedStrides(D, dims, axes, X_strides.data);
  for (int i = 0; i < D; ++i) {
    Y_dims.data[i] = FixedDivisor<int>(dims[axes[i]]);
  }
  MomentsCUDAKernel<T, D>
      <<<outer_size, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
          inner_size, X_strides, Y_dims, X, mean, var);
}

template <typename T>
CAFFE2_CUDA_EXPORT void MomentsCUDA(
    const int ndim,
    const int* X_dims,
    const int* Y_dims,
    const T* X,
    T* mean,
    T* var,
    CUDAContext* context) {
  CAFFE_ENFORCE(utils::CheckReduceDims(ndim, X_dims, Y_dims));
  const int X_size =
      std::accumulate(X_dims, X_dims + ndim, 1, std::multiplies<int>());
  const int Y_size =
      std::accumulate(Y_dims, Y_dims + ndim, 1, std::multiplies<int>());
  if (X_size == 0) {
    Set<T, CUDAContext>(Y_size, T(0), mean, context);
    Set<T, CUDAContext>(Y_size, T(0), var, context);
    return;
  }
  if (std::equal(X_dims, X_dims + ndim, Y_dims)) {
    cudaMemcpyAsync(
        mean,
        X,
        sizeof(T) * X_size,
        cudaMemcpyDeviceToDevice,
        context->cuda_stream());
    Set<T, CUDAContext>(Y_size, T(0), var, context);
    return;
  }
  int rows;
  int cols;
  if (utils::IsRowwiseReduce(ndim, X_dims, Y_dims, &rows, &cols)) {
    RowwiseMomentsCUDAKernel<T>
        <<<rows, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
            cols, X, mean, var);
    return;
  }
  if (utils::IsColwiseReduce(ndim, X_dims, Y_dims, &rows, &cols)) {
    ColwiseMomentsCUDAKernel<T>
        <<<cols, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
            rows, cols, X, mean, var);
    return;
  }
  int M;
  int N;
  int K;
  if (utils::IsBothEndsReduce(ndim, X_dims, Y_dims, &M, &N, &K)) {
    DISPATCH_REDUCE_KERNEL_BY_2D_BLOCK(
        K,
        BothEndsMomentsCUDAKernel,
        T,
        N,
        context->cuda_stream(),
        M,
        N,
        K,
        X,
        mean,
        var);
    return;
  }
  std::vector<int> axes(ndim);
  utils::ComputeTransposeAxesForReduceOp(ndim, Y_dims, axes.data());
  const int outer_size = Y_size;
  const int inner_size = X_size / Y_size;
  DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_1(
      ndim,
      MomentsCUDAImpl,
      T,
      outer_size,
      inner_size,
      X_dims,
      axes.data(),
      X,
      mean,
      var,
      context);
}

} // namespace

#define CAFFE2_SPECIALIZED_CUDA_MOMENTS(T)                       \
  template <>                                                    \
  CAFFE2_CUDA_EXPORT void Moments<T, CUDAContext>(               \
      const int ndim,                                            \
      const int* X_dims,                                         \
      const int* Y_dims,                                         \
      const T* X,                                                \
      T* mean,                                                   \
      T* var,                                                    \
      CUDAContext* context) {                                    \
    MomentsCUDA<T>(ndim, X_dims, Y_dims, X, mean, var, context); \
  }
CAFFE2_SPECIALIZED_CUDA_MOMENTS(float)
#undef CAFFE2_SPECIALIZED_CUDA_MOMENTS

} // namespace math
} // namespace caffe2
