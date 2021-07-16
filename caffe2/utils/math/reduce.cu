#include "caffe2/utils/math/reduce.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

#include <cub/block/block_reduce.cuh>
#include <cub/cub.cuh>

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math/elementwise.h"
#include "caffe2/utils/math/reduce.cuh"
#include "caffe2/utils/math/utils.h"

namespace caffe2 {
namespace math {

namespace {

template <typename T, class Reducer>
__global__ void RowwiseReduceCUDAKernel(
    const int cols,
    const Reducer reducer,
    const T init,
    const T alpha,
    const T* X,
    T* Y) {
  __shared__ typename BlockReduce<T>::TempStorage temp_storage;
  const int r = blockIdx.x;
  T val = init;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    val = reducer(val, __ldg(X + r * cols + c));
#else
    val = reducer(val, X[r * cols + c]);
#endif
  }
  val = BlockReduce<T>(temp_storage).Reduce(val, reducer);
  if (threadIdx.x == 0) {
    Y[r] = val * alpha;
  }
}

template <typename T, class Reducer>
__global__ void ColwiseReduceCUDAKernel(
    const int rows,
    const int cols,
    const Reducer reducer,
    const T init,
    const T alpha,
    const T* X,
    T* Y) {
  __shared__ typename BlockReduce<T>::TempStorage temp_storage;
  const int c = blockIdx.x;
  T val = init;
  for (int r = threadIdx.x; r < rows; r += blockDim.x) {
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    val = reducer(val, __ldg(X + r * cols + c));
#else
    val = reducer(val, X[r * cols + c]);
#endif
  }
  val = BlockReduce<T>(temp_storage).Reduce(val, reducer);
  if (threadIdx.x == 0) {
    Y[c] = val * alpha;
  }
}

template <typename T, class Reducer, int kBlockDimX, int kBlockDimY>
__global__ void BothEndsReduceCUDAKernel(
    const int M,
    const int N,
    const int K,
    const Reducer reducer,
    const T init,
    const T alpha,
    const T* X,
    T* Y) {
  __shared__ typename BlockReduce2D<T, kBlockDimX, kBlockDimY>::TempStorage
      temp_storage;
  const int n = blockIdx.x;
  T val = init;
  for (int m = threadIdx.x; m < M; m += blockDim.x) {
    for (int k = threadIdx.y; k < K; k += blockDim.y) {
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
      val = reducer(val, __ldg(X + (m * N + n) * K + k));
#else
      val = reducer(val, X[(m * N + n) * K + k]);
#endif
    }
  }
  val = BlockReduce2D<T, kBlockDimX, kBlockDimY>(temp_storage)
            .Reduce(val, reducer);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    Y[n] = val * alpha;
  }
}

template <typename T, class Reducer, int D>
__global__ void ReduceTensorCUDAKernel(
    const int inner_size,
    const SimpleArray<int, D> X_strides,
    const SimpleArray<int, D> Y_dims,
    const Reducer reducer,
    const T init,
    const T alpha,
    const T* X,
    T* Y) {
  __shared__ typename BlockReduce<T>::TempStorage temp_storage;
  const int x = blockIdx.x;
  T val = init;
  for (int y = threadIdx.x; y < inner_size; y += blockDim.x) {
    int X_index = 0;
    int Y_index = x * inner_size + y;
#pragma unroll
    for (int d = D - 1; d >= 0; --d) {
      X_index += Y_index % Y_dims.data[d] * X_strides.data[d];
      Y_index /= Y_dims.data[d];
    }
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    val = reducer(val, __ldg(X + X_index));
#else
    val = reducer(val, X[X_index]);
#endif
  }
  val = BlockReduce<T>(temp_storage).Reduce(val, reducer);
  if (threadIdx.x == 0) {
    Y[x] = val * alpha;
  }
}

template <typename T, class Reducer, int D>
void ReduceTensorCUDAImpl(
    const int outer_size,
    const int inner_size,
    const int* dims,
    const int* axes,
    const Reducer& reducer,
    const T init,
    const T alpha,
    const T* X,
    T* Y,
    CUDAContext* context) {
  SimpleArray<int, D> X_strides;
  SimpleArray<int, D> Y_dims;
  utils::ComputeTransposedStrides(D, dims, axes, X_strides.data);
  for (int i = 0; i < D; ++i) {
    Y_dims.data[i] = dims[axes[i]];
  }
  ReduceTensorCUDAKernel<T, Reducer, D>
      <<<outer_size, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
          inner_size, X_strides, Y_dims, reducer, init, alpha, X, Y);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T, class Reducer>
void ReduceTensorCUDA(
    const int ndim,
    const int* X_dims,
    const int* Y_dims,
    const Reducer& reducer,
    const T init,
    const T alpha,
    const T* X,
    T* Y,
    CUDAContext* context) {
  CAFFE_ENFORCE(utils::CheckReduceDims(ndim, X_dims, Y_dims));
  const int X_size =
      std::accumulate(X_dims, X_dims + ndim, 1, std::multiplies<int>());
  const int Y_size =
      std::accumulate(Y_dims, Y_dims + ndim, 1, std::multiplies<int>());
  if (X_size == 0) {
    Set<T, CUDAContext>(Y_size, init * alpha, Y, context);
    return;
  }
  if (std::equal(X_dims, X_dims + ndim, Y_dims)) {
    Scale<T, T, CUDAContext>(X_size, alpha, X, Y, context);
    return;
  }
  int rows;
  int cols;
  if (utils::IsRowwiseReduce(ndim, X_dims, Y_dims, &rows, &cols)) {
    RowwiseReduceCUDAKernel<T, Reducer>
        <<<rows, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
            cols, reducer, init, alpha, X, Y);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  if (utils::IsColwiseReduce(ndim, X_dims, Y_dims, &rows, &cols)) {
    ColwiseReduceCUDAKernel<T, Reducer>
        <<<cols, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
            rows, cols, reducer, init, alpha, X, Y);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  int M;
  int N;
  int K;
  if (utils::IsBothEndsReduce(ndim, X_dims, Y_dims, &M, &N, &K)) {
    DISPATCH_REDUCE_KERNEL_BY_2D_BLOCK_WITH_TYPE_2(
        K,
        BothEndsReduceCUDAKernel,
        T,
        Reducer,
        N,
        context->cuda_stream(),
        M,
        N,
        K,
        reducer,
        init,
        alpha,
        X,
        Y);
    return;
  }
  std::vector<int> axes(ndim);
  utils::ComputeTransposeAxesForReduceOp(ndim, Y_dims, axes.data());
  const int outer_size = Y_size;
  const int inner_size = X_size / Y_size;
  DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_2(
      ndim,
      ReduceTensorCUDAImpl,
      T,
      Reducer,
      outer_size,
      inner_size,
      X_dims,
      axes.data(),
      reducer,
      init,
      alpha,
      X,
      Y,
      context);
}

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
    const SimpleArray<int, D> Y_dims,
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
      X_index += Y_index % Y_dims.data[d] * X_strides.data[d];
      Y_index /= Y_dims.data[d];
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
void MomentsCUDAImpl(
    const int outer_size,
    const int inner_size,
    const int* dims,
    const int* axes,
    const T* X,
    T* mean,
    T* var,
    CUDAContext* context) {
  SimpleArray<int, D> X_strides;
  SimpleArray<int, D> Y_dims;
  utils::ComputeTransposedStrides(D, dims, axes, X_strides.data);
  for (int i = 0; i < D; ++i) {
    Y_dims.data[i] = dims[axes[i]];
  }
  MomentsCUDAKernel<T, D>
      <<<outer_size, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
          inner_size, X_strides, Y_dims, X, mean, var);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void MomentsCUDA(
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
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  if (utils::IsColwiseReduce(ndim, X_dims, Y_dims, &rows, &cols)) {
    ColwiseMomentsCUDAKernel<T>
        <<<cols, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
            rows, cols, X, mean, var);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  int M;
  int N;
  int K;
  if (utils::IsBothEndsReduce(ndim, X_dims, Y_dims, &M, &N, &K)) {
    DISPATCH_REDUCE_KERNEL_BY_2D_BLOCK_WITH_TYPE_1(
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

#define DELEGATE_CUDA_REDUCE_FUNCTION(T, Func, Reducer, kInit)         \
  template <>                                                          \
  CAFFE2_CUDA_EXPORT void Func<T, CUDAContext>(                        \
      const int ndim,                                                  \
      const int* X_dims,                                               \
      const int* Y_dims,                                               \
      const T alpha,                                                   \
      const T* X,                                                      \
      T* Y,                                                            \
      CUDAContext* context) {                                          \
    ReduceTensorCUDA<T, Reducer>(                                      \
        ndim, X_dims, Y_dims, Reducer(), kInit, alpha, X, Y, context); \
  }
DELEGATE_CUDA_REDUCE_FUNCTION(
    std::int32_t,
    ReduceMin,
    cub::Min,
    std::numeric_limits<std::int32_t>::max())
DELEGATE_CUDA_REDUCE_FUNCTION(
    std::int64_t,
    ReduceMin,
    cub::Min,
    std::numeric_limits<std::int64_t>::max())
DELEGATE_CUDA_REDUCE_FUNCTION(
    float,
    ReduceMin,
    cub::Min,
    std::numeric_limits<float>::max())
DELEGATE_CUDA_REDUCE_FUNCTION(
    double,
    ReduceMin,
    cub::Min,
    std::numeric_limits<double>::max())
DELEGATE_CUDA_REDUCE_FUNCTION(
    std::int32_t,
    ReduceMax,
    cub::Max,
    std::numeric_limits<std::int32_t>::lowest())
DELEGATE_CUDA_REDUCE_FUNCTION(
    std::int64_t,
    ReduceMax,
    cub::Max,
    std::numeric_limits<std::int64_t>::lowest())
DELEGATE_CUDA_REDUCE_FUNCTION(
    float,
    ReduceMax,
    cub::Max,
    std::numeric_limits<float>::lowest())
DELEGATE_CUDA_REDUCE_FUNCTION(
    double,
    ReduceMax,
    cub::Max,
    std::numeric_limits<double>::lowest())
DELEGATE_CUDA_REDUCE_FUNCTION(std::int32_t, ReduceSum, cub::Sum, 0)
DELEGATE_CUDA_REDUCE_FUNCTION(std::int64_t, ReduceSum, cub::Sum, 0LL)
DELEGATE_CUDA_REDUCE_FUNCTION(float, ReduceSum, cub::Sum, 0.0f)
DELEGATE_CUDA_REDUCE_FUNCTION(double, ReduceSum, cub::Sum, 0.0)
#undef DELEGATE_CUDA_REDUCE_FUNCTION

#define CAFFE2_SPECIALIZED_CUDA_REDUCE_MEAN(T)        \
  template <>                                         \
  CAFFE2_CUDA_EXPORT void ReduceMean<T, CUDAContext>( \
      const int ndim,                                 \
      const int* X_dims,                              \
      const int* Y_dims,                              \
      const T alpha,                                  \
      const T* X,                                     \
      T* Y,                                           \
      CUDAContext* context) {                         \
    int scale = 1;                                    \
    for (int i = 0; i < ndim; ++i) {                  \
      if (Y_dims[i] == 1) {                           \
        scale *= X_dims[i];                           \
      }                                               \
    }                                                 \
    ReduceTensorCUDA<T, cub::Sum>(                    \
        ndim,                                         \
        X_dims,                                       \
        Y_dims,                                       \
        cub::Sum(),                                   \
        T(0),                                         \
        alpha / static_cast<T>(scale),                \
        X,                                            \
        Y,                                            \
        context);                                     \
  }
CAFFE2_SPECIALIZED_CUDA_REDUCE_MEAN(float)
#undef CAFFE2_SPECIALIZED_CUDA_REDUCE_MEAN

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
CAFFE2_SPECIALIZED_CUDA_MOMENTS(double)
#undef CAFFE2_SPECIALIZED_CUDA_MOMENTS

} // namespace math
} // namespace caffe2
