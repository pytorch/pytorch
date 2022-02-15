#include "caffe2/operators/arg_ops.h"

#include <limits>

#include "caffe2/utils/cub_namespace.cuh"
#include <cub/block/block_reduce.cuh>

#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/fixed_divisor.h"

namespace caffe2 {

namespace {

template <typename K, typename V>
using KeyValuePair = cub::KeyValuePair<K, V>;

template <typename K, typename V>
using BlockReduce =
    cub::BlockReduce<KeyValuePair<K, V>, CAFFE_CUDA_NUM_THREADS>;

template <typename T, class Reducer>
__global__ void ComputeArgCUDAKernel(
    const int outer_size,
    const int inner_size,
    const FixedDivisor<int> stride,
    const Reducer reducer,
    const T init,
    const T* X,
    int64_t* Y) {
  __shared__ typename BlockReduce<int, T>::TempStorage temp_storage;
  const int d = stride.d();
  for (int idx = blockIdx.x; idx < outer_size; idx += gridDim.x) {
    int i;
    int j;
    stride.DivMod(idx, &i, &j);
    KeyValuePair<int, T> kv = {-1, init};
    for (int k = threadIdx.x; k < inner_size; k += blockDim.x) {
      kv = reducer({k, X[i * inner_size * d + k * d + j]}, kv);
    }
    kv = BlockReduce<int, T>(temp_storage).Reduce(kv, reducer);
    if (threadIdx.x == 0) {
      Y[idx] = static_cast<int64_t>(kv.key);
    }
    __syncthreads();
  }
}

} // namespace

template <>
template <typename T>
bool ArgMaxReducer<CUDAContext>::operator()(
    const int prev_size,
    const int next_size,
    const int n,
    const T* X,
    int64_t* Y,
    CUDAContext* context) const {
  const int outer_size = prev_size * next_size;
  const FixedDivisor<int> stride(next_size);
  ComputeArgCUDAKernel<<<
      std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context->cuda_stream()>>>(
      outer_size,
      n,
      stride,
      cub::ArgMax(),
      std::numeric_limits<T>::lowest(),
      X,
      Y);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

template <>
template <typename T>
bool ArgMinReducer<CUDAContext>::operator()(
    const int prev_size,
    const int next_size,
    const int n,
    const T* X,
    int64_t* Y,
    CUDAContext* context) const {
  const int outer_size = prev_size * next_size;
  const FixedDivisor<int> stride(next_size);
  ComputeArgCUDAKernel<<<
      std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context->cuda_stream()>>>(
      outer_size,
      n,
      stride,
      cub::ArgMin(),
      std::numeric_limits<T>::max(),
      X,
      Y);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

REGISTER_CUDA_OPERATOR(ArgMax, ArgOp<CUDAContext, ArgMaxReducer<CUDAContext>>);
REGISTER_CUDA_OPERATOR(ArgMin, ArgOp<CUDAContext, ArgMinReducer<CUDAContext>>);

} // namespace caffe2
