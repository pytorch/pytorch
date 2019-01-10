#include "caffe2/operators/arg_ops.h"

#include <limits>

#include <cub/block/block_reduce.cuh>
#include <cub/cub.cuh>

#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {

template <typename T>
using KeyValuePair = cub::KeyValuePair<TIndex, T>;

template <typename T>
using BlockReduce = cub::BlockReduce<KeyValuePair<T>, CAFFE_CUDA_NUM_THREADS>;

template <typename T, class Reducer>
__global__ void ComputeArgCUDAKernel(
    const TIndex outer_size,
    const TIndex inner_size,
    const TIndex stride,
    const Reducer reducer,
    const T init,
    const T* X,
    TIndex* Y) {
  __shared__ typename BlockReduce<T>::TempStorage temp_storage;
  for (TIndex idx = blockIdx.x; idx < outer_size; idx += gridDim.x) {
    const TIndex i = idx / stride;
    const TIndex j = idx % stride;
    KeyValuePair<T> kv = {-1, init};
    for (TIndex k = threadIdx.x; k < inner_size; k += blockDim.x) {
      kv = reducer({k, X[i * inner_size * stride + k * stride + j]}, kv);
    }
    kv = BlockReduce<T>(temp_storage).Reduce(kv, reducer);
    if (threadIdx.x == 0) {
      Y[idx] = kv.key;
    }
    __syncthreads();
  }
}

} // namespace

template <>
template <typename T>
bool ArgMaxReducer<CUDAContext>::operator()(
    const TIndex prev_size,
    const TIndex next_size,
    const TIndex n,
    const T* X,
    TIndex* Y,
    CUDAContext* context) const {
  const TIndex outer_size = prev_size * next_size;
  ComputeArgCUDAKernel<<<
      std::min(outer_size, static_cast<TIndex>(CAFFE_MAXIMUM_NUM_BLOCKS)),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context->cuda_stream()>>>(
      outer_size,
      n,
      next_size,
      cub::ArgMax(),
      std::numeric_limits<T>::lowest(),
      X,
      Y);
  return true;
}

template <>
template <typename T>
bool ArgMinReducer<CUDAContext>::operator()(
    const TIndex prev_size,
    const TIndex next_size,
    const TIndex n,
    const T* X,
    TIndex* Y,
    CUDAContext* context) const {
  const TIndex outer_size = prev_size * next_size;
  ComputeArgCUDAKernel<<<
      std::min(outer_size, static_cast<TIndex>(CAFFE_MAXIMUM_NUM_BLOCKS)),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context->cuda_stream()>>>(
      outer_size,
      n,
      next_size,
      cub::ArgMin(),
      std::numeric_limits<T>::max(),
      X,
      Y);
  return true;
}

REGISTER_CUDA_OPERATOR(ArgMax, ArgOp<CUDAContext, ArgMaxReducer<CUDAContext>>);
REGISTER_CUDA_OPERATOR(ArgMin, ArgOp<CUDAContext, ArgMinReducer<CUDAContext>>);

} // namespace caffe2
