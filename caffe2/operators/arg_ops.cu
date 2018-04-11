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

template <typename T, class ReduceOp>
__global__ void ComputeArgCUDAKernel(
    const T* X,
    const TIndex outer_size,
    const TIndex inner_size,
    const TIndex stride,
    const ReduceOp& reduce_op,
    const T init,
    TIndex* Y) {
  __shared__ typename BlockReduce<T>::TempStorage temp_storage;
  for (TIndex idx = blockIdx.x; idx < outer_size; idx += gridDim.x) {
    const TIndex i = idx / stride;
    const TIndex j = idx % stride;
    KeyValuePair<T> kv = {-1, init};
    for (TIndex k = threadIdx.x; k < inner_size; k += blockDim.x) {
      kv = reduce_op({k, X[i * inner_size * stride + k * stride + j]}, kv);
    }
    kv = BlockReduce<T>(temp_storage).Reduce(kv, reduce_op);
    if (threadIdx.x == 0) {
      Y[idx] = kv.key;
    }
    __syncthreads();
  }
}

} // namespace

template <typename T>
class ArgMaxOp<T, CUDAContext> final : public ArgOpBase<T, CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);

  ArgMaxOp(const OperatorDef& operator_def, Workspace* ws)
      : ArgOpBase<T, CUDAContext>(operator_def, ws) {}

 protected:
  bool Compute(
      const T* X,
      const TIndex prev_size,
      const TIndex next_size,
      const TIndex n,
      TIndex* Y) override;
};

template <typename T>
bool ArgMaxOp<T, CUDAContext>::Compute(
    const T* X,
    const TIndex prev_size,
    const TIndex next_size,
    const TIndex n,
    TIndex* Y) {
  const TIndex outer_size = prev_size * next_size;
  ComputeArgCUDAKernel<<<
      std::min(outer_size, static_cast<TIndex>(CAFFE_MAXIMUM_NUM_BLOCKS)),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      X,
      outer_size,
      n,
      next_size,
      cub::ArgMax(),
      std::numeric_limits<T>::lowest(),
      Y);
  return true;
}

template <typename T>
class ArgMinOp<T, CUDAContext> final : public ArgOpBase<T, CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);

  ArgMinOp(const OperatorDef& operator_def, Workspace* ws)
      : ArgOpBase<T, CUDAContext>(operator_def, ws) {}

 protected:
  bool Compute(
      const T* X,
      const TIndex prev_size,
      const TIndex next_size,
      const TIndex n,
      TIndex* Y) override;
};

template <typename T>
bool ArgMinOp<T, CUDAContext>::Compute(
    const T* X,
    const TIndex prev_size,
    const TIndex next_size,
    const TIndex n,
    TIndex* Y) {
  const TIndex outer_size = prev_size * next_size;
  ComputeArgCUDAKernel<<<
      std::min(outer_size, static_cast<TIndex>(CAFFE_MAXIMUM_NUM_BLOCKS)),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      X,
      outer_size,
      n,
      next_size,
      cub::ArgMin(),
      std::numeric_limits<T>::max(),
      Y);
  return true;
}

REGISTER_CUDA_OPERATOR(ArgMax, ArgMaxOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(ArgMin, ArgMinOp<float, CUDAContext>);

} // namespace caffe2
