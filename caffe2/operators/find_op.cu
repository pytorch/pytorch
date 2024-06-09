#include <cub/block/block_reduce.cuh>
#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/find_op.h"
#include "caffe2/utils/cub_namespace.cuh"

namespace caffe2 {

template <typename T>
__global__ void FindKernel(
    int num_needles,
    int idx_size,
    const T* idx,
    const T* needles,
    int* out,
    int missing_value) {
  int needle_idx = blockIdx.x; // One cuda block per needle
  T q = needles[needle_idx];
  int res = (-1);
  for (int j = threadIdx.x; j < idx_size; j += CAFFE_CUDA_NUM_THREADS) {
    if (idx[j] == q) {
      res = max(res, j);
    }
  }
  typedef cub::BlockReduce<int, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int min_res = BlockReduce(temp_storage).Reduce(res, cub::Max());
  if (threadIdx.x == 0) {
    out[needle_idx] = min_res == (-1) ? missing_value : min_res;
  }
}

template <>
template <typename T>
bool FindOp<CUDAContext>::DoRunWithType() {
  auto& idx = Input(0);
  auto& needles = Input(1);

  auto* res_indices = Output(0, needles.sizes(), at::dtype<int>());

  const T* idx_data = idx.data<T>();
  const T* needles_data = needles.data<T>();
  int* res_data = res_indices->template mutable_data<int>();

  FindKernel<
      T><<<needles.numel(), CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
      needles.numel(),
      idx.numel(),
      idx_data,
      needles_data,
      res_data,
      missing_value_);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

REGISTER_CUDA_OPERATOR(Find, FindOp<CUDAContext>)

} // namespace caffe2
