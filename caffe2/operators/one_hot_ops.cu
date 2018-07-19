#include <cub/block/block_reduce.cuh>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/one_hot_ops.h"

namespace caffe2 {

__global__ void OneHotOpKernel(
    const TIndex batch_size,
    const TIndex index_size,
    const TIndex* indices,
    float* output) {
  CUDA_1D_KERNEL_LOOP(i, batch_size) {
    output[i * index_size + indices[i]] = 1.;
  }
}

template <>
void OneHotOp<CUDAContext>::DoOneHotOp(
    TIndex batch_size,
    TIndex index_size,
    const Tensor<CUDAContext>& indices,
    Tensor<CUDAContext>* output) {
  float* output_ptr = output->mutable_data<float>();
  math::Set<float, CUDAContext>(output->size(), 0., output_ptr, &context_);
  OneHotOpKernel<<<
      CAFFE_GET_BLOCKS(batch_size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      batch_size, index_size, indices.data<TIndex>(), output_ptr);
}

REGISTER_CUDA_OPERATOR(OneHot, OneHotOp<CUDAContext>);
} // namespace
