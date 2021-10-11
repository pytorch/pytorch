#include <cub/block/block_reduce.cuh>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/one_hot_ops.h"

namespace caffe2 {

__global__ void OneHotOpKernel(
    const int64_t batch_size,
    const int64_t index_size,
    const int64_t* indices,
    float* output) {
  CUDA_1D_KERNEL_LOOP(i, batch_size) {
    output[i * index_size + indices[i]] = 1.;
  }
}

template <>
void OneHotOp<CUDAContext>::DoOneHotOp(
    int64_t batch_size,
    int64_t index_size,
    const Tensor& indices,
    Tensor* output) {
  float* output_ptr = output->template mutable_data<float>();
  math::Set<float, CUDAContext>(output->numel(), 0., output_ptr, &context_);
  OneHotOpKernel<<<
      CAFFE_GET_BLOCKS(batch_size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      batch_size, index_size, indices.data<int64_t>(), output_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

REGISTER_CUDA_OPERATOR(OneHot, OneHotOp<CUDAContext>);
} // namespace
