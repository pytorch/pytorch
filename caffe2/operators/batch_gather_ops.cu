#include <fstream>
#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/batch_gather_ops.h"
// Shared batch kernel
#include "caffe2/operators/gather_op.cuh"

namespace caffe2 {

template <>
bool BatchGatherOp<CUDAContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
      this, OperatorBase::Input<Tensor>(INDICES, CUDA));
}

template <>
template <typename TInd>
bool BatchGatherOp<CUDAContext>::DoRunWithType() {
  // BatchGather is a special-case of Gather with Axis = 1, wrap = false.
  return gather_helper::gather_impl_cuda<TInd>(
      this, DATA, INDICES, 0, 1, false, match_outer_);
}

template <typename T_INDEX, typename TData>
__global__ void BatchGatherGradientKernel(
    const TData* grad_data,
    TData* out,
    const T_INDEX* indices,
    const int outer_dims_product,
    const int N,
    const int data_batch_size,
    const int gathered_batch_size,
    const int block_size,
    const int src_indexing_axis_dim,
    const bool wrap_indices) {
  int begin_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_items = outer_dims_product * N * block_size;

  for (int s = begin_idx; s < num_items; s += blockDim.x * gridDim.x) {
    const int k = s % block_size;
    const int j = s / block_size % N;
    const int i = s / block_size / N;
    T_INDEX idx = indices[j];
    if (wrap_indices && idx < 0) {
      idx = idx + src_indexing_axis_dim;
    }
    const float* src_offset =
        grad_data + i * gathered_batch_size + j * block_size;
    float* dst_offset = out + i * data_batch_size + idx * block_size;
    atomicAdd(dst_offset + k, src_offset[k]);
  }
}

template <>
bool BatchGatherGradientOp<CUDAContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
      this, OperatorBase::Input<Tensor>(INDICES, CUDA));
}

template <>
template <typename TInd>
bool BatchGatherGradientOp<CUDAContext>::DoRunWithType() {
  return DispatchHelper<
      TensorTypes2<float, GenericTensorImplementation>,
      TInd>::call(this, OperatorBase::Input<Tensor>(DATA, CUDA));
}

template <>
template <typename TInd, typename TData>
bool BatchGatherGradientOp<CUDAContext>::DoRunWithType2() {
  CAFFE_ENFORCE(
      !match_outer_, "match_outer=true is currently only supported for CPU");

  auto& data = Input(DATA);
  auto& indices = Input(INDICES);
  auto& grad = Input(GRAD);

  // ONNX allows negative axis to index from the back, valid range: [-r, r].
  int axis = axis_;
  if (axis < 0) {
    axis = data.dim() + axis;
  }
  // Outer dimensions of input data and gradient should be the same
  // because they are preserved for gathers with axis > 0.
  for (int acheck = 0; acheck < axis; acheck++) {
    CAFFE_ENFORCE_EQ(
        data.size(acheck), grad.size(acheck), "batch sizes should be the same");
  }

  auto* output = Output(0, data.sizes(), at::dtype<float>());
  auto* out_data = output->template mutable_data<float>();
  math::Set<float, CUDAContext>(output->numel(), 0, out_data, &context_);

  const auto* grad_data = grad.template data<float>();
  const TInd* idxs = indices.template data<TInd>();

  // Treat all outer dimensions as a unit as they contribute to larger batch.
  const int outer_dims_product = grad.size_to_dim(axis);
  const int block_size = data.size_from_dim(axis + 1);

  const int N = indices.numel();
  const auto data_batch_size = data.size_from_dim(axis);
  const auto gathered_batch_size = N * block_size;
  const int src_indexing_axis_dim = data.dim(axis);

  // Assign each thread index its own 'float' in block_size * N (kernel will
  // loop if there is more data than fits NUM_BLOCKS * NUM_THREADS limit).
  BatchGatherGradientKernel<<<
      std::min(outer_dims_product, CAFFE_MAXIMUM_NUM_BLOCKS),
      std::min(N * block_size, CAFFE_CUDA_NUM_THREADS),
      0,
      context_.cuda_stream()>>>(
      grad_data,
      out_data,
      idxs,
      outer_dims_product,
      N,
      data_batch_size,
      gathered_batch_size,
      block_size,
      src_indexing_axis_dim,
      false);
  C10_CUDA_KERNEL_LAUNCH_CHECK(); // TBD: Add proper index wrapping support to Gather gradients.

  return true;
}

template <>
template <typename TInd>
bool BatchGatherGradientOp<CUDAContext>::DoRunWithOtherType2() {
  CAFFE_THROW(
      "BatchGatherGradient is not implemented on tensor of type ",
      Input(DATA).meta().name(),
      "consider adding it as a type in the DispatchHelper list or implementing"
      " a generic version (which won't work for duplicated indices though)");
}

REGISTER_CUDA_OPERATOR(BatchGather, BatchGatherOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(BatchGatherGradient, BatchGatherGradientOp<CUDAContext>);

} // namespace caffe2
