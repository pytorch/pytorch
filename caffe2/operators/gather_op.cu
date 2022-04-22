#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/gather_op.h"
#include "caffe2/operators/gather_op.cuh"

namespace caffe2 {

template <>
bool GatherOp<CUDAContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
      this, OperatorBase::Input<Tensor>(INDICES, CUDA));
}

template <>
template <typename Index>
bool GatherOp<CUDAContext>::DoRunWithType() {
  // Use shared implementation with BatchGather
  return gather_helper::gather_impl_cuda<Index>(
      this, DATA, INDICES, 0, axis_, wrap_indices_, match_outer_);
}

REGISTER_CUDA_OPERATOR(Gather, GatherOp<CUDAContext>);
} // namespace caffe2
