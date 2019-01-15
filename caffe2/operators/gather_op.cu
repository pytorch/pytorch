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
    return DispatchHelper<TensorTypes2<int8_t,int16_t,int32_t,int64_t,long,float,double,
           GenericTensorImplementation>,Index>::call(this, Input(DATA));
}

template <>
template <typename Index, typename TData>
bool GatherOp<CUDAContext>::DoRunWithType2() {
  // Use shared implementation with BatchGather
  return gather_helper::gather_impl_cuda<Index,TData>(
      this, DATA, INDICES, 0, axis_, wrap_indices_);
}

REGISTER_CUDA_OPERATOR(Gather, GatherOp<CUDAContext>);
} // namespace caffe2
