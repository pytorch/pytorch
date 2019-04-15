#include "caffe2/operators/enforce_finite_op.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

template <>
template <typename T>
bool EnforceFiniteOp<CUDAContext>::DoRunWithType() {
  buffer_.CopyFrom(Input(0)); // sync copy
  EnforceOnCPU<T>(buffer_);
  return true;
}

REGISTER_CUDA_OPERATOR(EnforceFinite, EnforceFiniteOp<CUDAContext>);
} // namespace caffe2
