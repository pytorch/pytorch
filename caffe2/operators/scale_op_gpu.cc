#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/scale_op.h"

namespace caffe2 {

template <>
bool ScaleOp<CUDAContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<at::Half, float>>::call(this, Input(0));
}

REGISTER_CUDA_OPERATOR(Scale, ScaleOp<CUDAContext>);

}  // namespace caffe2
