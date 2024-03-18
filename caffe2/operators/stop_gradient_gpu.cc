#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/stop_gradient.h"

namespace caffe2 {
REGISTER_CUDA_OPERATOR(StopGradient, StopGradientOp<CUDAContext>);
}  // namespace caffe2
