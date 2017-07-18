#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/zero_gradient_op.h"

namespace caffe2 {
namespace {
REGISTER_CUDA_OPERATOR(ZeroGradient, ZeroGradientOp<CUDAContext>);
}
}
