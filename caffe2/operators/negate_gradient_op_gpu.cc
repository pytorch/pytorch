#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/negate_gradient_op.h"

namespace caffe2 {
REGISTER_CUDA_OPERATOR(NegateGradient, NegateGradientOp<CUDAContext>)
} // namespace caffe2
