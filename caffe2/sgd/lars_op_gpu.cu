#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/operator_fallback_gpu.h"
#include "caffe2/sgd/lars_op.h"

namespace caffe2 {
REGISTER_CUDA_OPERATOR(Lars, GPUFallbackOp<LarsOp<float, CPUContext>>);
}
