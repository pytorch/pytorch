#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/filler_op.h"
#include "caffe2/operators/operator_fallback_gpu.h"

namespace caffe2 {
REGISTER_CUDA_OPERATOR(
    LengthsRangeFill,
    GPUFallbackOp<LengthsRangeFillOp<CPUContext>>);
}
