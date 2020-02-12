#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/bucketize_op.h"
#include "caffe2/operators/operator_fallback_gpu.h"

namespace caffe2 {
REGISTER_CUDA_OPERATOR(Bucketize, GPUFallbackOp);
} // namespace caffe2
