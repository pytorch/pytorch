#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/operator_fallback_gpu.h"
#include "caffe2/operators/sparse_normalize_op.h"

namespace caffe2 {
REGISTER_CUDA_OPERATOR(
    SparseNormalize,
    GPUFallbackOp);
}
