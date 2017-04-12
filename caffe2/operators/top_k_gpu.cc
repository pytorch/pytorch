#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/operator_fallback_gpu.h"
#include "caffe2/operators/top_k.h"

namespace caffe2 {
namespace {
REGISTER_CUDA_OPERATOR(TopK, GPUFallbackOp<TopKOp<float, CPUContext>>);
}
}
