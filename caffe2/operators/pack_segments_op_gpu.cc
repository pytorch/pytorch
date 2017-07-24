#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/operator_fallback_gpu.h"
#include "caffe2/operators/pack_segments.h"

namespace caffe2 {
REGISTER_CUDA_OPERATOR(PackSegments, GPUFallbackOp<PackSegmentsOp<CPUContext>>);
REGISTER_CUDA_OPERATOR(
    UnpackSegments,
    GPUFallbackOp<UnpackSegmentsOp<CPUContext>>);
}
