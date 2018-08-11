#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/upsample_op.h"
#include "caffe2/operators/operator_fallback_gpu.h"

namespace caffe2 {

namespace {}  // namespace

REGISTER_CUDA_OPERATOR(UpsampleBilinear,
		GPUFallbackOp<UpsampleBilinearOp<float, CPUContext>>);
REGISTER_CUDA_OPERATOR(UpsampleBilinearGradient,
		GPUFallbackOp<UpsampleBilinearGradientOp<float, CPUContext>>);

}  // namespace caffe2
