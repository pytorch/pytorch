#include "caffe2/core/context_gpu.h"
#include "caffe2/experiments/operators/tt_contraction_op.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(TTContraction, TTContractionOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    TTContractionGradient,
    TTContractionGradientOp<float, CUDAContext>);

} // namespace caffe2
