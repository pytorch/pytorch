#include "caffe2/operators/elementwise_sub_op.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(
    Sub,
    BinaryElementwiseOp<NumericTypes, CUDAContext, SubFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    SubGradient,
    BinaryElementwiseGradientOp<
        NumericTypes,
        CUDAContext,
        SubFunctor<CUDAContext>>);

} // namespace caffe2
