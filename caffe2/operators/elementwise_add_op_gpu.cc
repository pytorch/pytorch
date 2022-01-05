#include "caffe2/operators/elementwise_add_op.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(
    Add,
    BinaryElementwiseOp<NumericTypes, CUDAContext, AddFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    AddGradient,
    BinaryElementwiseGradientOp<
        NumericTypes,
        CUDAContext,
        AddFunctor<CUDAContext>>);

} // namespace caffe2
