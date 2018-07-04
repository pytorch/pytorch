#include "caffe2/operators/negative_op.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(
    Negative,
    UnaryElementwiseOp<
        NumericTypes,
        CUDAContext,
        NegativeFunctor<CUDAContext>>);

} // namespace caffe2
