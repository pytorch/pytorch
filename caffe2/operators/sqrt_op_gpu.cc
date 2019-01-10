#include "caffe2/operators/sqrt_op.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(
    Sqrt,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        SqrtFunctor<CUDAContext>>);

} // namespace caffe2
