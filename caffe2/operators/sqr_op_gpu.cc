#include "caffe2/operators/sqr_op.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(
    Sqr,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        SqrFunctor<CUDAContext>>);

} // namespace caffe2
