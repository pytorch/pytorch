#include "caffe2/operators/exp_op.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(
    Exp,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        ExpFunctor<CUDAContext>>);

} // namespace caffe2
