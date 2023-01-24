#include "caffe2/operators/log_op.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(
    Log,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CUDAContext,
        LogFunctor<CUDAContext>>);

} // namespace caffe2
