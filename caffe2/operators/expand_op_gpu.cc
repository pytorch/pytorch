#include "caffe2/operators/expand_op.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(
    Expand,
    ExpandOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        CUDAContext>);
REGISTER_CUDA_OPERATOR(
    ExpandGradient,
    ExpandGradientOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        CUDAContext>);
} // namespace caffe2
