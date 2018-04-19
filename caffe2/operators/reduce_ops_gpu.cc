#include "caffe2/operators/reduce_ops.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(ReduceSum, ReduceSumOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    ReduceSumGradient,
    ReduceSumGradientOp<float, CUDAContext>);

REGISTER_CUDA_OPERATOR(ReduceMean, ReduceMeanOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    ReduceMeanGradient,
    ReduceMeanGradientOp<float, CUDAContext>);

} // namespace caffe2
