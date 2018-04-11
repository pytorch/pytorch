#include "caffe2/operators/reduce_ops.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(ReduceSum, ReduceSumOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(ReduceMean, ReduceMeanOp<float, CUDAContext>);

} // namespace caffe2
