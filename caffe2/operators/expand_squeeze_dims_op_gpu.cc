#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/expand_squeeze_dims_op.h"

namespace caffe2 {
REGISTER_CUDA_OPERATOR(Squeeze, SqueezeOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(ExpandDims, ExpandDimsOp<CUDAContext>);
} // namespace caffe2
