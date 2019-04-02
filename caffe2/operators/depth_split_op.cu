#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/depth_split_op.h"

namespace caffe2 {
namespace {
REGISTER_CUDA_OPERATOR(DepthSplit, DepthSplitOp<float, CUDAContext>)
REGISTER_CUDA_OPERATOR(DepthConcat, DepthConcatOp<float, CUDAContext>)
}  // namespace
}  // namespace caffe2

