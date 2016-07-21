#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/concat_split_op.h"

namespace caffe2 {
namespace {
REGISTER_CUDA_OPERATOR(Split, SplitOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(Concat, ConcatOp<CUDAContext>);

// Backward compatibility settings
REGISTER_CUDA_OPERATOR(DepthSplit, SplitOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(DepthConcat, ConcatOp<CUDAContext>);
}  // namespace
}  // namespace caffe2

