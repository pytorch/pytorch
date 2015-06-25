#include "caffe2/operators/depth_split_op.h"

namespace caffe2 {
namespace {
REGISTER_CPU_OPERATOR(DepthSplit, DepthSplitOp<float, CPUContext>)
REGISTER_CPU_OPERATOR(DepthConcat, DepthConcatOp<float, CPUContext>)
}  // namespace
}  // namespace caffe2

