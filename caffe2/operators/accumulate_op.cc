#include "caffe2/operators/accumulate_op.h"

namespace caffe2 {
namespace {
REGISTER_CPU_OPERATOR(Accumulate, AccumulateOp<float, CPUContext>)
}  // namespace
}  // namespace caffe2
