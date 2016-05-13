#include "caffe2/operators/accumulate_op.h"

namespace caffe2 {
namespace {
REGISTER_CPU_OPERATOR(Accumulate, AccumulateOp<float, CPUContext>);

OPERATOR_SCHEMA(Accumulate).NumInputs(1).NumOutputs(1);

SHOULD_NOT_DO_GRADIENT(Accumulate);
}  // namespace
}  // namespace caffe2
