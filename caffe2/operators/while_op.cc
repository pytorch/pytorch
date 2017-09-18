#include "caffe2/operators/while_op.h"

namespace caffe2 {

template <>
bool WhileOp<CPUContext>::RunOnDevice() {
  CAFFE_ENFORCE(
      InputIsType<Tensor<CPUContext>>(0),
      "Invalid condition in While operator: tensor expected");

  const auto& condition = Input(0);
  CAFFE_ENFORCE_EQ(
      condition.size(),
      1,
      "Invalid condition tensor in While operator: single value expected");

  while (true) {
    if (cond_net_ && !cond_net_->Run()) {
      return false;
    }
    if (!*condition.data<bool>()) {
      return true;
    }
    if (!loop_net_->Run()) {
      return false;
    }
  }

  return true;
}

REGISTER_CPU_OPERATOR(While, WhileOp<CPUContext>);

OPERATOR_SCHEMA(While)
    .NumInputs(1, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .SetDoc(R"DOC(
'While' control operator, first input is a scalar boolean blob that stores loop's
condition value. Accepts 'loop_net' (required) and 'cond_net' (optional) arguments for
loop's body and condition subnets respectively. If condition subnet is specified,
it is executed before the first and after each iteration. Subnets are executed in
the same workspace as 'While'.
    )DOC")
    .Arg("loop_net", "Net executed on each iteration")
    .Arg("cond_net", "Net to (re)compute condition value")
    .Input(0, "condition", "Scalar boolean condition")
    .AllowInplace([](int in, int out) -> bool { return true; });

} // namespace caffe2
