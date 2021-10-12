#include "caffe2/operators/while_op.h"

namespace caffe2 {

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
