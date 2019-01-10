#include "caffe2/operators/assert_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Assert, AssertOp<CPUContext>);

OPERATOR_SCHEMA(Assert)
    .NumInputs(1)
    .NumOutputs(0)
    .SetDoc(R"DOC(
Assertion op. Takes in a tensor of bools, ints, longs, or long longs and checks
if all values are true when coerced into a boolean. In other words, for non-bool
types this asserts that all values in the tensor are non-zero.
	)DOC")
    .Arg(
        "error_msg",
        "An error message to print when the assert fails.",
        false);

} // namespace caffe2
