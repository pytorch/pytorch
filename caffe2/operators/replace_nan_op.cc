#include "caffe2/operators/replace_nan_op.h"

namespace caffe2 {

namespace {
REGISTER_CPU_OPERATOR(ReplaceNaN, ReplaceNaNOp<CPUContext>);

OPERATOR_SCHEMA(ReplaceNaN)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Replace the NaN (not a number) element in the input tensor with argument `value`
)DOC")
    .Arg("value (optional)", "the value to replace NaN, the default is 0")
    .Input(0, "input", "Input tensor")
    .Input(1, "output", "Output tensor");

SHOULD_NOT_DO_GRADIENT(ReplaceNaN);

} // namespace
} // namespace caffe2
