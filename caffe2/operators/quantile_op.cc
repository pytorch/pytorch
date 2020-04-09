#include "quantile_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Quantile, QuantileOp<CPUContext>);
OPERATOR_SCHEMA(Quantile)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1)
    .SetDoc(R"DOC(
    Calculate the quantile for the value in the given list of tensors.
)DOC")
    .Input(0, "X1, X2, ...", "*(type: Tensor`<float>`)* List of input tensors.")
    .Output(0, "quantile_value", "Value at the given quantile")
    .Arg("abs", "If true (default), apply abs() on the tensor values.")
    .Arg("tol", "multiplicative tolerance of the quantile_value.");

SHOULD_NOT_DO_GRADIENT(Quantile);
} // namespace caffe2
