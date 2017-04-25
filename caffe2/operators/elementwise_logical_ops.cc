#include "caffe2/operators/elementwise_logical_ops.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(Where, WhereOp<CPUContext>);

// Input: C, X, Y, output: Z
OPERATOR_SCHEMA(Where)
    .NumInputs(3)
    .NumOutputs(1)
    .AllowInplace({{1, 2}})
    .IdenticalTypeAndShapeOfInput(1)
    .SetDoc(R"DOC(
Operator Where takes three input data (Tensor<bool>, Tensor<T>, Tensor<T>) and
produces one output data (Tensor<T>) where z = c ? x : y is applied elementwise.
)DOC")
    .Input(0, "C", "input tensor containing booleans")
    .Input(1, "X", "input tensor")
    .Input(2, "Y", "input tensor")
    .Output(0, "Z", "output tensor");

SHOULD_NOT_DO_GRADIENT(Where);

} // namespace
} // namespace caffe2
