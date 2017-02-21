#include "caffe2/operators/elementwise_power_op.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(
    ElementwisePower,
    UnaryElementwiseWithArgsOp<TensorTypes<float>, CPUContext, PowCPUFunctor>);

OPERATOR_SCHEMA(ElementwisePower)
    .NumInputs(1)
    .NumOutputs(1)
    .Arg("exponent", "The exponent of the power function.")
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
ElementwisePower takes input data (Tensor<T>) and an argument exponent, and
produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
is applied to the data tensor elementwise.
)DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D input tensor");

} // namespace
} // namespace caffe2
