#include "caffe2/operators/quantized/int8_leaky_relu_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Int8LeakyRelu, int8::Int8LeakyReluOp);

OPERATOR_SCHEMA(Int8LeakyRelu)
    .NumInputs(1)
    .NumOutputs(1)
    .Arg("alpha", "Coefficient of leakage, default value is 0.01")
    .Arg("Y_scale", "Output tensor quantization scale")
    .Arg("Y_zero_point", "Output tensor quantization offset")
    .AllowInplace({{0, 0}})
    .CostInferenceFunction(PointwiseCostInference<2>)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
`f(x) = x for x >= 0`, is applied to the data tensor elementwise.
)DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D input tensor");

} // namespace caffe2
