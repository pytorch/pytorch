#include "caffe2/operators/quantized/int8_sigmoid_op.h"

namespace caffe2 {
REGISTER_CPU_OPERATOR(Int8Sigmoid, int8::Int8SigmoidOp);

OPERATOR_SCHEMA(Int8Sigmoid)
    .NumInputs(1)
    .NumOutputs(1)
    .Arg("Y_scale", "Output tensor quantization scale")
    .Arg("Y_zero_point", "Output tensor quantization offset")
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Apply the Sigmoid function element-wise to the input tensor. This is often used
as a non-linear activation function in a neural network. The sigmoid function is
defined as:

$$Sigmoid(x) = \frac{1}{1+\exp(-x)}$$

Github Links:

- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/sigmoid_op.cc
)DOC")
    .Input(
        0,
        "input",
        "The input tensor that's coerced into a 2D matrix of size (NxD) "
        "as described above.")
    .Output(
        0,
        "output",
        "The sigmoid normalized output values with the same "
        "shape as input tensor.");
} // namespace caffe2
