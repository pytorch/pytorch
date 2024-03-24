#include "caffe2/contrib/fakelowp/int8_swish_op_nnpi.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(SwishFakeInt8NNPI, int8::SwishInt8NNPIOp);

OPERATOR_SCHEMA(SwishFakeInt8NNPI)
    .IdenticalTypeAndShape()
    .Arg("X_scale", "Inout tensor quantization scale")
    .Arg("X_zero_point", "Input tensor quantization offset")
    .Arg("Y_scale", "Output tensor quantization scale")
    .Arg("Y_zero_point", "Output tensor quantization offset")
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Apply the Swish function element-wise after dequantizing input tensor.
$$Swish(x) = \frac{x}{1+\exp(-x)}$$
Quantize the Swish function output back to Int8.
The input and output of this operator are converted to fp16 precision
before applying the function.
<details>
</details>
)DOC")
    .Input(0, "X", "Int8 Tensor X.")
    .Output(0, "Y", "Int8 Tensor Y.");

} // namespace caffe2
