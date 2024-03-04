#include "caffe2/operators/quantized/int8_transpose_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Int8Transpose, int8::Int8TransposeOp);

OPERATOR_SCHEMA(Int8Transpose)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Transpose the input tensor by permuting the axes of the input according
to the `axes` argument. Similar to numpy's
[transpose](https://docs.scipy.org/doc/numpy/reference/generated/numpy.transpose.html)
function.

For example, when axes=(1, 0, 2), given an input tensor of shape
(1, 2, 3), the output shape will be (2, 1, 3).
)DOC")
    .Arg(
        "axes",
        "*(type: Tuple(int))* Order to permute axes of input tensor. Reverses "
        "the dimensions by default.")
    .Arg("Y_scale", "Output tensor quantization scale")
    .Arg("Y_zero_point", "Output tensor quantization offset")
    .Input(0, "X", "Input tensor")
    .Output(0, "Y", "Transposed output");

} // namespace caffe2
