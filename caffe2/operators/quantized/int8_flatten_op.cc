#include "caffe2/operators/quantized/int8_flatten_op.h"

#include "caffe2/operators/flatten_op.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Int8Flatten, int8::Int8FlattenOp);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Int8Flatten)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(TensorInferenceForFlatten)
    .SetDoc(R"DOC(
Flattens the input tensor into a 2D matrix. If input tensor has shape
(d_0, d_1, ... d_n) then the output will have shape
(d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn)
)DOC")
    .Input(0, "input", "A Int8 tensor of rank >= axis.")
    .Output(
        0,
        "output",
        "A 2D Int8 tensor with the contents of the input tensor, "
        "with input dimensions up to axis flattened to the outer dimension "
        "of the output and remaining input dimensions flattened into the inner "
        "dimension of the output.")
    .Arg("Y_scale", "Output tensor quantization scale")
    .Arg("Y_zero_point", "Output tensor quantization offset")
    .Arg(
        "axis",
        "(Default to 1) Indicate up to which input dimensions "
        "(exclusive) should be flattened to the outer dimension of the output");

} // namespace caffe2
