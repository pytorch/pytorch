#include "int8_given_tensor_fill_op.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Int8GivenTensorFill)
    .NumInputs(0)
    .NumOutputs(1)
    .Arg("values", "Input array of type char(byte)")
    .Arg("shape", "Input tensor shape")
    .Arg("Y_scale", "Output tensor quantization scale")
    .Arg("Y_zero_point", "Output tensor quantization offset")
    .SetDoc(R"DOC(
    Creates quantized tensor of type char(byte) with scale and zero point info.
)DOC")
    .Output(0, "Tensor", "An Int8TensorCPU with scale and zero point info")
    .TensorInferenceFunction(FillerTensorInference<>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Int8GivenIntTensorFill)
    .NumInputs(0)
    .NumOutputs(1)
    .Arg("values", "Input array of type int32")
    .Arg("shape", "Input tensor shape")
    .Arg("Y_scale", "Output tensor quantization scale")
    .Arg("Y_zero_point", "Output tensor quantization offset")
    .SetDoc(R"DOC(
    Creates quantized tensor of type int32 with scale and zero point info.
)DOC")
    .Output(0, "Tensor", "An Int8TensorCPU with scale and zero point info")
    .TensorInferenceFunction(FillerTensorInference<>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Int8GivenTensorFill, int8::Int8GivenTensorFillOp);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Int8GivenIntTensorFill, int8::Int8GivenIntTensorFillOp);

} // namespace caffe2
