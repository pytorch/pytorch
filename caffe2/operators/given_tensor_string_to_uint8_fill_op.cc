#include "caffe2/operators/given_tensor_string_to_uint8_fill_op.h"

namespace caffe2 {
REGISTER_CPU_OPERATOR(
    GivenTensorStringToUInt8Fill,
    GivenTensorStringToUInt8FillOp<CPUContext>);

NO_GRADIENT(GivenTensorStringToUInt8Fill);

OPERATOR_SCHEMA(GivenTensorStringToUInt8Fill)
    .NumInputs(0, 1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .Arg(
        "values",
        "The value for the elements of the output tensor.",
        true /* required */)
    .Arg(
        "shape",
        "The shape of the output tensor."
        "Cannot set the shape argument and pass in an input at the same time.")
    .Arg(
        "extra_shape",
        "The additional dimensions appended at the end of the shape indicated"
        "by the input blob."
        "Cannot set the extra_shape argument when there is no input blob.")
    .Arg(
        "input_as_shape",
        "1D tensor containing the desired output shape. First input must be in CPU context.")
    .TensorInferenceFunction(
        FillerTensorInference<TensorProto_DataType_STRING>);

} // namespace caffe2
