#include "caffe2/operators/given_tensor_fill_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(GivenTensorFill, GivenTensorFillOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    GivenTensorDoubleFill,
    GivenTensorFillOp<double, CPUContext>);
REGISTER_CPU_OPERATOR(GivenTensorBoolFill, GivenTensorFillOp<bool, CPUContext>);
REGISTER_CPU_OPERATOR(GivenTensorIntFill, GivenTensorFillOp<int, CPUContext>);
REGISTER_CPU_OPERATOR(
    GivenTensorInt64Fill,
    GivenTensorFillOp<int64_t, CPUContext>);
REGISTER_CPU_OPERATOR(
    GivenTensorStringFill,
    GivenTensorFillOp<std::string, CPUContext>);

NO_GRADIENT(GivenTensorFill);
NO_GRADIENT(GivenTensorDoubleFill);
NO_GRADIENT(GivenTensorBoolFill);
NO_GRADIENT(GivenTensorIntFill);
NO_GRADIENT(GivenTensorInt64Fill);
NO_GRADIENT(GivenTensorStringFill);

OPERATOR_SCHEMA(GivenTensorFill)
    .NumInputs(0, 1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .Arg(
        "values",
        "The value for the elements of the output tensor.",
        true /* required */)
    .Arg(
        "dtype",
        "The data type for the elements of the output tensor."
        "Strictly must be one of the types from DataType enum in TensorProto.")
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
    .TensorInferenceFunction(FillerTensorInference<>);

OPERATOR_SCHEMA(GivenTensorDoubleFill)
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
        FillerTensorInference<TensorProto_DataType_DOUBLE>);

OPERATOR_SCHEMA(GivenTensorBoolFill)
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
    .TensorInferenceFunction(FillerTensorInference<TensorProto_DataType_BOOL>);

OPERATOR_SCHEMA(GivenTensorIntFill)
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
    .TensorInferenceFunction(FillerTensorInference<TensorProto_DataType_INT32>);

OPERATOR_SCHEMA(GivenTensorInt64Fill)
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
    .TensorInferenceFunction(FillerTensorInference<TensorProto_DataType_INT64>);

OPERATOR_SCHEMA(GivenTensorStringFill)
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
