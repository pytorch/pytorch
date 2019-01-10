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

REGISTER_CPU_OPERATOR(IsMemberOf, IsMemberOfOp<CPUContext>);

// Input: X, output: Y
OPERATOR_SCHEMA(IsMemberOf)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(
        [](const OperatorDef&, const vector<TensorShape>& input_types) {
          vector<TensorShape> out(1);
          out[0] = input_types[0];
          out[0].set_data_type(TensorProto_DataType::TensorProto_DataType_BOOL);
          return out;
        })
    .Arg("value", "Declare one value for the membership test.")
    .Arg(
        "dtype",
        "The data type for the elements of the output tensor."
        "Strictly must be one of the types from DataType enum in TensorProto.")
    .SetDoc(R"DOC(
IsMemberOf takes input data (Tensor<T>) and a list of values as argument, and
produces one output data (Tensor<bool>) where the function `f(x) = x in values`,
is applied to the data tensor elementwise.
)DOC")
    .Input(0, "X", "Input tensor of any shape")
    .Output(0, "Y", "Output tensor (same size as X containing booleans)");

SHOULD_NOT_DO_GRADIENT(IsMemberOf);

} // namespace

template <>
std::unordered_set<int32_t>& IsMemberOfValueHolder::get<int32_t>() {
  return int32_values_;
}

template <>
std::unordered_set<int64_t>& IsMemberOfValueHolder::get<int64_t>() {
  return int64_values_;
}

template <>
std::unordered_set<bool>& IsMemberOfValueHolder::get<bool>() {
  return bool_values_;
}

template <>
std::unordered_set<string>& IsMemberOfValueHolder::get<string>() {
  return string_values_;
}

} // namespace caffe2
