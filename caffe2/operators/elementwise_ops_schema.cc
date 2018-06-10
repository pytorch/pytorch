#include "caffe2/operators/elementwise_ops.h"

#include "caffe2/core/operator_gradient.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

const char* kBroadcastDoc = R"DOC(
If necessary the right-hand-side argument will be broadcasted to match the
shape of left-hand-side argument. When broadcasting is specified, the second
tensor can either be of size 1 (a scalar value), or having its shape as a
contiguous subset of the first tensor's shape. The starting of the mutually
equal shape is specified by the argument "axis", and if it is not set, suffix
matching is assumed. 1-dim expansion doesn't work yet.

For example, the following tensor shapes are supported (with broadcast=1):

  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0

Argument `broadcast=1` needs to be passed to enable broadcasting.
)DOC";

std::function<void(OpSchema&)> MathDocGenerator(const char* name) {
  return [=](OpSchema& schema) {
    string doc = R"DOC(
Performs element-wise binary {name} (with limited broadcast support).
{broadcast_doc})DOC";
    ReplaceAll(doc, "{name}", name);
    ReplaceAll(doc, "{broadcast_doc}", kBroadcastDoc);
    schema.SetDoc(doc);
    schema.Arg("broadcast", "Pass 1 to enable broadcasting");
    schema.Arg(
        "axis",
        "If set, defines the broadcast dimensions. See doc for details.");
    schema.Input(
        0,
        "A",
        "First operand, should share the type with the second operand.");
    schema.Input(
        1,
        "B",
        "Second operand. With broadcasting can be of smaller size than A. "
        "If broadcasting is disabled it should be of the same size.");
    schema.Output(0, "C", "Result, has same dimensions and type as A");
  };
}

OPERATOR_SCHEMA(Add)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}, {1, 0}})
    .CostInferenceFunction(PointwiseCostInference<1>)
    .IdenticalTypeAndShapeOfInput(0)
    .FillUsing(MathDocGenerator("addition"))
    .InheritOnnxSchema("Add");
OPERATOR_SCHEMA(AddGradient)
    .NumInputs(3)
    .NumOutputs(2)
    .AllowInplace({{0, 0}, {1, 0}});

OPERATOR_SCHEMA(Sub)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}, {1, 0}})
    .CostInferenceFunction(PointwiseCostInference<1>)
    .IdenticalTypeAndShapeOfInput(0)
    .FillUsing(MathDocGenerator("subtraction"))
    .InheritOnnxSchema("Sub");
OPERATOR_SCHEMA(SubGradient)
    .NumInputs(3)
    .NumOutputs(2)
    .AllowInplace({{0, 0}, {1, 0}});

OPERATOR_SCHEMA(Mul)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}, {1, 0}})
    .CostInferenceFunction(PointwiseCostInference<1>)
    .IdenticalTypeAndShapeOfInput(0)
    .FillUsing(MathDocGenerator("multiplication"))
    .InheritOnnxSchema("Mul");
OPERATOR_SCHEMA(MulGradient)
    .NumInputs(3)
    .NumOutputs(2)
    .AllowInplace({{0, 0}, {1, 0}});

OPERATOR_SCHEMA(Div)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .CostInferenceFunction(PointwiseCostInference<1>)
    .IdenticalTypeAndShapeOfInput(0)
    .FillUsing(MathDocGenerator("division"))
    .InheritOnnxSchema("Div");
OPERATOR_SCHEMA(DivGradient).NumInputs(4).NumOutputs(2).AllowInplace({{0, 0}});

OPERATOR_SCHEMA(SumReduceLike)
    .NumInputs(2)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInput(0)
    .SetDoc(R"DOC(
SumReduceLike operator takes 2 tensors as input. It performs reduce sum to the
first input so that the output looks like the second one.
It assumes that the first input
has more dimensions than the second, and the dimensions of the second input is
the contiguous subset of the dimensions of the first.
For example, the following tensor shapes are supported:

  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 2, 5), shape(B) = (2), with axis=0
    )DOC")
    .Arg(
        "axis",
        "If set, defines the starting dimension for reduction. Args `axis` and "
        "`axis_str` cannot be used simultaneously.")
    .Arg(
        "axis_str",
        "If set, it could only be N or C or H or W. `order` arg should also be "
        "provided. It defines the reduction dimensions on NCHW or NHWC. Args "
        "`axis` and `axis_str` cannot be used simultaneously.")
    .Arg("order", "Either NHWC or HCWH")
    .Input(
        0,
        "A",
        "First operand, should share the type with the second operand.")
    .Input(
        1,
        "B",
        "Second operand. With broadcasting can be of smaller size than A. "
        "If broadcasting is disabled it should be of the same size.")
    .Output(0, "C", "Result, has same dimensions and type as B");

std::function<void(OpSchema&)> ComparisonDocGenerator(
    const char* name,
    const char* desc) {
  return [=](OpSchema& schema) {
    string doc = R"DOC(
Performs element-wise {desc} comparison `{name}` (with limited broadcast support).
{broadcast_doc})DOC";
    ReplaceAll(doc, "{name}", name);
    ReplaceAll(doc, "{desc}", desc);
    ReplaceAll(doc, "{broadcast_doc}", kBroadcastDoc);
    schema.SetDoc(doc);
    schema.Arg("broadcast", "Pass 1 to enable broadcasting");
    schema.Arg(
        "axis",
        "If set, defines the broadcast dimensions. See doc for details.");
    schema.Input(
        0,
        "A",
        "First operand, should share the type with the second operand.");
    schema.Input(
        1,
        "B",
        "Second operand. With broadcasting can be of smaller size than A. "
        "If broadcasting is disabled it should be of the same size.");
    schema.Output(0, "C", "Result, has same dimensions and A and type `bool`");
  };
}

#define CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(name, symbol, desc)             \
  OPERATOR_SCHEMA(name)                                                        \
      .NumInputs(2)                                                            \
      .NumOutputs(1)                                                           \
      .TensorInferenceFunction(                                                \
          [](const OperatorDef& def, const vector<TensorShape>& in) {          \
            ArgumentHelper helper(def);                                        \
            const auto broadcasted =                                           \
                helper.GetSingleArgument<bool>("broadcast", false);            \
            if (!broadcasted) {                                                \
              CAFFE_ENFORCE_EQ(in[0].dims().size(), in[1].dims().size());      \
              for (int i = 0; i < in[0].dims().size(); ++i) {                  \
                CAFFE_ENFORCE_EQ(in[0].dims(i), in[1].dims(i));                \
              }                                                                \
            }                                                                  \
            auto output_dims =                                                 \
                std::vector<TIndex>(in[0].dims().begin(), in[0].dims().end()); \
            return vector<TensorShape>{                                        \
                CreateTensorShape(output_dims, TensorProto::BOOL)};            \
          })                                                                   \
      .FillUsing(ComparisonDocGenerator(symbol, desc));                        \
  SHOULD_NOT_DO_GRADIENT(name)

CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(EQ, "==", "equal to");
CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(NE, "!=", "not equal to");
CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(LT, "<", "less than");
CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(LE, "<=", "less or equal than");
CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(GT, ">", "greater than");
CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(GE, ">=", "greater or equal than");

std::function<void(OpSchema&)> LogicalDocGenerator(const char* name) {
  return [=](OpSchema& schema) {
    string doc = R"DOC(
Performs element-wise logical operation `{name}` (with limited broadcast support).
Both input operands should be of type `bool`.
{broadcast_doc})DOC";
    ReplaceAll(doc, "{name}", name);
    ReplaceAll(doc, "{broadcast_doc}", kBroadcastDoc);
    schema.SetDoc(doc);
    schema.Arg("broadcast", "Pass 1 to enable broadcasting");
    schema.Arg(
        "axis",
        "If set, defines the broadcast dimensions. See doc for details.");
    schema.Input(0, "A", "First operand.");
    schema.Input(
        1,
        "B",
        "Second operand. With broadcasting can be of smaller size than A. "
        "If broadcasting is disabled it should be of the same size.");
    schema.Output(0, "C", "Result, has same dimensions and A and type `bool`");
  };
}

#define CAFFE2_SCHEMA_FOR_BINARY_LOGICAL_OP(name, symbol, onnx_schema) \
  OPERATOR_SCHEMA(name)                                                \
      .NumInputs(2)                                                    \
      .NumOutputs(1)                                                   \
      .AllowInplace({{0, 0}})                                          \
      .FillUsing(LogicalDocGenerator(symbol))                          \
      .InheritOnnxSchema(onnx_schema);                                 \
  SHOULD_NOT_DO_GRADIENT(name)

CAFFE2_SCHEMA_FOR_BINARY_LOGICAL_OP(Or, "or", "Or");
CAFFE2_SCHEMA_FOR_BINARY_LOGICAL_OP(And, "and", "And");
CAFFE2_SCHEMA_FOR_BINARY_LOGICAL_OP(Xor, "xor", "Xor");

#undef CAFFE2_SCHEMA_FOR_BINARY_LOGICAL_OP

std::function<void(OpSchema&)> BitwiseDocGenerator(const char* name) {
  return [=](OpSchema& schema) {
    string doc = R"DOC(
Performs element-wise bitwise operation `{name}` (with limited broadcast support).
Both input operands should be of type `bool`.
{broadcast_doc})DOC";
    ReplaceAll(doc, "{name}", name);
    ReplaceAll(doc, "{broadcast_doc}", kBroadcastDoc);
    schema.SetDoc(doc);
    schema.Arg("broadcast", "Pass 1 to enable broadcasting");
    schema.Arg(
        "axis",
        "If set, defines the broadcast dimensions. See doc for details.");
    schema.Input(0, "A", "First operand.");
    schema.Input(
        1,
        "B",
        "Second operand. With broadcasting can be of smaller size than A. "
        "If broadcasting is disabled it should be of the same size.");
    schema.Output(0, "C", "Result, has same dimensions and type with A.");
  };
}

#define CAFFE2_SCHEMA_FOR_BINARY_BITWISE_OP(name, symbol) \
  OPERATOR_SCHEMA(name)                                   \
      .NumInputs(2)                                       \
      .NumOutputs(1)                                      \
      .AllowInplace({{0, 0}})                             \
      .FillUsing(LogicalDocGenerator(symbol));            \
  SHOULD_NOT_DO_GRADIENT(name)

CAFFE2_SCHEMA_FOR_BINARY_BITWISE_OP(BitwiseOr, "bitwise_or");
CAFFE2_SCHEMA_FOR_BINARY_BITWISE_OP(BitwiseAnd, "bitwise_and");
CAFFE2_SCHEMA_FOR_BINARY_BITWISE_OP(BitwiseXor, "bitwise_xor");

#undef CAFFE2_SCHEMA_FOR_BINARY_BITWISE_OP

OPERATOR_SCHEMA(Not)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(Performs element-wise negation.)DOC")
    .Input(0, "X", "Input tensor of type `bool`.")
    .Output(0, "Y", "Output tensor of type `bool`.")
    .InheritOnnxSchema("Not");
SHOULD_NOT_DO_GRADIENT(Not);

OPERATOR_SCHEMA(Sign)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(Performs element-wise sign.)DOC")
    .Input(0, "X", "Input tensor.")
    .Output(0, "Y", "Output tensor.");
SHOULD_NOT_DO_GRADIENT(Sign);

} // namespace caffe2
