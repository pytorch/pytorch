#include "caffe2/core/operator_gradient.h"
#include "caffe2/operators/elementwise_op.h"
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
OPERATOR_SCHEMA(Sub)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}, {1, 0}})
    .CostInferenceFunction(PointwiseCostInference<1>)
    .IdenticalTypeAndShapeOfInput(0)
    .FillUsing(MathDocGenerator("subtraction"))
    .InheritOnnxSchema("Sub");
OPERATOR_SCHEMA(Mul)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}, {1, 0}})
    .CostInferenceFunction(PointwiseCostInference<1>)
    .IdenticalTypeAndShapeOfInput(0)
    .FillUsing(MathDocGenerator("multiplication"))
    .InheritOnnxSchema("Mul");
OPERATOR_SCHEMA(Div)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .CostInferenceFunction(PointwiseCostInference<1>)
    .IdenticalTypeAndShapeOfInput(0)
    .FillUsing(MathDocGenerator("division"))
    .InheritOnnxSchema("Div");
OPERATOR_SCHEMA(DivGradient).NumInputs(3).NumOutputs(2).AllowInplace({{0, 0}});

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

class GetAddGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    if (!ArgumentHelper::HasArgument(Def(), "broadcast")) {
      SetDense(0, GO(0));
      SetDense(1, GO(0));
      return vector<OperatorDef>();
    }
    SetDense(0, GO(0));

    return SingleGradientDef(
        "SumReduceLike",
        "",
        vector<string>{GO(0), I(1)},
        vector<string>{GI(1)});
  }
};
REGISTER_GRADIENT(Add, GetAddGradient);

// TODO(jiayq): Although we have Sub gradient implemented, we are still missing
// the Negative unary operator to be implemented.
class GetSubGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    if (!ArgumentHelper::HasArgument(Def(), "broadcast")) {
      SetDense(0, GO(0));
      return SingleGradientDef(
          "Negative", "", vector<string>{GO(0)}, vector<string>{GI(1)});
    } else {
      SetDense(0, GO(0));
      vector<OperatorDef> grad_ops;
      grad_ops.push_back(CreateOperatorDef(
          "Negative",
          "",
          vector<string>{GO(0)},
          vector<string>{GI(1) + "_autogen_pre_red"}));

      Argument axis, axis_str, order;
      if (ArgumentHelper::HasArgument(Def(), "axis")) {
        axis = GetArgument(Def(), "axis");
      } else {
        axis = MakeArgument<int>("axis", -1);
      }
      if (ArgumentHelper::HasArgument(Def(), "axis_str")) {
        axis_str = GetArgument(Def(), "axis_str");
      } else {
        axis_str = MakeArgument<string>("axis_str", "");
      }
      if (ArgumentHelper::HasArgument(Def(), "order")) {
        order = GetArgument(Def(), "order");
      } else {
        order = MakeArgument<string>("order", "NCHW");
      }
      grad_ops.push_back(CreateOperatorDef(
          "SumReduceLike",
          "",
          vector<string>{GI(1) + "_autogen_pre_red", I(1)},
          vector<string>{GI(1)},
          vector<Argument>{axis, axis_str, order}));

      return grad_ops;
    }
  }
  // Make sure the broadcast argument is not copied over.
  bool CopyArguments() const override {
    return false;
  }
};
REGISTER_GRADIENT(Sub, GetSubGradient);

class GetMulGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE(
        Def().input(0) != Def().output(0) && Def().input(1) != Def().output(0),
        "Gradient computation cannot be carried out if Mul uses in-place "
        "computation: ",
        ProtoDebugString(Def()));
    if (!ArgumentHelper::HasArgument(Def(), "broadcast")) {
      return vector<OperatorDef>{
          CreateOperatorDef(
              "Mul", "", vector<string>{GO(0), I(1)}, vector<string>{GI(0)}),
          CreateOperatorDef(
              "Mul", "", vector<string>{GO(0), I(0)}, vector<string>{GI(1)})};
    } else {
      Argument broadcast, axis, axis_str, order;
      if (ArgumentHelper::HasArgument(Def(), "broadcast")) {
        broadcast = GetArgument(Def(), "broadcast");
      } else {
        broadcast = MakeArgument<int>("broadcast", 0);
      }
      if (ArgumentHelper::HasArgument(Def(), "axis")) {
        axis = GetArgument(Def(), "axis");
      } else {
        axis = MakeArgument<int>("axis", -1);
      }
      if (ArgumentHelper::HasArgument(Def(), "axis_str")) {
        axis_str = GetArgument(Def(), "axis_str");
      } else {
        axis_str = MakeArgument<string>("axis_str", "");
      }
      if (ArgumentHelper::HasArgument(Def(), "order")) {
        order = GetArgument(Def(), "order");
      } else {
        order = MakeArgument<string>("order", "NCHW");
      }

      vector<OperatorDef> grad_ops;
      grad_ops.push_back(CreateOperatorDef(
          "Mul",
          "mul_grad_1st_op",
          vector<string>{GO(0), I(1)},
          vector<string>{GI(0)},
          vector<Argument>{broadcast, axis, axis_str, order}));
      grad_ops.push_back(CreateOperatorDef(
          "Mul",
          "mul_gradient_2nd_op",
          vector<string>{GO(0), I(0)},
          vector<string>{GI(1) + "_autogen_pre_red"}));

      grad_ops.push_back(CreateOperatorDef(
          "SumReduceLike",
          "mul_with_broadcast_grad_3",
          vector<string>{GI(1) + "_autogen_pre_red", I(1)},
          vector<string>{GI(1)},
          vector<Argument>{axis, axis_str, order}));

      return grad_ops;
    }
  }

  // Make sure the broadcast argument is not copied over.
  bool CopyArguments() const override {
    return false;
  }
};
REGISTER_GRADIENT(Mul, GetMulGradient);

class GetDivGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE(
        !ArgumentHelper::HasArgument(Def(), "broadcast"),
        "Gradient not ready yet for Div with broadcasting.");
    return SingleGradientDef(
        "DivGradient",
        "",
        vector<string>{I(1), O(0), GO(0)},
        vector<string>{GI(0), GI(1)});
  }
};
REGISTER_GRADIENT(Div, GetDivGradient);

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

#define CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(name, symbol, desc) \
  OPERATOR_SCHEMA(name).NumInputs(2).NumOutputs(1).FillUsing(      \
      ComparisonDocGenerator(symbol, desc));                       \
  SHOULD_NOT_DO_GRADIENT(name)

CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(LT, "<", "less than");
CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(LE, "<=", "less or equal than");
CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(GT, ">", "greater than");
CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(GE, ">=", "greater or equal than");
CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(EQ, "==", "equality");

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
  OPERATOR_SCHEMA(name)                                   \
      .NumInputs(2)                                       \
      .NumOutputs(1)                                      \
      .AllowInplace({{0, 0}})                             \
      .FillUsing(LogicalDocGenerator(symbol))             \
      .InheritOnnxSchema(onnx_schema);                    \
  SHOULD_NOT_DO_GRADIENT(name)

CAFFE2_SCHEMA_FOR_BINARY_LOGICAL_OP(Or, "or", "Or");
CAFFE2_SCHEMA_FOR_BINARY_LOGICAL_OP(And, "and", "And");
CAFFE2_SCHEMA_FOR_BINARY_LOGICAL_OP(Xor, "xor", "Xor");

OPERATOR_SCHEMA(Not)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(Performs element-wise negation.)DOC")
    .Input(0, "X", "Input tensor of type `bool`.")
    .Output(0, "Y", "Output tensor of type `bool`.")
    .InheritOnnxSchema("Not");
SHOULD_NOT_DO_GRADIENT(Not);

} // namespace caffe2
