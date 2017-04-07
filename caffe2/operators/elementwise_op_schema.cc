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
    .FillUsing(MathDocGenerator("addition"));
OPERATOR_SCHEMA(Sub)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}, {1, 0}})
    .FillUsing(MathDocGenerator("subtraction"));
OPERATOR_SCHEMA(Mul)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}, {1, 0}})
    .FillUsing(MathDocGenerator("multiplication"));
OPERATOR_SCHEMA(Div)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .FillUsing(MathDocGenerator("division"));
OPERATOR_SCHEMA(DivGradient).NumInputs(3).NumOutputs(2).AllowInplace({{0, 0}});

class GetAddGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    if (!HasArgument(Def(), "broadcast")) {
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
    if (!HasArgument(Def(), "broadcast")) {
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

      Argument broadcast, axis, axis_str, order;
      if (HasArgument(Def(), "broadcast")) {
        broadcast = GetArgument(Def(), "broadcast");
      } else {
        broadcast = MakeArgument<int>("broadcast", 0);
      }
      if (HasArgument(Def(), "axis")) {
        axis = GetArgument(Def(), "axis");
      } else {
        axis = MakeArgument<int>("axis", -1);
      }
      if (HasArgument(Def(), "axis_str")) {
        axis_str = GetArgument(Def(), "axis_str");
      } else {
        axis_str = MakeArgument<string>("axis_str", "");
      }
      if (HasArgument(Def(), "order")) {
        order = GetArgument(Def(), "order");
      } else {
        order = MakeArgument<string>("order", "NCHW");
      }
      grad_ops.push_back(CreateOperatorDef(
          "SumReduceLike",
          "",
          vector<string>{GI(1) + "_autogen_pre_red", I(1)},
          vector<string>{GI(1)},
          vector<Argument>{broadcast, axis, axis_str, order}));

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
    if (!HasArgument(Def(), "broadcast")) {
      return vector<OperatorDef>{
          CreateOperatorDef(
              "Mul", "", vector<string>{GO(0), I(1)}, vector<string>{GI(0)}),
          CreateOperatorDef(
              "Mul", "", vector<string>{GO(0), I(0)}, vector<string>{GI(1)})};
    } else {
      Argument broadcast, axis, axis_str, order;
      if (HasArgument(Def(), "broadcast")) {
        broadcast = GetArgument(Def(), "broadcast");
      } else {
        broadcast = MakeArgument<int>("broadcast", 0);
      }
      if (HasArgument(Def(), "axis")) {
        axis = GetArgument(Def(), "axis");
      } else {
        axis = MakeArgument<int>("axis", -1);
      }
      if (HasArgument(Def(), "axis_str")) {
        axis_str = GetArgument(Def(), "axis_str");
      } else {
        axis_str = MakeArgument<string>("axis_str", "");
      }
      if (HasArgument(Def(), "order")) {
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
          vector<Argument>{broadcast, axis, axis_str, order}));

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
        !HasArgument(Def(), "broadcast"),
        "Gradient not ready yet for Div with broadcasting.");
    return SingleGradientDef(
        "DivGradient",
        "",
        vector<string>{I(1), O(0), GO(0)},
        vector<string>{GI(0), GI(1)});
  }
};
REGISTER_GRADIENT(Div, GetDivGradient);

std::function<void(OpSchema&)> ComparisonDocGenerator(const char* name) {
  return [=](OpSchema& schema) {
    string doc = R"DOC(
Performs element-wise comparison `{name}` (with limited broadcast support).
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
    schema.Output(0, "C", "Result, has same dimensions and A and type `bool`");
  };
}

#define CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(name, symbol)  \
  OPERATOR_SCHEMA(name).NumInputs(2).NumOutputs(1).FillUsing( \
      ComparisonDocGenerator(symbol));                        \
  SHOULD_NOT_DO_GRADIENT(name)

CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(LT, "<");
CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(LE, "<=");
CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(GT, ">");
CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(GE, ">=");
CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(EQ, "==");

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

#define CAFFE2_SCHEMA_FOR_BINARY_LOGICAL_OP(name, symbol) \
  OPERATOR_SCHEMA(name)                                   \
      .NumInputs(2)                                       \
      .NumOutputs(1)                                      \
      .AllowInplace({{0, 0}})                             \
      .FillUsing(LogicalDocGenerator(symbol));            \
  SHOULD_NOT_DO_GRADIENT(name)

CAFFE2_SCHEMA_FOR_BINARY_LOGICAL_OP(Or, "or");
CAFFE2_SCHEMA_FOR_BINARY_LOGICAL_OP(And, "and");
CAFFE2_SCHEMA_FOR_BINARY_LOGICAL_OP(Xor, "xor");

OPERATOR_SCHEMA(Not)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(Performs element-wise negation.)DOC")
    .Input(0, "X", "Input tensor of type `bool`.")
    .Output(0, "Y", "Output tensor of type `bool`.");
SHOULD_NOT_DO_GRADIENT(Not);

} // namespace caffe2
