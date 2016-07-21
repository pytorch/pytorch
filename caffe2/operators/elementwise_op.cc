#include "caffe2/operators/elementwise_op.h"
#include "caffe2/core/operator_gradient.h"

namespace caffe2 {

template <>
bool DivGradientOp<float, CPUContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& Z = Input(1);
  auto& dZ = Input(2);
  auto* dX = Output(0);
  auto* dY = Output(1);
  DCHECK_GT(Y.size(), 0);
  DCHECK_GT(Z.size(), 0);
  dX->ResizeLike(Y);
  dY->ResizeLike(Y);

  const float* Ydata = Y.data<float>();
  const float* Zdata = Z.data<float>();
  const float* dZdata = dZ.data<float>();
  float* dXdata = dX->mutable_data<float>();
  float* dYdata = dY->mutable_data<float>();
  #pragma omp parallel for
  for (int i = 0; i < Y.size(); ++i) {
    dXdata[i] = dZdata[i] / Ydata[i];
    dYdata[i] = - (dZdata[i] * Zdata[i]) / Ydata[i];
  }
  return true;
}

namespace {

REGISTER_CPU_OPERATOR(Add, AddOp<CPUContext>);
REGISTER_CPU_OPERATOR(Sub, SubOp<CPUContext>);
REGISTER_CPU_OPERATOR(Mul, MulOp<CPUContext>);
REGISTER_CPU_OPERATOR(Div, DivOp<CPUContext>);
REGISTER_CPU_OPERATOR(DivGradient, DivGradientOp<float, CPUContext>);

const char* kBroadcastDoc = R"DOC(
If necessary the right-hand-side argument will be broadcasted to match the
shape of left-hand-side argument. Only suffix matching is supported for now,
1-dim expansion doesn't work yet.

More precisely tensors A and B" can be operated on iff
`shape(A)[-len(shape(B)):] == shape(B)`

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
    schema.Input(
        0,
        "A",
        "First operand, should share the type with the second operand.");
    schema.Input(
        1,
        "B",
        "Second operand. With broadcasting can be of smaller size than A, "
        "without a few first dimensions more specifically. If broadcasting is "
        "disabled should be of exactly the same size as A");
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
OPERATOR_SCHEMA(DivGradient)
    .NumInputs(3)
    .NumOutputs(2)
    .AllowInplace({{0, 0}});

class GetAddGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    // TODO(jiayq): write gradient for the broadcast case.
    CAFFE_ENFORCE(!HasArgument(Def(), "broadcast"),
                  "Gradient not ready yet for Add with broadcasting.");
    SetDense(0, GO(0));
    SetDense(1, GO(0));
    return vector<OperatorDef>();
  }
};
REGISTER_GRADIENT(Add, GetAddGradient);

// TODO(jiayq): Although we have Sub gradient implemented, we are still missing
// the Negative unary operator to be implemented.
class GetSubGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    // TODO(jiayq): write gradient for the broadcast case.
    CAFFE_ENFORCE(!HasArgument(Def(), "broadcast"),
                  "Gradient not ready yet for Sub with broadcasting.");
    SetDense(0, GO(0));
    return SingleGradientDef(
        "Negative", "", vector<string>{GO(0)}, vector<string>{GI(1)});
  }
};
REGISTER_GRADIENT(Sub, GetSubGradient);

class GetMulGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    // TODO(jiayq): write gradient for the broadcast case.
    CAFFE_ENFORCE(!HasArgument(Def(), "broadcast"),
                  "Gradient not ready yet for Mul with broadcasting.");
    CAFFE_ENFORCE(
        Def().input(0) != Def().output(0) && Def().input(1) != Def().output(0),
        "Gradient computation cannot be carried out if Mul uses in-place "
        "computation: ", ProtoDebugString(Def()));
    return vector<OperatorDef>{
        CreateOperatorDef(
            "Mul", "",
            vector<string>{I(1), GO(0)},
            vector<string>{GI(0)}),
        CreateOperatorDef(
            "Mul", "",
            vector<string>{I(0), GO(0)},
            vector<string>{GI(1)}),
    };
  }
};
REGISTER_GRADIENT(Mul, GetMulGradient);

class GetDivGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    // TODO(jiayq): write gradient for the broadcast case.
    CAFFE_ENFORCE(!HasArgument(Def(), "broadcast"),
                  "Gradient not ready yet for Div with broadcasting.");
    return SingleGradientDef(
            "DivGradient", "",
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
    schema.Input(
        0,
        "A",
        "First operand, should share the type with the second operand.");
    schema.Input(
        1,
        "B",
        "Second operand. With broadcasting can be of smaller size than A, "
        "without a few first dimensions more specifically. If broadcasting is "
        "disabled should be of exactly the same size as A");
    schema.Output(0, "C", "Result, has same dimensions and A and type `bool`");
  };
}

#define CAFFE2_REGISTER_BINARY_COMPARISON_OP(name, symbol)    \
  REGISTER_CPU_OPERATOR(name, name##Op<CPUContext>);          \
  OPERATOR_SCHEMA(name).NumInputs(2).NumOutputs(1).FillUsing( \
      ComparisonDocGenerator(symbol));                        \
  SHOULD_NOT_DO_GRADIENT(name)

CAFFE2_REGISTER_BINARY_COMPARISON_OP(LT, "<");
CAFFE2_REGISTER_BINARY_COMPARISON_OP(LE, "<=");
CAFFE2_REGISTER_BINARY_COMPARISON_OP(GT, ">");
CAFFE2_REGISTER_BINARY_COMPARISON_OP(GE, ">=");

#undef REGISTER_BINIARY_COMPARISON_OP

}  // namespace
}  // namespace caffe2
