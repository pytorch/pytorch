#include "caffe2/operators/scale_gradient.h"

namespace caffe2 {
REGISTER_CPU_OPERATOR(ScaleGradient, ScaleGradientOp<CPUContext>);

OPERATOR_SCHEMA(ScaleGradient)
    .NumInputs(1, 1)
    .NumOutputs(1, 1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
ScaleGradient is a helper operator that does no actual numerical computation,
and in the gradient computation phase scales the gradient from being computed
through it. For example:

Forward
  (input = X) ==> (output = X)

Backward
  (otput_grad = G) ==> (input_grad = scale * G)

Note: this operator may break the mathematica gradient back-propogation and
break the net gradient checking. You need to use it carefully. One of the use
cases: during multi-task learning, we may use this operator to control how
much a task will affect the shared layers.
)DOC");

class GetScaleGradientGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE(
        GradOut(0).IsDense(), "Input gradient ", O(0), " should be dense.");

    ArgumentHelper argsHelper(def_);
    const float scale = argsHelper.GetSingleArgument<float>("scale", 1.0f);
    const string scale_str = string("scale=") + to_string(scale);

    return SingleGradientDef(
        "Scale", scale_str, vector<string>{GO(0)}, vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(ScaleGradient, GetScaleGradientGradient);
} // namespace caffe2
