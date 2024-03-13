#include "caffe2/operators/zero_gradient_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(ZeroGradient, ZeroGradientOp<CPUContext>);
OPERATOR_SCHEMA(ZeroGradient)
    .NumInputs(1)
    .NumOutputs(0)
    .SetDoc(R"DOC(
ZeroGradient operators doesn't produce any output blobs. One can use
this operator to produce 0 gradient for the input blob.
)DOC");

struct GetZeroGradientOpGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "ConstantFill",
        "",
        vector<string>{I(0)},
        vector<string>{GI(0)},
        vector<Argument>{MakeArgument<float>("value", 0.0)});
  }
};

REGISTER_GRADIENT(ZeroGradient, GetZeroGradientOpGradient);

} // namespace caffe2
