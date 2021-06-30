#include "caffe2/operators/zero_gradient_op.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(ZeroGradient, ZeroGradientOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(ZeroGradient, GetZeroGradientOpGradient);

} // namespace caffe2
