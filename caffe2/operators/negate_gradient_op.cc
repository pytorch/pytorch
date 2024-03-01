#include "caffe2/operators/negate_gradient_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(NegateGradient, NegateGradientOp<CPUContext>);
OPERATOR_SCHEMA(NegateGradient)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
NegateGradient operator in forward pass simply copies input to the
output, and in backward pass, flips the sign of the output gradient
)DOC");

struct GetNegateGradientGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE_EQ(def_.input_size(), 1);
    return SingleGradientDef(
        "Negative", "", vector<string>{GO(0)}, vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(NegateGradient, GetNegateGradientGradient);

} // namespace caffe2
