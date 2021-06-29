#include "caffe2/operators/scale_op.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Scale, ScaleOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Scale)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Scale takes one input data (Tensor) and produces one output data
(Tensor) whose value is the input data tensor scaled element-wise.
)DOC")
    .Arg("scale", "(float, default 1.0) the scale to apply.");

class GetScaleGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    // CopyArguments is true by default so the "scale" arg is going to be copied
    return SingleGradientDef(
        "Scale", "", vector<string>{GO(0)}, vector<string>{GI(0)});
  }
};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(Scale, GetScaleGradient);
}  // namespace caffe2
