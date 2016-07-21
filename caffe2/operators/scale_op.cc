#include "caffe2/operators/scale_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Scale, ScaleOp<float, CPUContext>);
OPERATOR_SCHEMA(Scale)
  .NumInputs(1)
  .NumOutputs(1)
  .AllowInplace({{0, 0}})
  .SetDoc(R"DOC(
Scale takes one input data (Tensor<float>) and produces one output data
(Tensor<float>) whose value is the input data tensor scaled element-wise.
)DOC")
  .Arg("scale", "(float, default 1.0) the scale to apply.");

class GetScaleGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "Scale", "",
        vector<string>{GO(0)},
        vector<string>{GI(0)},
        Def().arg());
  }
};
REGISTER_GRADIENT(Scale, GetScaleGradient);
}  // namespace caffe2
