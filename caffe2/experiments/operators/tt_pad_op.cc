#include "caffe2/experiments/operators/tt_pad_op.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(TTPad, TTPadOp<float, CPUContext>);
OPERATOR_SCHEMA(TTPad).NumInputs(1).NumOutputs(2).EnforceInplace({{0, 0}});

REGISTER_CPU_OPERATOR(TTPadGradient, TTPadGradientOp<float, CPUContext>);
OPERATOR_SCHEMA(TTPadGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .EnforceInplace({{0, 0}});

class GetTTPadGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "TTPadGradient",
        "",
        vector<string>{GO(0), O(1)},
        vector<string>{GI(0)},
        Def().arg());
  }
};

REGISTER_GRADIENT(TTPad, GetTTPadGradient);

} // namespace
} // namespace caffe2
