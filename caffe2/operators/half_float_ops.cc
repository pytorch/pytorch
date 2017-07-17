#include "caffe2/operators/half_float_ops.h"

namespace caffe2 {
OPERATOR_SCHEMA(FloatToHalf).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(HalfToFloat).NumInputs(1).NumOutputs(1);

class GetFloatToHalfGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "HalfToFloat", "", vector<string>{GO(0)}, vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(FloatToHalf, GetFloatToHalfGradient);

class GetHalfToFloatGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "FloatToHalf", "", vector<string>{GO(0)}, vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(HalfToFloat, GetHalfToFloatGradient);
} // namespace caffe2
