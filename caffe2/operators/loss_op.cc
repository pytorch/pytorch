#include "caffe2/operators/loss_op.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(AveragedLoss, AveragedLoss<float, CPUContext>);
REGISTER_CPU_OPERATOR(WeightedSumLoss, WeightedSumLoss<float, CPUContext>);
REGISTER_CPU_OPERATOR(AveragedLossGradient,
                      AveragedLossGradient<float, CPUContext>);
REGISTER_CPU_OPERATOR(WeightedSumLossGradient,
                      WeightedSumLossGradient<float, CPUContext>);

OPERATOR_SCHEMA(AveragedLoss).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(WeightedSumLoss).NumInputs(2).NumOutputs(1);
OPERATOR_SCHEMA(AveragedLossGradient).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(WeightedSumLossGradient).NumInputs(1).NumOutputs(1);

class GetAveragedLossGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "AveragedLossGradient", "",
        vector<string>{I(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(AveragedLoss, GetAveragedLossGradient);

class GetWeightedSumLossGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "WeightedSumLossGradient", "",
        vector<string>{I(1)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(WeightedSumLoss, GetWeightedSumLossGradient);

}  // namespace
}  // namespace caffe2
