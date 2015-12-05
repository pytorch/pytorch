#include "caffe2/operators/loss_op.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(AveragedLoss, AveragedLoss<float, CPUContext>);
REGISTER_CPU_OPERATOR(WeightedSumLoss, WeightedSumLoss<float, CPUContext>);
REGISTER_CPU_OPERATOR(AveragedLossGradient,
                      AveragedLossGradient<float, CPUContext>);
REGISTER_CPU_OPERATOR(WeightedSumLossGradient,
                      WeightedSumLossGradient<float, CPUContext>);


struct GetAveragedLossGradient : public GetGradientDefBase {
  vector<OperatorDef>* Create(const OperatorDef& def) override {
    return SingleGradientDef(
        "AveragedLossGradient", "",
        vector<string>{I(def, 0)},
        vector<string>{GI(def, 0)});
  }
};
REGISTER_GRADIENT(AveragedLoss, GetAveragedLossGradient);

struct GetWeightedSumLossGradient : public GetGradientDefBase {
  vector<OperatorDef>* Create(const OperatorDef& def) override {
    return SingleGradientDef(
        "WeightedSumLossGradient", "",
        vector<string>{I(def, 1)},
        vector<string>{GI(def, 0)});
  }
};
REGISTER_GRADIENT(WeightedSumLoss, GetWeightedSumLossGradient);

}  // namespace
}  // namespace caffe2
