#include "caffe2/operators/fully_connected_op.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(FC, FullyConnectedOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(FCGradient, FullyConnectedGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(FC).NumInputs(3).NumOutputs(1);
OPERATOR_SCHEMA(FCGradient).NumInputs(3).NumOutputs(2, 3);

class GetFCGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_CHECK_EQ(def_.input_size(), 3);
    return SingleGradientDef(
        "FCGradient", "",
        vector<string>{I(0), I(1), GO(0)},
        vector<string>{GI(1), GI(2), GI(0)});
  }
};
REGISTER_GRADIENT(FC, GetFCGradient);
}  // namespace
}  // namespace caffe2
