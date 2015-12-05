#include "caffe2/operators/fully_connected_op.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(FC, FullyConnectedOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(FCGradient, FullyConnectedGradientOp<float, CPUContext>);

struct GetFCGradient : public GetGradientDefBase {
  vector<OperatorDef>* Create(const OperatorDef& def) override {
    CAFFE_CHECK_EQ(def.input_size(), 3);
    return SingleGradientDef(
        "FCGradient", "",
        vector<string>{I(def, 0), I(def, 1), GO(def, 0)},
        vector<string>{GI(def, 1), GI(def, 2), GI(def, 0)});
  }
};
REGISTER_GRADIENT(FC, GetFCGradient);
}  // namespace
}  // namespace caffe2
