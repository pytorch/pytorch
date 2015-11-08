#include "caffe2/operators/fully_connected_op.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(FC, FullyConnectedOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(FCGradient, FullyConnectedGradientOp<float, CPUContext>);

struct GetFCGradient : public GetGradientDefBase {
  static vector<OperatorDef>* Create(const OperatorDef& def) {
    CAFFE_CHECK_EQ(def.input_size(), 3);
    return new vector<OperatorDef>{
        CreateOperatorDef(
            "FCGradient", "",
            std::vector<string>{def.input(0), def.input(1),
                                GradientName(def.output(0))},
            std::vector<string>{GradientName(def.input(1)),
                                GradientName(def.input(2)),
                                GradientName(def.input(0))})};
  }
};
REGISTER_GRADIENT(FC, GetFCGradient);
}  // namespace
}  // namespace caffe2
