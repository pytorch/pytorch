#include "channel_shuffle_op.h"

namespace caffe2 {

class GetChannelShuffleGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        def_.type() + "Gradient",
        "",
        vector<string>{GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_CPU_OPERATOR(ChannelShuffle, ChannelShuffleOp<CPUContext>);
REGISTER_CPU_OPERATOR(
    ChannelShuffleGradient,
    ChannelShuffleGradientOp<CPUContext>);
REGISTER_GRADIENT(ChannelShuffle, GetChannelShuffleGradient);
OPERATOR_SCHEMA(ChannelShuffle)
    .IdenticalTypeAndShape()
    .NumInputs(1)
    .NumOutputs(1)
    .InheritOnnxSchema("ChannelShuffle");
OPERATOR_SCHEMA(ChannelShuffleGradient)
    .IdenticalTypeAndShape()
    .NumInputs(1)
    .NumOutputs(1);
}
