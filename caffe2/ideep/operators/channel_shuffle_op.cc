#include <caffe2/ideep/operators/conv_pool_base_op.h>

using namespace caffe2;

namespace {

class ChannelShuffleOp final : public IDEEPConvPoolOpBase {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_CONV_POOL_BASE_FUNCTIONS();

  ChannelShuffleOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPConvPoolOpBase(operator_def, ws) {}

  bool RunOnDeviceWithOrderNCHW() override {
    const auto& X = Input(INPUT);
    auto* Y = Output(OUTPUT);

    ideep::channel_shuffle_forward::compute(X, *Y, group_);

    return true;
  }

 private:
  INPUT_TAGS(INPUT);
  OUTPUT_TAGS(OUTPUT);
};

class ChannelShuffleGradientOp final : public IDEEPConvPoolOpBase {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_CONV_POOL_BASE_FUNCTIONS();

  ChannelShuffleGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPConvPoolOpBase(operator_def, ws) {}

  bool RunOnDeviceWithOrderNCHW() override {
    const auto& dY = Input(OUTPUT_GRAD);
    auto* dX = Output(INPUT_GRAD);

    ideep::channel_shuffle_backward::compute(dY, *dX, group_);

    return true;
  }

 private:
  INPUT_TAGS(OUTPUT_GRAD);
  OUTPUT_TAGS(INPUT_GRAD);
};


// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_IDEEP_OPERATOR(ChannelShuffle, ChannelShuffleOp);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_IDEEP_OPERATOR(ChannelShuffleGradient, ChannelShuffleGradientOp);

} // namespace
