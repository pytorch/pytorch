#include "caffe2/operators/conv_transpose_op.h"
#include "caffe2/operators/conv_transpose_op_impl.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    ConvTransposeGradient,
    ConvTransposeGradientOp<float, CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(ConvTransposeGradient).NumInputs(3).NumOutputs(1, 3);

class GetConvTransposeGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    auto compute_dX =
        !ArgumentHelper::GetSingleArgument(def_, "no_gradient_to_input", false);

    CAFFE_ENFORCE(3 == def_.input_size() || 2 == def_.input_size());
    if (def_.input_size() == 3 && compute_dX) {
      return SingleGradientDef(
          "ConvTransposeGradient",
          "",
          vector<string>{I(0), I(1), GO(0)},
          vector<string>{GI(1), GI(2), GI(0)});
    } else if (def_.input_size() == 3) {
      return SingleGradientDef(
          "ConvTransposeGradient",
          "",
          vector<string>{I(0), I(1), GO(0)},
          vector<string>{GI(1), GI(2)});
    } else if (compute_dX) {
      return SingleGradientDef(
          "ConvTransposeGradient",
          "",
          vector<string>{I(0), I(1), GO(0)},
          vector<string>{GI(1), GI(0)},
          vector<Argument>{MakeArgument<bool>("no_bias", true)});
    } else {
      return SingleGradientDef(
          "ConvTransposeGradient",
          "",
          vector<string>{I(0), I(1), GO(0)},
          vector<string>{GI(1)},
          vector<Argument>{MakeArgument<bool>("no_bias", true)});
    }
  }
};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(ConvTranspose, GetConvTransposeGradient);

} // namespace caffe2
