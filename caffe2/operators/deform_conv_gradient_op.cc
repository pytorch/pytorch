#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/operators/deform_conv_op.h"
#include "caffe2/operators/deform_conv_op_impl.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(DeformConvGradient).NumInputs(4, 4).NumOutputs(2, 4);

namespace {

class GetDeformConvGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE(def_.input_size() == 3 || def_.input_size() == 4);

    ArgumentHelper argsHelper(def_);

    auto compute_dX =
        // NOLINTNEXTLINE(modernize-use-bool-literals)
        !argsHelper.GetSingleArgument<bool>("no_gradient_to_input", 0);

    if (def_.input_size() == 4) {
      if (compute_dX) {
        return SingleGradientDef(
            "DeformConvGradient",
            "",
            vector<string>{I(0), I(1), I(2), GO(0)},
            vector<string>{GI(1), GI(2), GI(3), GI(0)});
      } else {
        return SingleGradientDef(
            "DeformConvGradient",
            "",
            vector<string>{I(0), I(1), I(2), GO(0)},
            vector<string>{GI(1), GI(2), GI(3)});
      }
    } else {
      if (compute_dX) {
        return SingleGradientDef(
            "DeformConvGradient",
            "",
            vector<string>{I(0), I(1), I(2), GO(0)},
            vector<string>{GI(1), GI(2), GI(0)},
            vector<Argument>{MakeArgument<int>("no_bias", 1)});
      } else {
        return SingleGradientDef(
            "DeformConvGradient",
            "",
            vector<string>{I(0), I(1), I(2), GO(0)},
            vector<string>{GI(1), GI(2)},
            vector<Argument>{MakeArgument<int>("no_bias", 1)});
      }
    }
  }
};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(DeformConv, GetDeformConvGradient);

} // namespace
} // namespace caffe2
