#include "caffe2/operators/conv_op.h"
#include "caffe2/operators/conv_op_impl.h"
#include "caffe2/operators/conv_pool_op_base.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(ConvGradient, ConvGradientOp<float, CPUContext>);
OPERATOR_SCHEMA(ConvGradient).NumInputs(2, 3).NumOutputs(1, 3);
class GetConvGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE(def_.input_size() == 3 || def_.input_size() == 2);

    ArgumentHelper argsHelper(def_);

    auto compute_dX = !argsHelper.GetSingleArgument<bool>("no_gradient_to_input", 0);

    if (def_.input_size() == 3) {
      if (compute_dX) {
        return SingleGradientDef(
            "ConvGradient",
            "",
            vector<string>{I(0), I(1), GO(0)},
            vector<string>{GI(1), GI(2), GI(0)});
      } else {
        return SingleGradientDef(
            "ConvGradient",
            "",
            vector<string>{I(0), I(1), GO(0)},
            vector<string>{GI(1), GI(2)});
      }
    } else {
      if (compute_dX) {
        return SingleGradientDef(
            "ConvGradient",
            "",
            vector<string>{I(0), I(1), GO(0)},
            vector<string>{GI(1), GI(0)},
            vector<Argument>{MakeArgument<int>("no_bias", 1)});
      } else {
        return SingleGradientDef(
            "ConvGradient",
            "",
            vector<string>{I(0), I(1), GO(0)},
            vector<string>{GI(1)},
            vector<Argument>{MakeArgument<int>("no_bias", 1)});
      }
    }
  }
};
REGISTER_GRADIENT(Conv, GetConvGradient);

} // namespace caffe2
