#include "caffe2/operators/conv_op.h"
#include "caffe2/operators/conv_op_impl.h"

namespace caffe2 {
namespace {
REGISTER_CPU_OPERATOR(Conv, ConvOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(ConvGradient, ConvGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(Conv).NumInputs(3).NumOutputs(1);
OPERATOR_SCHEMA(ConvGradient).NumInputs(3).NumOutputs(2, 3);

class GetConvGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_CHECK_EQ(def_.input_size(), 3);
    return SingleGradientDef(
        "ConvGradient", "",
        vector<string>{I(0), I(1), GO(0)},
        vector<string>{GI(1), GI(2), GI(0)});
  }
};
REGISTER_GRADIENT(Conv, GetConvGradient);

}  // namespace
}  // namespace caffe2
