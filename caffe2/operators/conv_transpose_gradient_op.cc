#include "caffe2/operators/conv_transpose_op.h"
#include "caffe2/operators/conv_transpose_op_impl.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    ConvTransposeGradient,
    ConvTransposeGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(ConvTransposeGradient).NumInputs(3).NumOutputs(2, 3);

class GetConvTransposeGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE(3 == def_.input_size());
    return SingleGradientDef(
        "ConvTransposeGradient",
        "",
        vector<string>{I(0), I(1), GO(0)},
        vector<string>{GI(1), GI(2), GI(0)});
  }
};
REGISTER_GRADIENT(ConvTranspose, GetConvTransposeGradient);

} // namespace caffe2
