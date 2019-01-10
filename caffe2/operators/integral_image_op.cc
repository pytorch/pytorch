#include "integral_image_op.h"
namespace caffe2 {

REGISTER_CPU_OPERATOR(IntegralImage, IntegralImageOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    IntegralImageGradient,
    IntegralImageGradientOp<float, CPUContext>);

// Input: X; Output: Y
OPERATOR_SCHEMA(IntegralImage)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Computes an integral image, which contains the sum of pixel values within
an image vertically and horizontally. This integral image can then be used
with other detection and tracking techniques.
)DOC")
    .Input(0, "X", "Images tensor of the form (N, C, H, W)")
    .Output(0, "Y", "Integrated image of the form (N, C, H+1, W+1)");

// Input: X, dY (aka "gradOutput"); Output: dX (aka "gradInput")
OPERATOR_SCHEMA(IntegralImageGradient).NumInputs(2).NumOutputs(1);

class GetIntegralImageGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "IntegralImageGradient",
        "",
        vector<string>{I(0), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(IntegralImage, GetIntegralImageGradient);

} // namespace caffe2
