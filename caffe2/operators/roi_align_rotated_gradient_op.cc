#include "roi_align_rotated_gradient_op.h"

#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

// Input: X, rois, dY (aka "gradOutput");
// Output: dX (aka "gradInput")
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(RoIAlignRotatedGradient)
    .NumInputs(3)
    .NumOutputs(1)
    .Input(0, "X", "See RoIAlignRotated.")
    .Input(1, "RoIs", "See RoIAlignRotated.")
    .Input(2, "dY", "Gradient of forward output 0 (Y)")
    .Output(0, "dX", "Gradient of forward input 0 (X)");

namespace {

class GetRoIAlignRotatedGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "RoIAlignRotatedGradient",
        "",
        vector<string>{I(0), I(1), GO(0)},
        vector<string>{GI(0)});
  }
};

} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(RoIAlignRotated, GetRoIAlignRotatedGradient);

} // namespace caffe2
