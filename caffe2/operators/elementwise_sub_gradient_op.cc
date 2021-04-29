#include "caffe2/operators/elementwise_sub_op.h"

#include <string>
#include <vector>

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    SubGradient,
    BinaryElementwiseGradientOp<
        NumericTypes,
        CPUContext,
        SubFunctor<CPUContext>>);

namespace {

class GetSubGradient final : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;

  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SubGradient",
        "",
        std::vector<std::string>{GO(0), I(0), I(1)},
        std::vector<std::string>{GI(0), GI(1)});
  }
};

} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(Sub, GetSubGradient);

} // namespace caffe2
