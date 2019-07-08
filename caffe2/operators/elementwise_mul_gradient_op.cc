#include "caffe2/operators/elementwise_mul_op.h"

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    MulGradient,
    BinaryElementwiseGradientOp<
        NumericTypes,
        CPUContext,
        MulFunctor<CPUContext>>);

namespace {

class GetMulGradient final : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;

  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "MulGradient",
        "",
        std::vector<std::string>{GO(0), I(0), I(1)},
        std::vector<std::string>{GI(0), GI(1)});
  }
};

} // namespace

REGISTER_GRADIENT(Mul, GetMulGradient);

} // namespace caffe2
