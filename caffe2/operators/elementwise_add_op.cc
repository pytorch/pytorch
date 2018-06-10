#include "caffe2/operators/elementwise_add_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    Add,
    BinaryElementwiseOp<NumericTypes, CPUContext, AddFunctor<CPUContext>>);

#if !CAFFE2_MOBILE

REGISTER_CPU_OPERATOR(
    AddGradient,
    BinaryElementwiseGradientOp<
        NumericTypes,
        CPUContext,
        AddFunctor<CPUContext>>);

namespace {

class GetAddGradient final : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;

  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "AddGradient",
        "",
        std::vector<std::string>{GO(0), I(0), I(1)},
        std::vector<std::string>{GI(0), GI(1)});
  }
};

} // namespace

REGISTER_GRADIENT(Add, GetAddGradient);

#endif // !CAFFE2_MOBILE

} // namespace caffe2
