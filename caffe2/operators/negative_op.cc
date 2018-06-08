#include "caffe2/operators/negative_op.h"

#include <string>
#include <vector>

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    Negative,
    UnaryElementwiseOp<NumericTypes, CPUContext, NegativeFunctor<CPUContext>>);

// Input: X, output: Y
OPERATOR_SCHEMA(Negative)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Computes the element-wise negative of the input.
)DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D input tensor")
    .InheritOnnxSchema("Neg");

namespace {

class GetNegativeGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "Negative",
        "",
        std::vector<std::string>{GO(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

REGISTER_GRADIENT(Negative, GetNegativeGradient);

} // namespace caffe2
