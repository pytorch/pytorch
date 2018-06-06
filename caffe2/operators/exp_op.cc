#include "caffe2/operators/exp_op.h"

#include <string>
#include <vector>

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    Exp,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, ExpFunctor<CPUContext>>);

OPERATOR_SCHEMA(Exp)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Calculates the exponential of the given input tensor, element-wise. This
operation can be done in an in-place fashion too, by providing the same input
and output blobs.
)DOC")
    .Input(0, "input", "Input tensor")
    .Output(
        0,
        "output",
        "The exponential of the input tensor computed "
        "element-wise")
    .InheritOnnxSchema("Exp");

namespace {

class GetExpGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "Mul",
        "",
        std::vector<std::string>{O(0), GO(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

REGISTER_GRADIENT(Exp, GetExpGradient);

} // namespace caffe2
