#include "caffe2/operators/sqrt_op.h"

#include <string>
#include <vector>

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    Sqrt,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        SqrtFunctor<CPUContext>>);

// Input: X, output: Y
OPERATOR_SCHEMA(Sqrt)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Computes the element-wise sqrt of the input.
)DOC")
    .Input(0, "X", "ND input tensor")
    .Output(0, "Y", "ND input tensor");

namespace {

class GetSqrtGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    Argument scale_arg;
    scale_arg.set_name("scale");
    scale_arg.set_f(0.5);
    return std::vector<OperatorDef>{CreateOperatorDef(
                                        "Scale",
                                        "",
                                        std::vector<std::string>{GO(0)},
                                        std::vector<std::string>{GI(0)},
                                        std::vector<Argument>{scale_arg}),
                                    CreateOperatorDef(
                                        "Div",
                                        "",
                                        std::vector<std::string>{GI(0), O(0)},
                                        std::vector<std::string>{GI(0)})};
  }
};

} // namespace

REGISTER_GRADIENT(Sqrt, GetSqrtGradient);

} // namespace caffe2
