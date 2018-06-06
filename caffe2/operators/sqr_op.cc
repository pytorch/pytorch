#include "caffe2/operators/sqr_op.h"

#include <string>
#include <vector>

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    Sqr,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, SqrFunctor<CPUContext>>);

OPERATOR_SCHEMA(Sqr)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc("Square (x^2) the elements of the input")
    .Input(0, "input", "Input tensor")
    .Output(0, "output", "Squared elements of the input");

namespace {

class GetSqrGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    Argument scale_arg;
    scale_arg.set_name("scale");
    scale_arg.set_f(2.0);
    return std::vector<OperatorDef>{CreateOperatorDef(
                                        "Scale",
                                        "",
                                        std::vector<std::string>{GO(0)},
                                        std::vector<std::string>{GO(0)},
                                        std::vector<Argument>{scale_arg}),
                                    CreateOperatorDef(
                                        "Mul",
                                        "",
                                        std::vector<std::string>{GO(0), I(0)},
                                        std::vector<std::string>{GI(0)})};
  }
};

} // namespace

REGISTER_GRADIENT(Sqr, GetSqrGradient);

} // namespace caffe2
