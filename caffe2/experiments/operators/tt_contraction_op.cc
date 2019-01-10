#include "caffe2/experiments/operators/tt_contraction_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(TTContraction, TTContractionOp<float, CPUContext>);

OPERATOR_SCHEMA(TTContraction)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Tensor contraction C = A * B
)DOC")
    .Arg("K", "i_{k-1} * r_k")
    .Arg("M", "r_{k-1} * o_{k-1}")
    .Arg("N", "o_k")
    .Input(0, "A", "2D matrix of size (K x M)")
    .Input(1, "B", "tensor")
    .Output(0, "C", "contracted tensor");

REGISTER_CPU_OPERATOR(
    TTContractionGradient,
    TTContractionGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(TTContractionGradient).NumInputs(3).NumOutputs(2);

class GetTTContractionGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "TTContractionGradient",
        "",
        vector<string>{GO(0), I(0), I(1)},
        vector<string>{GI(0), GI(1)},
        Def().arg());
  }
};

REGISTER_GRADIENT(TTContraction, GetTTContractionGradient);

} // namespace caffe2
