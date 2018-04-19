#include "caffe2/operators/reduce_ops.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(ReduceSum, ReduceSumOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    ReduceSumGradient,
    ReduceSumGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(ReduceSum)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
  Computes the sum of the input tensor's element along the provided axes.
  The resulted tensor has the same rank as the input if keepdims equal 1.
  If keepdims equal 0, then the resulted tensor have the reduced dimension pruned.
)DOC")
    .Arg("axes", "A list of integers, along which to reduce.")
    .Arg(
        "keepdims",
        "Keep the reduced dimension(s) or not, default 1 keeps the reduced dimension(s).")
    .Input(0, "data", "An input tensor.")
    .Output(0, "reduced", "Reduced output tensor.");


REGISTER_CPU_OPERATOR(ReduceMean, ReduceMeanOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    ReduceMeanGradient,
    ReduceMeanGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(ReduceMean)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
      Computes the mean of the input tensor's element along the provided axes.
      The resulted tensor has the same rank as the input if keepdims equal 1.
      If keepdims equal 0, then the resulted tensor have the reduced dimension pruned.
    )DOC")
    .Arg("axes", "A list of integers, along which to reduce.")
    .Arg(
        "keepdims",
        "Keep the reduced dimension(s) or not, default 1 keeps the reduced dimension(s).")
    .Input(0, "data", "An input tensor.")
    .Output(0, "reduced", "Reduced output tensor.");

class GetReduceGradient final : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;

  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        def_.type() + "Gradient",
        "",
        std::vector<string>{GO(0), I(0)},
        std::vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(ReduceSum, GetReduceGradient);
REGISTER_GRADIENT(ReduceMean, GetReduceGradient);

} // namespace caffe2
