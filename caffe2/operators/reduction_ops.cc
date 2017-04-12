#include "caffe2/operators/reduction_ops.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(SumElements, SumElementsOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(SumSqrElements, SumSqrElementsOp<float, CPUContext>);

REGISTER_CPU_OPERATOR(
    SumElementsGradient,
    SumElementsGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(SumElements)
    .NumInputs(1)
    .NumOutputs(1)
    .ScalarType(TensorProto::FLOAT)
    .SetDoc("Sums the elements of the input tensor.")
    .Arg("average", "whether to average or not")
    .Input(0, "X", "Tensor to sum up")
    .Output(0, "sum", "Scalar sum");

OPERATOR_SCHEMA(SumSqrElements)
    .NumInputs(1)
    .NumOutputs(1)
    .ScalarType(TensorProto::FLOAT)
    .SetDoc("Sums the squares elements of the input tensor.")
    .Arg("average", "whether to average or not")
    .Input(0, "X", "Tensor to sum up")
    .Output(0, "sum", "Scalar sum of squares");

OPERATOR_SCHEMA(SumElementsGradient).NumInputs(2).NumOutputs(1);

class GetSumElementsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SumElementsGradient",
        "",
        vector<string>{I(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(SumElements, GetSumElementsGradient);

} // namespace

template <typename T, class Context>
bool SumElementsGradientOp<T, Context>::RunOnDevice() {
  auto& X = Input(0);
  TensorCPU sum_grad = TensorCPU(Input(1));
  auto* dX = Output(0);
  dX->ResizeLike(X);
  DCHECK_EQ(sum_grad.size(), 1);
  math::Set<T, Context>(
      dX->size(),
      static_cast<T>(sum_grad.data<T>()[0] * (average_ ? 1.0 / X.size() : 1)),
      dX->template mutable_data<T>(),
      &context_);
  return true;
}

} // namespace caffe2
