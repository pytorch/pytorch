#include "caffe2/operators/tan_op.h"

#include <algorithm>
#include <functional>

namespace caffe2 {

template <>
template <typename T>
bool TanGradientFunctor<CPUContext>::Forward(
    const std::vector<int>& dY_dims,
    const std::vector<int>& /* X_dims */,
    const T* dY,
    const T* X,
    T* dX,
    CPUContext* /* context */) const {
  const int size = std::accumulate(
      dY_dims.cbegin(), dY_dims.cend(), 1, std::multiplies<int>());
  ConstEigenVectorArrayMap<T> dY_arr(dY, size);
  ConstEigenVectorArrayMap<T> X_arr(X, size);
  EigenVectorMap<T>(dX, size) = dY_arr / X_arr.cos().square();
  return true;
}

REGISTER_CPU_OPERATOR(
    Tan,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, TanFunctor<CPUContext>>);
REGISTER_CPU_OPERATOR(
    TanGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        TanGradientFunctor<CPUContext>>);

OPERATOR_SCHEMA(Tan)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Calculates the tangent of the given input tensor, element-wise.
)DOC")
    .Input(0, "input", "Input tensor")
    .Output(
        0,
        "output",
        "The tangent of the input tensor computed element-wise");

OPERATOR_SCHEMA(TanGradient).NumInputs(2).NumOutputs(1).IdenticalTypeAndShape();

namespace {

class GetTanGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "TanGradient",
        "",
        std::vector<std::string>{GO(0), I(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

REGISTER_GRADIENT(Tan, GetTanGradient);

} // namespace caffe2
