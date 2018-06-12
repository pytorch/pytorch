#include "caffe2/operators/atan_op.h"

#include <algorithm>
#include <functional>

namespace caffe2 {

template <>
template <typename T>
bool AtanGradientFunctor<CPUContext>::Forward(
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
  EigenVectorMap<T>(dX, size) = dY_arr / (T(1) + X_arr.square());
  return true;
}

REGISTER_CPU_OPERATOR(
    Atan,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        AtanFunctor<CPUContext>>);
REGISTER_CPU_OPERATOR(
    AtanGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        AtanGradientFunctor<CPUContext>>);

OPERATOR_SCHEMA(Atan)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Calculates the arctangent of the given input tensor, element-wise.
)DOC")
    .Input(0, "input", "Input tensor")
    .Output(
        0,
        "output",
        "The arctangent of the input tensor computed element-wise");

OPERATOR_SCHEMA(AtanGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .IdenticalTypeAndShape();

namespace {

class GetAtanGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "AtanGradient",
        "",
        std::vector<std::string>{GO(0), I(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

REGISTER_GRADIENT(Atan, GetAtanGradient);

} // namespace caffe2
