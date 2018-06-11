#include "caffe2/operators/abs_op.h"

#include <algorithm>
#include <functional>

namespace caffe2 {

template <>
template <typename T>
bool AbsGradientFunctor<CPUContext>::Forward(
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
  EigenVectorMap<T>(dX, size) =
      (X_arr == T(0)).select(T(0), (X_arr > T(0)).select(dY_arr, -dY_arr));
  return true;
}

REGISTER_CPU_OPERATOR(
    Abs,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, AbsFunctor<CPUContext>>);
REGISTER_CPU_OPERATOR(
    AbsGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        AbsGradientFunctor<CPUContext>>);

OPERATOR_SCHEMA(Abs)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Calculates the absolute value of the given input tensor, element-wise.
)DOC")
    .Input(0, "input", "Input tensor")
    .Output(
        0,
        "output",
        "The absolute value of the input tensor computed element-wise")
    .InheritOnnxSchema("Abs");

OPERATOR_SCHEMA(AbsGradient).NumInputs(2).NumOutputs(1).IdenticalTypeAndShape();

namespace {

class GetAbsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "AbsGradient",
        "",
        std::vector<std::string>{GO(0), I(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

REGISTER_GRADIENT(Abs, GetAbsGradient);

} // namespace caffe2
