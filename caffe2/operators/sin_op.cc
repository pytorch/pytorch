#include "caffe2/operators/sin_op.h"

#include <algorithm>
#include <functional>

namespace caffe2 {

template <>
template <typename T>
bool SinGradientFunctor<CPUContext>::Forward(
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
  EigenVectorMap<T>(dX, size) = dY_arr * X_arr.cos();
  return true;
}

REGISTER_CPU_OPERATOR(
    Sin,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, SinFunctor<CPUContext>>);
REGISTER_CPU_OPERATOR(
    SinGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        SinGradientFunctor<CPUContext>>);

OPERATOR_SCHEMA(Sin)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Calculates the sine of the given input tensor, element-wise.
)DOC")
    .Input(0, "input", "Input tensor")
    .Output(0, "output", "The sine of the input tensor computed element-wise");

OPERATOR_SCHEMA(SinGradient).NumInputs(2).NumOutputs(1).IdenticalTypeAndShape();

namespace {

class GetSinGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SinGradient",
        "",
        std::vector<std::string>{GO(0), I(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

REGISTER_GRADIENT(Sin, GetSinGradient);

} // namespace caffe2
