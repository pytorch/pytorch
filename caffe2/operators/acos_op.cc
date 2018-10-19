#include "caffe2/operators/acos_op.h"
#include "caffe2/utils/eigen_utils.h"

#include <algorithm>
#include <functional>

namespace caffe2 {

template <>
template <typename T>
bool AcosGradientFunctor<CPUContext>::Forward(
    const std::vector<int>& X_dims,
    const std::vector<int>& /* dY_dims */,
    const T* X,
    const T* dY,
    T* dX,
    CPUContext* /* context */) const {
  const int size = std::accumulate(
      X_dims.cbegin(), X_dims.cend(), 1, std::multiplies<int>());
  ConstEigenVectorArrayMap<T> dY_arr(dY, size);
  ConstEigenVectorArrayMap<T> X_arr(X, size);
  EigenVectorMap<T>(dX, size) = -dY_arr * (T(1) - X_arr.square()).rsqrt();
  return true;
}

REGISTER_CPU_OPERATOR(
    Acos,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        AcosFunctor<CPUContext>>);
REGISTER_CPU_OPERATOR(
    AcosGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        AcosGradientFunctor<CPUContext>>);

OPERATOR_SCHEMA(Acos)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Calculates the arccosine of the given input tensor, element-wise.
)DOC")
    .Input(0, "input", "Input tensor")
    .Output(
        0,
        "output",
        "The arccosine of the input tensor computed element-wise");

OPERATOR_SCHEMA(AcosGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .IdenticalTypeAndShape();

namespace {

class GetAcosGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "AcosGradient",
        "",
        std::vector<std::string>{I(0), GO(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

REGISTER_GRADIENT(Acos, GetAcosGradient);

} // namespace caffe2
