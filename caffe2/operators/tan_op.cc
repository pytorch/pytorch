#include "caffe2/operators/tan_op.h"

#include "caffe2/utils/eigen_utils.h"

#include <algorithm>
#include <functional>

namespace caffe2 {

template <>
template <typename T>
bool TanGradientFunctor<CPUContext>::Forward(
    const std::vector<int>& X_dims,
    const std::vector<int>& /* dY_dims */,
    const T* X,
    const T* dY,
    T* dX,
    CPUContext* /* context */) const {
  const int size = std::accumulate(
      // NOLINTNEXTLINE(modernize-use-transparent-functors)
      X_dims.cbegin(), X_dims.cend(), 1, std::multiplies<int>());
  ConstEigenVectorArrayMap<T> dY_arr(dY, size);
  ConstEigenVectorArrayMap<T> X_arr(X, size);
  EigenVectorMap<T>(dX, size) = dY_arr / X_arr.cos().square();
  return true;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    Tan,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, TanFunctor<CPUContext>>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    TanGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        TanGradientFunctor<CPUContext>>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(TanGradient).NumInputs(2).NumOutputs(1).IdenticalTypeAndShape();

namespace {

class GetTanGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "TanGradient",
        "",
        std::vector<std::string>{I(0), GO(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(Tan, GetTanGradient);

} // namespace caffe2
