#include "caffe2/operators/rsqrt_op.h"

#include <algorithm>
#include <functional>
#include <string>

namespace caffe2 {

template <>
struct RSqrtFunctor<CPUContext> {
  template <typename T>
  bool operator()(const int size, const T* X, T* Y, CPUContext* /* context */)
      const {
    EigenArrayMap<T>(Y, 1, size) = ConstEigenArrayMap<T>(X, 1, size).rsqrt();
    return true;
  }
};

template <>
template <typename T>
bool RSqrtGradientFunctor<CPUContext>::Forward(
    const std::vector<int>& dY_dims,
    const std::vector<int>& /* Y_dims */,
    const T* dY,
    const T* Y,
    T* dX,
    CPUContext* /* context */) const {
  const int size = std::accumulate(
      dY_dims.cbegin(), dY_dims.cend(), 1, std::multiplies<int>());
  EigenVectorMap<T>(dX, size) = ConstEigenVectorMap<T>(dY, size).array() *
      ConstEigenVectorMap<T>(Y, size).array().cube() * static_cast<T>(-0.5);
  return true;
}

REGISTER_CPU_OPERATOR(
    RSqrt,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        RSqrtFunctor<CPUContext>>);
REGISTER_CPU_OPERATOR(
    RSqrtGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        RSqrtGradientFunctor<CPUContext>>);

OPERATOR_SCHEMA(RSqrt)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc("Computes the element-wise rsqrt of the input.")
    .Input(0, "X", "ND input tensor")
    .Output(0, "Y", "ND output tensor");

OPERATOR_SCHEMA(RSqrtGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}});

namespace {

class GetRSqrtGradient final : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;

  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "RSqrtGradient",
        "",
        std::vector<std::string>{GO(0), O(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

REGISTER_GRADIENT(RSqrt, GetRSqrtGradient);

} // namespace caffe2
