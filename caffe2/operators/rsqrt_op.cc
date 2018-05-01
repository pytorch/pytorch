#include "caffe2/operators/rsqrt_op.h"

#include <string>
#include <vector>

namespace caffe2 {

template <>
struct RSqrtFunctor<CPUContext> {
  template <typename T>
  inline void operator()(
      const int size,
      const T* X,
      T* Y,
      CPUContext* /* context */) const {
    EigenArrayMap<T>(Y, 1, size) = ConstEigenArrayMap<T>(X, 1, size).rsqrt();
  }
};

template <>
template <typename T>
void RSqrtGradientFunctor<CPUContext>::Run(
    const int size,
    const T* dY,
    const T* Y,
    T* dX,
    CPUContext* /* context */) const {
  EigenArrayMap<T>(dX, 1, size) = ConstEigenArrayMap<T>(dY, 1, size) *
      ConstEigenArrayMap<T>(Y, 1, size).cube() * static_cast<T>(-0.5);
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
        WithoutBroadcast<RSqrtGradientFunctor<CPUContext>>>);

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
