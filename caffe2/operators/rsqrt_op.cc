#include "caffe2/operators/rsqrt_op.h"

#include "caffe2/utils/eigen_utils.h"

#include <algorithm>
#include <functional>
#include <string>

namespace caffe2 {

template <>
template <typename T>
bool RsqrtGradientFunctor<CPUContext>::Forward(
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
    Rsqrt,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        RsqrtFunctor<CPUContext>>);
REGISTER_CPU_OPERATOR(
    RsqrtGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        RsqrtGradientFunctor<CPUContext>>);

OPERATOR_SCHEMA(Rsqrt)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc("Computes the element-wise rsqrt of the input.")
    .Input(0, "X", "ND input tensor")
    .Output(0, "Y", "ND output tensor");

OPERATOR_SCHEMA(RsqrtGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}});

namespace {

class GetRsqrtGradient final : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;

  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "RsqrtGradient",
        "",
        std::vector<std::string>{GO(0), O(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

REGISTER_GRADIENT(Rsqrt, GetRsqrtGradient);

} // namespace caffe2
