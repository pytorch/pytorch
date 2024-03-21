#include "caffe2/operators/cbrt_op.h"
#include "caffe2/utils/eigen_utils.h"

#include <algorithm>
#include <functional>
#include <string>

namespace caffe2 {

template <>
template <typename T>
bool CbrtGradientFunctor<CPUContext>::Forward(
    const std::vector<int>& dY_dims,
    const std::vector<int>& /* Y_dims */,
    const T* dY,
    const T* Y,
    T* dX,
    CPUContext* /* context */) const {
  const int size = std::accumulate(
      // NOLINTNEXTLINE(modernize-use-transparent-functors)
      dY_dims.cbegin(), dY_dims.cend(), 1, std::multiplies<int>());
  EigenVectorMap<T>(dX, size) = ConstEigenVectorArrayMap<T>(dY, size) /
      ConstEigenVectorArrayMap<T>(Y, size).square() / T(3);
  return true;
}

REGISTER_CPU_OPERATOR(
    Cbrt,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        CbrtFunctor<CPUContext>>);
REGISTER_CPU_OPERATOR(
    CbrtGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        CbrtGradientFunctor<CPUContext>>);

OPERATOR_SCHEMA(Cbrt)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .Input(0, "X", "*(type: Tensor`<float>`)* Input tensor.")
    .Output(
        0,
        "Y",
        "*(type: Tensor`<float>`)* Output tensor calculated as the cbrt of the input tensor, element-wise.");

OPERATOR_SCHEMA(CbrtGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShapeOfInput(0);

namespace {

class GetCbrtGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "CbrtGradient",
        "",
        std::vector<std::string>{GO(0), O(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

REGISTER_GRADIENT(Cbrt, GetCbrtGradient);

} // namespace caffe2
