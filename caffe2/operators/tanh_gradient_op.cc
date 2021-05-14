#include "caffe2/operators/tanh_op.h"

#include "caffe2/utils/eigen_utils.h"

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

namespace caffe2 {

template <>
template <>
bool TanhGradientFunctor<CPUContext>::Forward<float>(
    const std::vector<int>& Y_dims,
    const std::vector<int>& /* dY_dims */,
    const float* Y,
    const float* dY,
    float* dX,
    CPUContext* /* context */) const {
  const int size = std::accumulate(
      // NOLINTNEXTLINE(modernize-use-transparent-functors)
      Y_dims.cbegin(), Y_dims.cend(), 1, std::multiplies<int>());
  ConstEigenVectorArrayMap<float> dY_arr(dY, size);
  ConstEigenVectorArrayMap<float> Y_arr(Y, size);
  EigenVectorMap<float>(dX, size) = dY_arr * (1 - Y_arr * Y_arr);
  return true;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    TanhGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        TanhGradientFunctor<CPUContext>>);

namespace {

class GetTanhGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "TanhGradient",
        "",
        std::vector<std::string>{O(0), GO(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(Tanh, GetTanhGradient);

} // namespace caffe2
