#include "caffe2/operators/sigmoid_op.h"

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

namespace caffe2 {

template <>
template <typename T>
bool SigmoidGradientFunctor<CPUContext>::Forward(
    const std::vector<int>& dY_dims,
    const std::vector<int>& /* Y_dims */,
    const T* dY,
    const T* Y,
    T* dX,
    CPUContext* /* context */) const {
  const int size = std::accumulate(
      dY_dims.cbegin(), dY_dims.cend(), 1, std::multiplies<int>());
  ConstEigenVectorArrayMap<T> dY_arr(dY, size);
  ConstEigenVectorArrayMap<T> Y_arr(Y, size);
  EigenVectorArrayMap<T>(dX, size) = dY_arr * Y_arr * (1. - Y_arr);
  return true;
}

REGISTER_CPU_OPERATOR(
    SigmoidGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        SigmoidGradientFunctor<CPUContext>>);

namespace {

class GetSigmoidGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SigmoidGradient",
        "",
        std::vector<std::string>{GO(0), O(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

REGISTER_GRADIENT(Sigmoid, GetSigmoidGradient);

} // namespace caffe2
