#include "caffe2/operators/reciprocal_op.h"

#include "caffe2/utils/eigen_utils.h"

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

namespace caffe2 {

template <>
template <typename T>
bool ReciprocalGradientFunctor<CPUContext>::Forward(
    const std::vector<int>& Y_dims,
    const std::vector<int>& /* dY_dims */,
    const T* Y,
    const T* dY,
    T* dX,
    CPUContext* /* context */) const {
  const int size = std::accumulate(
      Y_dims.cbegin(), Y_dims.cend(), 1, std::multiplies<int>());
  ConstEigenVectorArrayMap<T> dY_arr(dY, size);
  ConstEigenVectorArrayMap<T> Y_arr(Y, size);
  EigenVectorMap<T>(dX, size) = dY_arr * (-Y_arr.square());
  return true;
}

REGISTER_CPU_OPERATOR(
    ReciprocalGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        ReciprocalGradientFunctor<CPUContext>>);

namespace {

class GetReciprocalGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "ReciprocalGradient",
        "",
        std::vector<std::string>{O(0), GO(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

REGISTER_GRADIENT(Reciprocal, GetReciprocalGradient);

} // namespace caffe2
