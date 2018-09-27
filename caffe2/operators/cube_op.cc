#include "caffe2/operators/cube_op.h"
#include "caffe2/utils/eigen_utils.h"

#include <algorithm>
#include <functional>
#include <string>

namespace caffe2 {

template <>
template <typename T>
bool CubeGradientFunctor<CPUContext>::Forward(
    const std::vector<int>& dY_dims,
    const std::vector<int>& /* X_dims */,
    const T* dY,
    const T* X,
    T* dX,
    CPUContext* /* context */) const {
  const int size = std::accumulate(
      dY_dims.cbegin(), dY_dims.cend(), 1, std::multiplies<int>());
  EigenVectorMap<T>(dX, size) = ConstEigenVectorArrayMap<T>(dY, size) *
      ConstEigenVectorArrayMap<T>(X, size).square() * T(3);
  return true;
}

REGISTER_CPU_OPERATOR(
    Cube,
    UnaryElementwiseOp<NumericTypes, CPUContext, CubeFunctor<CPUContext>>);
REGISTER_CPU_OPERATOR(
    CubeGradient,
    BinaryElementwiseOp<
        NumericTypes,
        CPUContext,
        CubeGradientFunctor<CPUContext>>);

OPERATOR_SCHEMA(Cube)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .Input(0, "X", "*(type: Tensor`<float>`)* Input tensor.")
    .Output(
        0,
        "Y",
        "*(type: Tensor`<float>`)* Output tensor calculated as the cube of the input tensor, element-wise.");

OPERATOR_SCHEMA(CubeGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .IdenticalTypeAndShape();

namespace {

class GetCubeGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "CubeGradient",
        "",
        std::vector<std::string>{GO(0), I(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

REGISTER_GRADIENT(Cube, GetCubeGradient);

} // namespace caffe2
