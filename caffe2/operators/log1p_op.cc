#include "caffe2/operators/log1p_op.h"
#include "caffe2/utils/eigen_utils.h"

#include <algorithm>
#include <functional>

namespace caffe2 {

template <>
template <typename T>
bool Log1pGradientFunctor<CPUContext>::Forward(
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
  EigenVectorMap<T>(dX, size) = dY_arr / (T(1) + X_arr);
  return true;
}

REGISTER_CPU_OPERATOR(
    Log1p,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, Log1pFunctor<CPUContext>>);
REGISTER_CPU_OPERATOR(
    Log1pGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        Log1pGradientFunctor<CPUContext>>);

OPERATOR_SCHEMA(Log1p)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Calculates Log1p of the given input tensor element-wise. This
operation can be done in an in-place fashion too, by providing the same input
and output blobs.

Github Link:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/log1p_op.cc
)DOC")
    .Input(0, "input", "Input data blob to be operated on.")
    .Output(0, "output", "Output data blob with same shape as input")
    .InheritOnnxSchema();

OPERATOR_SCHEMA(Log1pGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInput(0);

namespace {

class GetLog1pGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "Log1pGradient",
        "",
        std::vector<std::string>{I(0), GO(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

REGISTER_GRADIENT(Log1p, GetLog1pGradient);

} // namespace caffe2
