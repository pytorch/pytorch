#include "caffe2/operators/mish_op.h"

#include <string>
#include <vector>

#include "caffe2/core/types.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <>
template <typename T>
bool MishFunctor<CPUContext>::
operator()(const int N, const T* X, T* Y, CPUContext* context) const {
  ConstEigenVectorArrayMap<T> X_arr(X, N);
  EigenVectorArrayMap<T> Y_arr(Y, N);
  math::Exp<T, CPUContext>(N, X, Y, context);
  math::Log1p<T, CPUContext>(N, Y, Y, context);
  Y_arr = X_arr * Y_arr.tanh();
  return true;
}

template <>
template <typename T>
bool MishGradientOp<CPUContext>::DoRunWithType() {
  const auto& X = Input(INPUT);
  const auto& Y = Input(OUTPUT);
  const auto& dY = Input(OUTPUT_GRAD);

  CAFFE_ENFORCE_EQ(X.numel(), Y.numel());
  CAFFE_ENFORCE_EQ(dY.numel(), Y.numel());
  auto* dX = Output(INPUT_GRAD, X.sizes(), at::dtype<T>());

  const T* X_data = X.template data<T>();
  const T* Y_data = Y.template data<T>();
  const T* dY_data = dY.template data<T>();
  T* dX_data = dX->template mutable_data<T>();

  const int64_t N = X.numel();
  ConstEigenVectorArrayMap<T> X_arr(X_data, N);
  ConstEigenVectorArrayMap<T> Y_arr(Y_data, N);
  ConstEigenVectorArrayMap<T> dY_arr(dY_data, N);
  EigenVectorArrayMap<T> dX_arr(dX_data, N);

  math::Exp<T, CPUContext>(N, X_data, dX_data, &context_);
  math::Log1p<T, CPUContext>(N, dX_data, dX_data, &context_);
  math::Tanh<T, CPUContext>(N, dX_data, dX_data, &context_);
  dX_arr = dY_arr *
      (dX_arr +
       X_arr * (T(1) - dX_arr.square()) * T(0.5) *
           ((X_arr * T(0.5)).tanh() + T(1)));

  return true;
}

REGISTER_CPU_OPERATOR(
    Mish,
    UnaryElementwiseOp<
        TensorTypes<float, double>,
        CPUContext,
        MishFunctor<CPUContext>>);
REGISTER_CPU_OPERATOR(MishGradient, MishGradientOp<CPUContext>);

// Input: X, output: Y
OPERATOR_SCHEMA(Mish)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Mish takes one input data (Tensor) and produces one output data
(Tensor) where the Mish function, y = x * tanh(ln(1 + exp(x))), is applied to the
tensor elementwise.
)DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D output tensor");
// Input: X, Y, dY, output: dX
OPERATOR_SCHEMA(MishGradient).NumInputs(3).NumOutputs(1).SetDoc(R"DOC(
MishGradient takes X, Y and dY and uses this to update dX according to the
chain rule and derivatives of the Mish function.
)DOC");

namespace {

class GetMishGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "MishGradient",
        "",
        std::vector<std::string>{I(0), O(0), GO(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

REGISTER_GRADIENT(Mish, GetMishGradient);

} // namespace caffe2
