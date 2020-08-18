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
operator()(const int N, const T* X, T* Y, CPUContext* /* context */) const {
  ConstEigenVectorArrayMap<T> X_arr(X, N);
  EigenVectorArrayMap<T>(Y, N) = X_arr * (T(1) + X_arr.exp()).log().tanh();
  return true;
}

template <>
template <typename T>
bool MishGradientOp<CPUContext>::DoRunWithType() {
  auto& Xin = Input(X);
  auto& Yin = Input(Y);
  auto& DYin = Input(DY);

  CAFFE_ENFORCE_EQ(Xin.numel(), Yin.numel());
  CAFFE_ENFORCE_EQ(DYin.numel(), Yin.numel());
  auto* DXout = Output(DX, Yin.sizes(), at::dtype<float>());

  const float* Xdata = Xin.template data<float>();
  const float* Ydata = Yin.template data<float>();
  const float* dYdata = DYin.template data<float>();
  float* dXdata = DXout->template mutable_data<float>();

  EigenVectorArrayMap<float> dXvec(dXdata, DXout->numel());
  ConstEigenVectorArrayMap<float> Xvec(Xdata, Xin.numel());
  ConstEigenVectorArrayMap<float> Yvec(Ydata, Yin.numel());
  ConstEigenVectorArrayMap<float> dYvec(dYdata, DYin.numel());

  // w = e^(3x) + 4*e^2x + e^x * (6 + 4x) + 4(1 + x)
  // q = (e^x + 1)^2 + 1
  // dX = dY * e^x * w / q^2
  dXvec = dYvec *
      (T(4) * (Xvec+T(1)) * (-T(3)*Xvec).exp() + T(4)*(-Xvec).exp() + T(1) + (T(4)*Xvec+T(6))*(-T(2)*Xvec).exp()) /
      (T(1) + T(4)*(-Xvec).exp() + T(8)*(-T(2)*Xvec).exp() + T(8)*(-T(3)*Xvec).exp() + T(4)*(-T(4)*Xvec).exp());

  return true;
}

REGISTER_CPU_OPERATOR(
    Mish,
    UnaryElementwiseOp<
        TensorTypes<float>,
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
OPERATOR_SCHEMA(MishGradient)
    .NumInputs(3)
    .NumOutputs(1)
    .AllowInplace({{2, 0}})
    .SetDoc(R"DOC(
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
