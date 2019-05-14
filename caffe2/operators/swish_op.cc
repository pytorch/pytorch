#include "caffe2/operators/swish_op.h"

#include <string>
#include <vector>

#include "caffe2/core/types.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <>
template <typename T>
bool SwishFunctor<CPUContext>::
operator()(const int N, const T* X, T* Y, CPUContext* /* context */) const {
  ConstEigenVectorArrayMap<T> X_arr(X, N);
  EigenVectorArrayMap<T>(Y, N) = X_arr / (T(1) + (-X_arr).exp());
  return true;
}

template <>
template <typename T>
bool SwishGradientOp<CPUContext>::DoRunWithType() {
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

  // dx = dy * (y + sigmoid(x)*(1-y))
  dXvec = dYvec * (Yvec + (T(1) / (T(1) + (-Xvec).exp())) * (T(1) - Yvec));
  return true;
}

REGISTER_CPU_OPERATOR(
    Swish,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        SwishFunctor<CPUContext>>);
REGISTER_CPU_OPERATOR(SwishGradient, SwishGradientOp<CPUContext>);

// Input: X, output: Y
OPERATOR_SCHEMA(Swish)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Swish takes one input data (Tensor) and produces one output data
(Tensor) where the swish function, y = x / (1 + exp(-x)), is applied to the
tensor elementwise.
)DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D output tensor");
// Input: X, Y, dY, output: dX
OPERATOR_SCHEMA(SwishGradient)
    .NumInputs(3)
    .NumOutputs(1)
    .AllowInplace({{2, 0}})
    .SetDoc(R"DOC(
SwishGradient takes X, Y and dY and uses this to update dX according to the
chain rule and derivatives of the swish function.
)DOC");

namespace {

class GetSwishGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SwishGradient",
        "",
        std::vector<std::string>{I(0), O(0), GO(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

REGISTER_GRADIENT(Swish, GetSwishGradient);

} // namespace caffe2
