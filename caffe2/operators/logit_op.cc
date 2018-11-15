#include "caffe2/operators/logit_op.h"

#include <string>
#include <vector>

#include "caffe2/operators/elementwise_ops.h"
#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

template <>
template <typename T>
bool LogitFunctor<CPUContext>::
operator()(const int size, const T* X, T* Y, CPUContext* /* context */) const {
  ConstEigenVectorMap<T> X_vec(X, size);
  EigenVectorMap<T> Y_vec(Y, size);
  Y_vec = X_vec.array().min(static_cast<T>(1.0f - eps_));
  Y_vec = Y_vec.array().max(eps_);
  Y_vec = (Y_vec.array() / (T(1) - Y_vec.array())).log();
  return true;
}

template <>
bool LogitGradientOp<float, CPUContext>::RunOnDevice() {
  const auto& X = Input(0);
  const auto& dY = Input(1);
  auto* dX = Output(0);
  dX->ResizeLike(X);
  int channels = X.dim32(X.dim() - 1);
  ConstEigenArrayMap<float> Xmat(
      X.template data<float>(), channels, X.numel() / channels);
  ConstEigenArrayMap<float> dYmat(
      dY.template data<float>(), channels, X.numel() / channels);
  EigenArrayMap<float> dXmat(
      dX->template mutable_data<float>(), channels, X.numel() / channels);
  dXmat = (Xmat < eps_ || Xmat > 1.0 - eps_)
              .select(0, dYmat * ((1 - Xmat) * Xmat).inverse());
  return true;
}

REGISTER_CPU_OPERATOR(
    Logit,
    UnaryElementwiseWithArgsOp<
        TensorTypes<float>,
        CPUContext,
        LogitFunctor<CPUContext>>);

REGISTER_CPU_OPERATOR(LogitGradient, LogitGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(Logit)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Elementwise logit transform: logit(x) = log(x / (1 - x)), where x is the
input data clampped in (eps, 1-eps).
)DOC")
    .Arg("eps (optional)", "small positive epsilon value, the default is 1e-6.")
    .Input(0, "X", "input float tensor")
    .Output(0, "Y", "output float tensor");

OPERATOR_SCHEMA(LogitGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .Input(0, "X", "input float tensor")
    .Input(1, "dY", "input float tensor")
    .Output(0, "dX", "output float tensor")
    .Arg("eps", "small positive epsilon value, the default is 1e-6.");

namespace {

class GetLogitGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return vector<OperatorDef>{CreateOperatorDef(
        "LogitGradient",
        "",
        std::vector<std::string>{I(0), GO(0)},
        std::vector<std::string>{GI(0)})};
  }
};

} // namespace

REGISTER_GRADIENT(Logit, GetLogitGradient);

} // namespace caffe2
