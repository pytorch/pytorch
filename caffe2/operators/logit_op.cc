#include "caffe2/operators/logit_op.h"
#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {
struct LogitCPUFunctor {
  explicit LogitCPUFunctor(OperatorBase& op)
      : eps_(op.GetSingleArgument<float>("eps", 1e-6f)) {
    CAFFE_ENFORCE_GT(eps_, 0.0);
    CAFFE_ENFORCE_LT(eps_, 0.5);
  }
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CPUContext* /* unused */) {
    ConstEigenArrayMap<T> X(x, n, 1);
    EigenArrayMap<T> Y(y, n, 1);
    const T k_one = 1.0;

    Y = X.min(k_one - eps_);
    Y = Y.max(eps_);
    Y = (Y / (k_one - Y)).log();
  }

 private:
  float eps_;
};

template <>
bool LogitGradientOp<float, CPUContext>::RunOnDevice() {
  const auto& X = Input(0);
  const auto& dY = Input(1);
  auto* dX = Output(0);
  dX->ResizeLike(X);
  int channels = X.dim32(X.ndim() - 1);
  ConstEigenArrayMap<float> Xmat(
      X.template data<float>(), channels, X.size() / channels);
  ConstEigenArrayMap<float> dYmat(
      dY.template data<float>(), channels, X.size() / channels);
  EigenArrayMap<float> dXmat(
      dX->template mutable_data<float>(), channels, X.size() / channels);
  dXmat = (Xmat < eps_ || Xmat > 1.0 - eps_)
              .select(0, dYmat * ((1 - Xmat) * Xmat).inverse());
  return true;
}

REGISTER_CPU_OPERATOR(
    Logit,
    UnaryElementwiseWithArgsOp<
        TensorTypes<float>,
        CPUContext,
        LogitCPUFunctor>);

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

class GetLogitGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return vector<OperatorDef>{CreateOperatorDef(
        "LogitGradient",
        "",
        std::vector<string>{I(0), GO(0)},
        std::vector<string>{GI(0)})};
  }
};

REGISTER_GRADIENT(Logit, GetLogitGradient);
} // namespace caffe2
