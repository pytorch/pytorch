#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {
namespace {
struct LogitCPUFunctor {
  explicit LogitCPUFunctor(OperatorBase& op)
      : eps_(op.GetSingleArgument<float>("eps", 1e-6)) {
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

REGISTER_CPU_OPERATOR(
    Logit,
    UnaryElementwiseWithArgsOp<
        TensorTypes<float>,
        CPUContext,
        LogitCPUFunctor>);

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
    .Input(1, "Y", "output float tensor");

GRADIENT_NOT_IMPLEMENTED_YET(Logit);

} // namespace
} // namespace caffe2
