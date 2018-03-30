#include "caffe2/operators/lpnorm_op.h"

namespace caffe2 {

template <>
bool LpNormOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(X_IN);
  auto* norm = Output(OUT);
  norm->Resize(1);
  const float* X_data = X.data<float>();
  const float size = average_ ? (float)X.size() : 1.0f;
  CAFFE_ENFORCE_GT(size, 0);
  if (p_ == 1) {
    *(norm->mutable_data<float>()) =
        (ConstEigenVectorMap<float>(X_data, X.size()).array()).abs().sum() /
        size;
    // L1(x) = sum(|x|), L1_average(x) = sum(\x\) / x.size()
  } else if (p_ == 2) {
    *(norm->mutable_data<float>()) =
        (ConstEigenVectorMap<float>(X_data, X.size()).array()).square().sum() /
        size;
    // L2(x) = (sum(|x|^2)), L2_average(x) = sum(|x|^2) / x.size()
  }
  return true;
}

template <>
bool LpNormGradientOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(X_IN);
  auto& dnorm = Input(DER_NORM_IN);
  auto* dX = Output(DER_X_OUT);
  CAFFE_ENFORCE_EQ(dnorm.ndim(), 1);
  CAFFE_ENFORCE_EQ(dnorm.dim32(0), 1);
  dX->ResizeLike(X);
  const float kEps = 1e-12f;
  const float size = average_ ? (float)X.size() : 1.0f;
  if (p_ == 1) {
    // Todo: implement in eigen
    for (int i = 0; i < X.size(); ++i) {
      float temp = (X.data<float>())[i];
      if (temp < -kEps) {
        dX->mutable_data<float>()[i] = -(dnorm.data<float>())[0] / size;
      } else if (temp > kEps) {
        dX->mutable_data<float>()[i] = (dnorm.data<float>())[0] / size;
      } else {
        dX->mutable_data<float>()[i] = 0;
      }
    }
  } else if (p_ == 2) {
    EigenVectorMap<float>(dX->mutable_data<float>(), X.size()).array() =
        ConstEigenVectorMap<float>(X.data<float>(), X.size()).array() * 2.0f *
        ((dnorm.data<float>())[0] / size);
  }

  return true;
}

namespace {
// LpNorm
REGISTER_CPU_OPERATOR(LpNorm, LpNormOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(LpNormGradient, LpNormGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(LpNorm)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Given one input float tensor X, and produces one output float tensor
of the Lp norm of tensor X, computed as Lp(x) = sum over |x^p|,
in which p is either 1 or 2(currently only supports l1 and l2 norm),
determined by the argument p.
)DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Z", "1D output tensor")
    .Arg("p", "Order of the norm in p-norm")
    .Arg(
        "average",
        "whehther we calculate norm or averaged_norm."
        "The Lp_averaged_norm(x) is defined as"
        "Lp_averaged_norm(x) = LpNorm(x) / size(x)");

OPERATOR_SCHEMA(LpNormGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Given one input float tensor X, derivative dout, and produces one output
float tensor dX. dX is the derivative of the Lp norm of tensor X, computed as
dx = d(sum over |x^p|)/dx, in which p is either 1 or 2(currently only
supports l1 and l2 norm) determined by the argument p.
)DOC")
    .Input(0, "X", "1D input tensor")
    .Input(1, "dout", "1D input tensor")
    .Output(0, "dx", "1D output tensor")
    .Arg("p", "Order of the norm in p-norm")
    .Arg(
        "average",
        "whehther we calculate norm or averaged_norm."
        "The Lp_averaged_norm(x) is defined as"
        "Lp_averaged_normgradient(x) = LpNormGradient(x) / size(x)");

class GetLpNormGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "LpNormGradient",
        "",
        vector<string>{I(0), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(LpNorm, GetLpNormGradient);
} // namespace

} // namespace caffe2
