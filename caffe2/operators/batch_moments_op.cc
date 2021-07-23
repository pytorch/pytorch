#include "caffe2/operators/batch_moments_op.h"

#include <string>
#include <vector>

#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <>
bool BatchMomentsOp<float, CPUContext>::ComputeBatchMomentsNCHW(
    const int N,
    const int C,
    const int HxW,
    const float* X,
    float* mu,
    float* var) {
  math::Set<float, CPUContext>(C, 0.0f, mu, &context_);
  math::Set<float, CPUContext>(C, 0.0f, var, &context_);
  EigenVectorArrayMap<float> mu_arr(mu, C);
  EigenVectorArrayMap<float> var_arr(var, C);
  const float* X_ptr = X;
  const int stride = C * HxW;
  for (int i = 0; i < N; ++i) {
    ConstEigenArrayMap<float> X_arr(X_ptr, HxW, C);
    mu_arr += X_arr.colwise().sum();
    var_arr += X_arr.square().colwise().sum();
    X_ptr += stride;
  }
  const float scale = 1.0f / static_cast<float>(N * HxW);
  math::Scale<float, float, CPUContext>(C, scale, mu, mu, &context_);
  math::Scale<float, float, CPUContext>(C, scale, var, var, &context_);
  return true;
}

template <>
bool BatchMomentsOp<float, CPUContext>::ComputeBatchMomentsNHWC(
    const int N,
    const int C,
    const int HxW,
    const float* X,
    float* mu,
    float* var) {
  ConstEigenArrayMap<float> X_arr(X, C, N * HxW);
  EigenVectorMap<float>(mu, C) = X_arr.rowwise().mean();
  EigenVectorMap<float>(var, C) = X_arr.square().rowwise().mean();
  return true;
}

template <>
bool BatchMomentsGradientOp<float, CPUContext>::ComputeBatchMomentsGradientNCHW(
    const int N,
    const int C,
    const int HxW,
    const float* dmu,
    const float* dvar,
    const float* X,
    float* dX) {
  ConstEigenVectorArrayMap<float> dmu_arr(dmu, C);
  ConstEigenVectorArrayMap<float> dvar_arr(dvar, C);
  const float* X_ptr = X;
  float* dX_ptr = dX;
  const int stride = C * HxW;
  for (int i = 0; i < N; ++i) {
    EigenArrayMap<float> dX_arr(dX_ptr, HxW, C);
    dX_arr = ConstEigenArrayMap<float>(X_ptr, HxW, C).rowwise() *
        dvar_arr.transpose() * 2.0f;
    dX_arr.rowwise() += dmu_arr.transpose();
    X_ptr += stride;
    dX_ptr += stride;
  }
  const float scale = 1.0f / static_cast<float>(N * HxW);
  math::Scale<float, float, CPUContext>(N * C * HxW, scale, dX, dX, &context_);
  return true;
}

template <>
bool BatchMomentsGradientOp<float, CPUContext>::ComputeBatchMomentsGradientNHWC(
    const int N,
    const int C,
    const int HxW,
    const float* dmu,
    const float* dvar,
    const float* X,
    float* dX) {
  const float scale = 1.0f / static_cast<float>(N * HxW);
  EigenArrayMap<float> dX_arr(dX, C, N * HxW);
  dX_arr = ConstEigenArrayMap<float>(X, C, N * HxW).colwise() *
      ConstEigenVectorArrayMap<float>(dvar, C) * 2.0f;
  dX_arr.colwise() += ConstEigenVectorArrayMap<float>(dmu, C);
  math::Scale<float, float, CPUContext>(N * C * HxW, scale, dX, dX, &context_);
  return true;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(BatchMoments, BatchMomentsOp<float, CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    BatchMomentsGradient,
    BatchMomentsGradientOp<float, CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(BatchMoments).NumInputs(1).NumOutputs(2);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(BatchMomentsGradient).NumInputs(3).NumOutputs(1);

namespace {

class GetBatchMomentsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;

  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "BatchMomentsGradient",
        "",
        std::vector<std::string>{GO(0), GO(1), I(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(BatchMoments, GetBatchMomentsGradient);

} // namespace caffe2
