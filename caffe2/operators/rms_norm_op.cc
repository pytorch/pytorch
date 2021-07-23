#include "caffe2/operators/rms_norm_op.h"

#include <array>
#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "ATen/Parallel.h"

#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math/utils.h"

namespace caffe2 {

template <>
template <typename T>
bool RMSNormOp<CPUContext>::DoRunWithType() {
  const auto& X = Input(0);
  const auto& gamma = Input(1);
  const auto& beta = Input(2);
  auto* Y = Output(0, X.sizes(), at::dtype<T>());
  CAFFE_ENFORCE_GE(X.dim(), 2, "RMSNorm requires input dim >= 2.");
  const int canonical_axis = X.canonical_axis_index(axis_);
  const std::vector<int64_t> rms_dims(
      X.sizes().cbegin(), X.sizes().cbegin() + canonical_axis);
  auto* rrms = Output(1, rms_dims, at::dtype<T>());
  const int64_t M = X.size_to_dim(canonical_axis);
  const int64_t N = X.size_from_dim(canonical_axis);
  CAFFE_ENFORCE_EQ(gamma.numel(), N);
  CAFFE_ENFORCE_EQ(beta.numel(), N);

  const T* X_data = X.template data<T>();
  const T* gamma_data = gamma.template data<T>();
  const T* beta_data = beta.template data<T>();
  T* Y_data = Y->template data<T>();
  T* rrms_data = rrms->template data<T>();

  ConstEigenArrayMap<T> X_arr(X_data, N, M);
  ConstEigenVectorArrayMap<T> gamma_arr(gamma_data, N);
  ConstEigenVectorArrayMap<T> beta_arr(beta_data, N);
  EigenArrayMap<T> Y_arr(Y_data, N, M);
  at::parallel_for(0, M, 1, [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; ++i) {
      const T rrms_val =
          T(1) / std::sqrt(X_arr.col(i).square().mean() + static_cast<T>(eps_));
      Y_arr.col(i) = rrms_val * X_arr.col(i) * gamma_arr + beta_arr;
      rrms_data[i] = rrms_val;
    }
  });

  return true;
}

template <>
template <typename T>
void RMSNormGradientOp<CPUContext>::RMSNormBackward(
    int64_t M,
    int64_t N,
    const T* dY,
    const T* X,
    const T* gamma,
    const T* rrms,
    T* dX) {
  ConstEigenArrayMap<T> dY_arr(dY, N, M);
  ConstEigenArrayMap<T> X_arr(X, N, M);
  ConstEigenVectorArrayMap<T> gamma_arr(gamma, N);
  EigenArrayMap<T> dX_arr(dX, N, M);
  const T scale = T(1) / static_cast<T>(N);
  at::parallel_for(0, M, 1, [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; ++i) {
      const T ds = (dY_arr.col(i) * X_arr.col(i) * gamma_arr).sum();
      const T c1 = rrms[i];
      const T c2 = -scale * ds * math::utils::Cube<T>(rrms[i]);
      dX_arr.col(i) = c1 * dY_arr.col(i) * gamma_arr + c2 * X_arr.col(i);
    }
  });
}

template <>
template <typename T>
void RMSNormGradientOp<CPUContext>::GammaBetaBackward(
    int64_t M,
    int64_t N,
    const T* dY,
    const T* X,
    const T* rrms,
    T* dgamma,
    T* dbeta) {
  math::Set<T, CPUContext>(N, T(0), dgamma, &context_);
  math::Set<T, CPUContext>(N, T(0), dbeta, &context_);
  ConstEigenArrayMap<T> dY_arr(dY, N, M);
  ConstEigenArrayMap<T> X_arr(X, N, M);
  EigenVectorArrayMap<T> dgamma_arr(dgamma, N);
  EigenVectorArrayMap<T> dbeta_arr(dbeta, N);
  for (int64_t i = 0; i < M; ++i) {
    dgamma_arr += dY_arr.col(i) * X_arr.col(i) * rrms[i];
    dbeta_arr += dY_arr.col(i);
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(RMSNorm, RMSNormOp<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(RMSNormGradient, RMSNormGradientOp<CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(RMSNorm)
    .NumInputs(3)
    .NumOutputs(2)
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      std::vector<TensorShape> out(2);
      const auto input_dims_long = GetDimsVector(in[0]);
      const std::vector<int> input_dims(
          input_dims_long.cbegin(), input_dims_long.cend());
      out[0] = CreateTensorShape(input_dims, in[0].data_type());
      ArgumentHelper helper(def);
      const int axis = helper.GetSingleArgument<int32_t>("axis", 1);
      const int canonical_axis =
          canonical_axis_index_(axis, in[0].dims().size());
      const std::vector<int> rms_dims(
          input_dims.cbegin(), input_dims.cbegin() + canonical_axis);
      out[1] = CreateTensorShape(rms_dims, in[0].data_type());
      return out;
    })
    .Arg(
        "axis",
        "(int) default to 1; Describes axis of the inputs. Defaults to one "
        "because the 0th axis most likely describes the batch size")
    .Arg(
        "epsilon",
        "(float) default to 0.001. Small value to be added to the stdev when"
        " dividing out by that value. This prevents division by zero.")
    .Input(
        0,
        "input",
        "Input tensor which layer normalization will be applied to")
    .Input(
        1,
        "gamma",
        "scale tensor for elementwise_affine, the shape should be the same as "
        "the dimensions of X begin from axis")
    .Input(
        2,
        "beta",
        "bias tensor for elementwise_affine, the shape should be the same as "
        "the dimensions of X begin from axis")
    .Output(0, "output", "Normalized values")
    .Output(
        1,
        "rrms",
        "Reciprocal of root mean square for each feature vector");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(RMSNormGradient).NumInputs(4).NumOutputs(3);

namespace {

class GetRMSNormGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "RMSNormGradient",
        "",
        std::vector<std::string>{GO(0), I(0), I(1), O(1)},
        std::vector<std::string>{GI(0), GI(1), GI(2)});
  }
};

} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(RMSNorm, GetRMSNormGradient);

} // namespace caffe2
