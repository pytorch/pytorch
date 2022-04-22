#include "caffe2/operators/layer_norm_op.h"

#include <array>

#include "caffe2/core/export_c10_op_to_caffe2.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <>
template <typename T>
void LayerNormOp<CPUContext>::ComputeSigmaAndFusedParams(
    const int N,
    const float eps,
    const T* mean,
    const T* var,
    T* sigma,
    T* scale,
    T* bias) {
  ConstEigenVectorArrayMap<T> var_arr(var, N);
  EigenVectorArrayMap<T> sigma_arr(sigma, N);
  sigma_arr = var_arr + static_cast<T>(eps);
  math::Rsqrt<T, CPUContext>(N, sigma, scale, &context_);
  math::Mul<T, CPUContext>(N, scale, sigma, sigma, &context_);
  EigenVectorArrayMap<T>(bias, N) = -ConstEigenVectorArrayMap<T>(scale, N) *
      ConstEigenVectorArrayMap<T>(mean, N);
}

template <>
template <typename T>
void LayerNormOp<CPUContext>::LayerNormForward(
    const int M,
    const int N,
    const T* X,
    const T* scale,
    const T* bias,
    const T* gamma,
    const T* beta,
    T* Y) {
  ConstEigenArrayMap<T> X_arr(X, N, M);
  ConstEigenVectorArrayMap<T> scale_arr(scale, M);
  ConstEigenVectorArrayMap<T> bias_arr(bias, M);
  EigenArrayMap<T> Y_arr(Y, N, M);
  if (gamma != nullptr && beta != nullptr) {
    ConstEigenVectorArrayMap<T> gamma_arr(gamma, N);
    ConstEigenVectorArrayMap<T> beta_arr(beta, N);
    Y_arr = (((X_arr.rowwise() * scale_arr.transpose()).rowwise() +
              bias_arr.transpose())
                 .colwise() *
             gamma_arr)
                .colwise() +
        beta_arr;
  } else {
    CAFFE_ENFORCE(gamma == nullptr);
    CAFFE_ENFORCE(beta == nullptr);
    Y_arr = (X_arr.rowwise() * scale_arr.transpose()).rowwise() +
        bias_arr.transpose();
  }
}

REGISTER_CPU_OPERATOR(LayerNorm, LayerNormOp<CPUContext>);

template <>
template <typename T>
void LayerNormGradientOp<CPUContext>::ComputeInternalGradients(
    const int M,
    const int N,
    const T* dY,
    const T* X,
    const T* gamma,
    T* dYxX,
    T* ds,
    T* db) {
  math::Mul<T, CPUContext>(M * N, dY, X, dYxX, &context_);
  ConstEigenArrayMap<T> dYxX_arr(dYxX, N, M);
  ConstEigenArrayMap<T> dY_arr(dY, N, M);
  if (gamma != nullptr) {
    ConstEigenVectorArrayMap<T> gamma_arr(gamma, N);
    for (int i = 0; i < M; ++i) {
      ds[i] = (dYxX_arr.col(i) * gamma_arr).sum();
      db[i] = (dY_arr.col(i) * gamma_arr).sum();
    }
  } else {
    EigenVectorArrayMap<T>(ds, M) = dYxX_arr.colwise().sum();
    EigenVectorArrayMap<T>(db, M) = dY_arr.colwise().sum();
  }
}

template <>
template <typename T>
void LayerNormGradientOp<CPUContext>::ComputeFusedParams(
    const int M,
    const int N,
    const T* mean,
    const T* sigma,
    const T* ds,
    const T* db,
    T* rstd,
    T* X_scale,
    T* bias,
    T* g_scale) {
  const T scale = T(1) / static_cast<T>(N);
  ConstEigenVectorArrayMap<T> mean_arr(mean, M);
  ConstEigenVectorArrayMap<T> ds_arr(ds, M);
  ConstEigenVectorArrayMap<T> db_arr(db, M);
  EigenVectorArrayMap<T> rstd_arr(rstd, M);
  EigenVectorArrayMap<T> X_scale_arr(X_scale, M);
  rstd_arr = ConstEigenVectorArrayMap<T>(sigma, M).inverse();
  X_scale_arr = (db_arr * mean_arr - ds_arr) * rstd_arr.cube() * scale;
  EigenVectorArrayMap<T>(bias, M) =
      -X_scale_arr * mean_arr - db_arr * rstd_arr * scale;
  if (g_scale != nullptr) {
    EigenVectorArrayMap<T>(g_scale, M) = -rstd_arr * mean_arr;
  }
}

template <>
template <typename T>
void LayerNormGradientOp<CPUContext>::LayerNormBackward(
    const int M,
    const int N,
    const T* dY,
    const T* X,
    const T* gamma,
    const T* dY_scale,
    const T* X_scale,
    const T* bias,
    T* dX) {
  ConstEigenArrayMap<T> dY_arr(dY, N, M);
  ConstEigenArrayMap<T> X_arr(X, N, M);
  EigenArrayMap<T> dX_arr(dX, N, M);
  if (gamma != nullptr) {
    ConstEigenVectorArrayMap<T> gamma_arr(gamma, N);
    for (int i = 0; i < M; ++i) {
      dX_arr.col(i) = dY_arr.col(i) * gamma_arr * dY_scale[i] +
          X_arr.col(i) * X_scale[i] + bias[i];
    }
  } else {
    ConstEigenVectorArrayMap<T> dY_scale_arr(dY_scale, M);
    ConstEigenVectorArrayMap<T> X_scale_arr(X_scale, M);
    ConstEigenVectorArrayMap<T> bias_arr(bias, M);
    dX_arr = (dY_arr.rowwise() * dY_scale_arr.transpose() +
              X_arr.rowwise() * X_scale_arr.transpose())
                 .rowwise() +
        bias_arr.transpose();
  }
}

template <>
template <typename T>
void LayerNormGradientOp<CPUContext>::GammaBetaBackward(
    const int M,
    const int N,
    const T* dYxX,
    const T* dY,
    const T* rstd,
    const T* g_scale,
    T* dgamma,
    T* dbeta) {
  math::Set<T, CPUContext>(N, T(0), dgamma, &context_);
  math::Set<T, CPUContext>(N, T(0), dbeta, &context_);
  ConstEigenArrayMap<T> dYxX_arr(dYxX, N, M);
  ConstEigenArrayMap<T> dY_arr(dY, N, M);
  EigenVectorArrayMap<T> dgamma_arr(dgamma, N);
  EigenVectorArrayMap<T> dbeta_arr(dbeta, N);
  for (int i = 0; i < M; ++i) {
    dgamma_arr += dYxX_arr.col(i) * rstd[i] + dY_arr.col(i) * g_scale[i];
    dbeta_arr += dY_arr.col(i);
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-avoid-magic-numbers)
OPERATOR_SCHEMA(LayerNormGradient).NumInputs({5, 6}).NumOutputs({1, 3});

REGISTER_CPU_OPERATOR(LayerNormGradient, LayerNormGradientOp<CPUContext>);

namespace {

class GetLayerNormGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    bool elementwise_affine = false;
    if (ArgumentHelper::HasArgument(Def(), "elementwise_affine")) {
      elementwise_affine = GetArgument(Def(), "elementwise_affine").i();
    }
    if (elementwise_affine) {
      return SingleGradientDef(
          "LayerNormGradient",
          "",
          std::vector<std::string>{GO(0), O(0), O(1), O(2), I(0), I(1)},
          std::vector<std::string>{GI(0), GI(1), GI(2)});
    } else {
      return SingleGradientDef(
          "LayerNormGradient",
          "",
          std::vector<std::string>{GO(0), O(0), O(1), O(2), I(0)},
          std::vector<std::string>{GI(0)});
    }
  }
};

} // namespace

REGISTER_GRADIENT(LayerNorm, GetLayerNormGradient);

OPERATOR_SCHEMA(LayerNorm)
    .NumInputs({1, 3})
    .NumOutputs(3)
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      std::vector<TensorShape> out(3);
      auto input_dims_long = GetDimsVector(in[0]);
      std::vector<int> input_dims(
          input_dims_long.begin(), input_dims_long.end());
      out[0] = CreateTensorShape(input_dims, TensorProto::FLOAT);

      ArgumentHelper helper(def);

      auto axis = helper.GetSingleArgument<int32_t>("axis", 1);
      const auto canonical_axis =
          canonical_axis_index_(axis, in[0].dims().size());
      std::vector<int> stat_dims(
          input_dims.begin(), input_dims.begin() + canonical_axis);
      stat_dims.push_back(1);
      out[1] = CreateTensorShape(stat_dims, TensorProto::FLOAT);
      out[2] = CreateTensorShape(stat_dims, TensorProto::FLOAT);
      return out;
    })
    .SetDoc(R"DOC(
Computes layer normalization as described in https://arxiv.org/pdf/1607.06450.pdf.
Given an input vector x \in [a_0, a_1, ...,a_{k-1}, a_k, ..., a_{n-1}],
this op treats dimensions a_k through a_{n-1} as feature vectors. For each
feature vector, the op contains the mean and standard deviation. Then,
it returns the normalized values (with respect to the feature vector).

Note that this op does not contain the scale an bias terms described in the
paper. Simply follow this op with an FC op to add those. Concretely, this op
implements:

h = \frac{1}{\sigma}(a - \mu)
where \mu = \frac{1}{H}\sum_{i=1}^{H} a_i
and \sigma = \sqrt{\frac{1}{H}\sum_{i=1}^{H}(a_i - \mu)^2}
where H is the number of hidden units (i.e. product of dimensions from 'axis'
to the end.)
)DOC")
    .Arg(
        "axis",
        "(int) default to 1; Describes axis of the inputs. Defaults to one "
        "because the 0th axis most likely describes the batch size")
    .Arg(
        "epsilon",
        "(float) default to 0.001. Small value to be added to the stdev when"
        " dividing out by that value. This prevents division by zero.")
    .Arg(
        "elementwise_affine",
        "(bool) default to False; If true, this op will do affine "
        "transformation after normalization.")
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
    .Output(1, "mean", "Mean values for each feature vector")
    .Output(2, "stddev", "Standard deviations for each feature vector");

} // namespace caffe2

C10_EXPORT_CAFFE2_OP_TO_C10_CPU(
    LayerNorm,
    "_caffe2::LayerNorm("
    "    Tensor X,"
    "    Tensor? gamma,"
    "    Tensor? beta,"
    "    int axis = 1,"
    "    float epsilon = 1e-5,"
    "    bool elementwise_affine = False"
    ") -> (Tensor Y, Tensor mean, Tensor std)",
    caffe2::LayerNormOp<caffe2::CPUContext>)

namespace caffe2 {

C10_EXPORT_C10_OP_TO_CAFFE2_CPU(
    "_caffe2::LayerNorm",
    C10LayerNorm_DontUseThisOpYet);

} // namespace caffe2
