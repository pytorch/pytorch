// ------------------------------------------------------------------
// GroupNorm op in Caffe2 for CPU
// Written by Kaiming He
// Improved by Xiaomeng Yang
// see https://arxiv.org/abs/1803.08494
// This is a stand-alone op: Y = gamma * (X - mu) / sig + beta
// ------------------------------------------------------------------

#include "caffe2/operators/group_norm_op.h"

#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

// Math:
// Y = gamma * (X - mu) * rsig + beta
// let s = gamma * rsig
// let b = beta - gamma * mu * rsig
// Y = s * X + b
// let n = K * HxW
// dL/dX = dL/dY * dY/dX = dL/dY * (d(s * X)/dX + db/dX)
// d(s * X)/dX = s + X * ds/dX = s + gamma * X * drsig/dX
// db/dX = -gamma * u * drsig/dX - gamma * rsig * dmu/dX
// drsig/dX = -rsig^3 * (X - mu) / n
// dmu/dX = 1 / n

namespace {

template <typename T, StorageOrder kOrder>
void ComputeInternalGradients(
    int N,
    int C,
    int HxW,
    const T* dY,
    const T* X,
    T* ds,
    T* db);

template <>
void ComputeInternalGradients<float, StorageOrder::NCHW>(
    const int N,
    const int C,
    const int HxW,
    const float* dY,
    const float* X,
    float* ds,
    float* db) {
  ConstEigenArrayMap<float> dY_arr(dY, HxW, N * C);
  ConstEigenArrayMap<float> X_arr(X, HxW, N * C);
  for (int i = 0; i < N * C; ++i) {
    ds[i] = (dY_arr.col(i) * X_arr.col(i)).sum();
    db[i] = dY_arr.col(i).sum();
  }
}

template <>
void ComputeInternalGradients<float, StorageOrder::NHWC>(
    const int N,
    const int C,
    const int HxW,
    const float* dY,
    const float* X,
    float* ds,
    float* db) {
  EigenArrayMap<float> ds_arr(ds, C, N);
  EigenArrayMap<float> db_arr(db, C, N);
  for (int i = 0; i < N; ++i) {
    ConstEigenArrayMap<float> dY_arr(dY + i * C * HxW, C, HxW);
    ConstEigenArrayMap<float> X_arr(X + i * C * HxW, C, HxW);
    ds_arr.col(i) = dY_arr.col(0) * X_arr.col(0);
    db_arr.col(i) = dY_arr.col(0);
    for (int j = 1; j < HxW; ++j) {
      ds_arr.col(i) += dY_arr.col(j) * X_arr.col(j);
      db_arr.col(i) += dY_arr.col(j);
    }
  }
}

template <typename T>
void ComputeGradientFusedParams(
    const int N,
    const int G,
    const int K,
    const int HxW,
    const T* ds,
    const T* db,
    const T* mu,
    const T* rsig,
    const T* gamma,
    T* dY_scale,
    T* X_scale,
    T* bias) {
  ConstEigenArrayMap<T> rsig_arr(rsig, G, N);
  ConstEigenArrayMap<T> gamma_arr(gamma, K, G);
  for (int i = 0; i < N; ++i) {
    EigenArrayMap<T>(dY_scale + i * G * K, K, G) =
        gamma_arr.rowwise() * (rsig_arr.col(i).transpose());
  }
  ConstEigenVectorArrayMap<T> mu_arr(mu, N * G);
  ConstEigenVectorArrayMap<T> rsig_vec(rsig, N * G);
  EigenVectorArrayMap<T> X_scale_arr(X_scale, N * G);
  EigenVectorArrayMap<T> bias_arr(bias, N * G);
  for (int i = 0; i < N; ++i) {
    ConstEigenArrayMap<T> ds_arr(ds + i * G * K, K, G);
    ConstEigenArrayMap<T> db_arr(db + i * G * K, K, G);
    for (int j = 0; j < G; ++j) {
      X_scale_arr(i * G + j) = (ds_arr.col(j) * gamma_arr.col(j)).sum();
      bias_arr(i * G + j) = (db_arr.col(j) * gamma_arr.col(j)).sum();
    }
  }
  const T alpha = T(1) / static_cast<T>(K * HxW);
  X_scale_arr = (bias_arr * mu_arr - X_scale_arr) * rsig_vec.cube() * alpha;
  bias_arr = -X_scale_arr * mu_arr - bias_arr * rsig_vec * alpha;
}

template <typename T, StorageOrder kOrder>
void GroupNormBackward(
    int N,
    int G,
    int K,
    int HxW,
    const T* dY_scale,
    const T* dY,
    const T* X_scale,
    const T* X,
    const T* bias,
    T* dX);

template <>
void GroupNormBackward<float, StorageOrder::NCHW>(
    const int N,
    const int G,
    const int K,
    const int HxW,
    const float* dY_scale,
    const float* dY,
    const float* X_scale,
    const float* X,
    const float* bias,
    float* dX) {
  const int C = G * K;
  ConstEigenArrayMap<float> dY_arr(dY, HxW, N * C);
  ConstEigenArrayMap<float> X_arr(X, HxW, N * C);
  EigenArrayMap<float> dX_arr(dX, HxW, N * C);
  for (int i = 0; i < N * G; ++i) {
    for (int j = 0; j < K; ++j) {
      const int c = i * K + j;
      dX_arr.col(c) =
          dY_arr.col(c) * dY_scale[c] + X_arr.col(c) * X_scale[i] + bias[i];
    }
  }
}

template <>
void GroupNormBackward<float, StorageOrder::NHWC>(
    const int N,
    const int G,
    const int K,
    const int HxW,
    const float* dY_scale,
    const float* dY,
    const float* X_scale,
    const float* X,
    const float* bias,
    float* dX) {
  const int C = G * K;
  ConstEigenArrayMap<float> X_scale_arr(X_scale, G, N);
  ConstEigenArrayMap<float> bias_arr(bias, G, N);
  for (int n = 0; n < N; ++n) {
    ConstEigenArrayMap<float> dY_scale_arr(dY_scale + n * C, K, G);
    for (int i = 0; i < HxW; ++i) {
      const int m = n * HxW + i;
      ConstEigenArrayMap<float> dY_arr(dY + m * C, K, G);
      ConstEigenArrayMap<float> X_arr(X + m * C, K, G);
      EigenArrayMap<float> dX_arr(dX + m * C, K, G);
      dX_arr = (dY_arr * dY_scale_arr +
                X_arr.rowwise() * X_scale_arr.col(n).transpose())
                   .rowwise() +
          bias_arr.col(n).transpose();
    }
  }
}

template <typename T>
void GammaBetaBackward(
    const int N,
    const int G,
    const int K,
    const T* ds,
    const T* db,
    const T* mu,
    const T* rsig,
    T* dgamma,
    T* dbeta) {
  const int C = G * K;
  ConstEigenArrayMap<T> ds0_arr(ds, K, G);
  ConstEigenArrayMap<T> db0_arr(db, K, G);
  ConstEigenArrayMap<T> mu_arr(mu, G, N);
  ConstEigenArrayMap<T> rsig_arr(rsig, G, N);
  EigenArrayMap<T> dgamma_arr(dgamma, K, G);
  EigenArrayMap<T> dbeta_arr(dbeta, K, G);
  dgamma_arr =
      (ds0_arr - db0_arr.rowwise() * mu_arr.col(0).transpose()).rowwise() *
      rsig_arr.col(0).transpose();
  dbeta_arr = db0_arr;
  for (int i = 1; i < N; ++i) {
    ConstEigenArrayMap<T> dsi_arr(ds + i * C, K, G);
    ConstEigenArrayMap<T> dbi_arr(db + i * C, K, G);
    dgamma_arr +=
        (dsi_arr - dbi_arr.rowwise() * mu_arr.col(i).transpose()).rowwise() *
        rsig_arr.col(i).transpose();
    dbeta_arr += dbi_arr;
  }
}

} // namespace

template <>
void GroupNormOp<float, CPUContext>::ComputeFusedParams(
    const int N,
    const int G,
    const int K,
    const float* mu,
    const float* rsig,
    const float* gamma,
    const float* beta,
    float* scale,
    float* bias) {
  const int C = G * K;
  ConstEigenArrayMap<float> mu_arr(mu, G, N);
  ConstEigenArrayMap<float> rsig_arr(rsig, G, N);
  ConstEigenArrayMap<float> gamma_arr(gamma, K, G);
  ConstEigenArrayMap<float> beta_arr(beta, K, G);
  for (int i = 0; i < N; ++i) {
    EigenArrayMap<float> scale_arr(scale + i * C, K, G);
    EigenArrayMap<float> bias_arr(bias + i * C, K, G);
    scale_arr = gamma_arr.rowwise() * rsig_arr.col(i).transpose();
    bias_arr = beta_arr - scale_arr.rowwise() * mu_arr.col(i).transpose();
  }
}

template <>
void GroupNormOp<float, CPUContext>::GroupNormForwardNCHW(
    const int N,
    const int C,
    const int HxW,
    const float* X,
    const float* scale,
    const float* bias,
    float* Y) {
  EigenArrayMap<float>(Y, HxW, N * C) =
      (ConstEigenArrayMap<float>(X, HxW, N * C).rowwise() *
       ConstEigenVectorArrayMap<float>(scale, N * C).transpose())
          .rowwise() +
      ConstEigenVectorArrayMap<float>(bias, N * C).transpose();
}

template <>
void GroupNormOp<float, CPUContext>::GroupNormForwardNHWC(
    const int N,
    const int C,
    const int HxW,
    const float* X,
    const float* scale,
    const float* bias,
    float* Y) {
  const int stride = HxW * C;
  for (int i = 0; i < N; ++i) {
    EigenArrayMap<float>(Y + i * stride, C, HxW) =
        (ConstEigenArrayMap<float>(X + i * stride, C, HxW).colwise() *
         ConstEigenVectorArrayMap<float>(scale + i * C, C))
            .colwise() +
        ConstEigenVectorArrayMap<float>(bias + i * C, C);
  }
}

template <>
bool GroupNormOp<float, CPUContext>::RunOnDeviceWithOrderNHWC(
    const int N,
    const int G,
    const int K,
    const int HxW,
    const float* X,
    const float* gamma,
    const float* beta,
    float* Y,
    float* mu,
    float* rsig) {
  const int C = G * K;
  ReinitializeTensor(&scale_, {N, C}, at::dtype<float>().device(CPU));
  ReinitializeTensor(&bias_, {N, C}, at::dtype<float>().device(CPU));
  float* scale_data = scale_.mutable_data<float>();
  float* bias_data = bias_.mutable_data<float>();
  EigenVectorArrayMap<float> mu_arr(mu, N * G);
  EigenVectorArrayMap<float> rsig_arr(rsig, N * G);
  mu_arr.setZero();
  rsig_arr.setZero();
  for (int n = 0; n < N; ++n) {
    for (int i = 0; i < HxW; ++i) {
      const int m = n * HxW + i;
      ConstEigenArrayMap<float> X_arr(X + m * C, K, G);
      for (int j = 0; j < G; ++j) {
        mu_arr(n * G + j) += X_arr.col(j).sum();
        rsig_arr(n * G + j) += X_arr.col(j).square().sum();
      }
    }
  }
  const float scale = 1.0f / static_cast<float>(K * HxW);
  mu_arr *= scale;
  rsig_arr = (rsig_arr * scale - mu_arr.square() + epsilon_).rsqrt();
  ComputeFusedParams(N, G, K, mu, rsig, gamma, beta, scale_data, bias_data);
  GroupNormForwardNHWC(N, C, HxW, X, scale_data, bias_data, Y);
  return true;
}

// Math:
// let: s = gamma * rsig
// let: b = beta - mu * gamma * rsig
// then: Y = s * X + b
template <>
bool GroupNormGradientOp<float, CPUContext>::RunOnDeviceWithOrderNCHW(
    const int N,
    const int G,
    const int K,
    const int HxW,
    const float* dY_data,
    const float* X_data,
    const float* mu_data,
    const float* rsig_data,
    const float* gamma_data,
    float* dX_data,
    float* dgamma_data,
    float* dbeta_data) {
  const int C = G * K;
  ReinitializeTensor(&ds_, {N, C}, at::dtype<float>().device(CPU));
  ReinitializeTensor(&db_, {N, C}, at::dtype<float>().device(CPU));
  ReinitializeTensor(&dY_scale_, {N, C}, at::dtype<float>().device(CPU));
  ReinitializeTensor(&X_scale_, {N, G}, at::dtype<float>().device(CPU));
  ReinitializeTensor(&bias_, {N, G}, at::dtype<float>().device(CPU));
  float* ds_data = ds_.mutable_data<float>();
  float* db_data = db_.mutable_data<float>();
  float* dY_scale_data = dY_scale_.mutable_data<float>();
  float* X_scale_data = X_scale_.mutable_data<float>();
  float* bias_data = bias_.mutable_data<float>();
  ComputeInternalGradients<float, StorageOrder::NCHW>(
      N, C, HxW, dY_data, X_data, ds_data, db_data);
  ComputeGradientFusedParams<float>(
      N,
      G,
      K,
      HxW,
      ds_data,
      db_data,
      mu_data,
      rsig_data,
      gamma_data,
      dY_scale_data,
      X_scale_data,
      bias_data);
  GroupNormBackward<float, StorageOrder::NCHW>(
      N,
      G,
      K,
      HxW,
      dY_scale_data,
      dY_data,
      X_scale_data,
      X_data,
      bias_data,
      dX_data);
  GammaBetaBackward<float>(
      N, G, K, ds_data, db_data, mu_data, rsig_data, dgamma_data, dbeta_data);
  return true;
}

template <typename T, class Context>
bool GroupNormGradientOp<T, Context>::RunOnDeviceWithOrderNHWC(
    const int N,
    const int G,
    const int K,
    const int HxW,
    const T* dY_data,
    const T* X_data,
    const T* mu_data,
    const T* rsig_data,
    const T* gamma_data,
    T* dX_data,
    T* dgamma_data,
    T* dbeta_data) {
  const int C = G * K;
  ReinitializeTensor(&ds_, {N, C}, at::dtype<float>().device(CPU));
  ReinitializeTensor(&db_, {N, C}, at::dtype<float>().device(CPU));
  ReinitializeTensor(&dY_scale_, {N, C}, at::dtype<float>().device(CPU));
  ReinitializeTensor(&X_scale_, {N, G}, at::dtype<float>().device(CPU));
  ReinitializeTensor(&bias_, {N, G}, at::dtype<float>().device(CPU));
  float* ds_data = ds_.mutable_data<float>();
  float* db_data = db_.mutable_data<float>();
  float* dY_scale_data = dY_scale_.mutable_data<float>();
  float* X_scale_data = X_scale_.mutable_data<float>();
  float* bias_data = bias_.mutable_data<float>();
  ComputeInternalGradients<float, StorageOrder::NHWC>(
      N, C, HxW, dY_data, X_data, ds_data, db_data);
  ComputeGradientFusedParams<float>(
      N,
      G,
      K,
      HxW,
      ds_data,
      db_data,
      mu_data,
      rsig_data,
      gamma_data,
      dY_scale_data,
      X_scale_data,
      bias_data);
  GroupNormBackward<float, StorageOrder::NHWC>(
      N,
      G,
      K,
      HxW,
      dY_scale_data,
      dY_data,
      X_scale_data,
      X_data,
      bias_data,
      dX_data);
  GammaBetaBackward<float>(
      N, G, K, ds_data, db_data, mu_data, rsig_data, dgamma_data, dbeta_data);
  return true;
}

REGISTER_CPU_OPERATOR(GroupNorm, GroupNormOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    GroupNormGradient,
    GroupNormGradientOp<float, CPUContext>);

// Warning: mu and rsig are for backward usage or reference. They should NOT be
// used as forward activations as they have no direct gradients computed.

// Input: X, gamma, beta; Output: Y, mu, sig
OPERATOR_SCHEMA(GroupNorm)
    .NumInputs(3)
    .NumOutputs({1, 3})
    .SetDoc(R"DOC(
Group Normalization (GN) operation: https://arxiv.org/abs/1803.08494
)DOC")
    .Arg("num_groups", "(int) default 32; number of groups used by GN.")
    .Arg("epsilon", "(float) default 1e-5; small constant added to var.")
    .Input(
        0,
        "X",
        ">=4D feature map input of shape (N, C, H, W) or (N, C, T, H, W)")
    .Input(
        1,
        "gamma",
        "The scale as a 1-dimensional tensor of size C to be applied to the "
        "output.")
    .Input(
        2,
        "beta",
        "The bias as a 1-dimensional tensor of size C to be applied to the "
        "output.")
    .Output(0, "Y", "The output >=4-dimensional tensor of the same shape as X.")
    .Output(
        1,
        "mean",
        "The mean of shape (N, G). "
        "For backward usage or reference. "
        "Cannot be used as activations.")
    .Output(
        2,
        "std",
        "The std of shape (N, G). "
        "For backward usage or reference. "
        "Cannot be used as activations.");

// Input: dY, X, gamma, beta, mu, sig; Output: dX, dgamma, dbeta
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-avoid-magic-numbers)
OPERATOR_SCHEMA(GroupNormGradient).NumInputs(6).NumOutputs(3);

namespace {

class GetGroupNormGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "GroupNormGradient",
        "",
        std::vector<std::string>{GO(0), I(0), I(1), I(2), O(1), O(2)},
        std::vector<std::string>{GI(0), GI(1), GI(2)});
  }
};

} // namespace

REGISTER_GRADIENT(GroupNorm, GetGroupNormGradient);

} // namespace caffe2
