// ------------------------------------------------------------------
// GroupNorm op in Caffe2 for CPU
// Written by Kaiming He
// Improved by Xiaomeng Yang
// see https://arxiv.org/abs/1803.08494
// This is a stand-alone op: Y = gamma * (X - mu) / sig + beta
// ------------------------------------------------------------------

#include "group_norm_op.h"

#include <array>

#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

template <typename T>
inline T Cube(const T& x) {
  return x * x * x;
}

template <typename T, StorageOrder kOrder>
void GroupNormForward(
    const std::array<int, 4>& dims,
    const T* X,
    const T* mu,
    const T* rsig,
    const T* gamma,
    const T* beta,
    T* Y) {
  constexpr int kGDim = kOrder == StorageOrder::NCHW ? 1 : 2;
  constexpr int kDDim = kOrder == StorageOrder::NCHW ? 2 : 3;
  const int size = dims[0] * dims[1] * dims[2] * dims[3];
  std::array<int, 4> index = {0, 0, 0, 0};
  for (int i = 0; i < size; ++i) {
    const int i_mu = index[0] * dims[kGDim] + index[kGDim];
    const int i_gamma = index[kGDim] * dims[kDDim] + index[kDDim];
    Y[i] = gamma[i_gamma] * (X[i] - mu[i_mu]) * rsig[i_mu] + beta[i_gamma];
    math::internal::IncreaseIndexInDims(4, dims.data(), index.data());
  }
}

template <typename T, StorageOrder kOrder>
void ComputeInternalGradients(
    const std::array<int, 4>& dims,
    const T* dY,
    const T* X,
    const T* gamma,
    T* ds,
    T* db) {
  constexpr int kGDim = kOrder == StorageOrder::NCHW ? 1 : 2;
  constexpr int kDDim = kOrder == StorageOrder::NCHW ? 2 : 3;
  const int size = dims[0] * dims[1] * dims[2] * dims[3];
  std::array<int, 4> index = {0, 0, 0, 0};
  for (int i = 0; i < size; ++i) {
    const int i_mu = index[0] * dims[kGDim] + index[kGDim];
    const int i_gamma = index[kGDim] * dims[kDDim] + index[kDDim];
    ds[i_mu] += gamma[i_gamma] * dY[i] * X[i];
    db[i_mu] += gamma[i_gamma] * dY[i];
    math::internal::IncreaseIndexInDims(4, dims.data(), index.data());
  }
}

// Math:
// Y = gamma * (X - mu) * rsig + beta
// let s = gamma * rsig
// let b = beta - mu * rsig
// Y = s * X + b
// let n = D * HxW
// dL/dX = dL/dY * dY/dX = dL/dY * (d(s * X)/dX + db/dX)
// d(s * X)/dX = s + X * ds/dX = s + gamma * X * drsig/dX
// db/dX = -u * drsig/dX - rsig * dmu/dX
// drsig/dX = -rsig^3 * (X - mu) / n
// dmu/dX = 1 / n
template <typename T, StorageOrder kOrder>
void GroupNormBackward(
    const std::array<int, 4>& dims,
    const T* dY,
    const T* X,
    const T* mu,
    const T* rsig,
    const T* gamma,
    const T* ds,
    const T* db,
    T* dX,
    T* dgamma,
    T* dbeta) {
  constexpr int kGDim = kOrder == StorageOrder::NCHW ? 1 : 2;
  constexpr int kDDim = kOrder == StorageOrder::NCHW ? 2 : 3;
  const int size = dims[0] * dims[1] * dims[2] * dims[3];
  const int HxW = kOrder == StorageOrder::NCHW ? dims[3] : dims[1];
  const T denom = T(1) / static_cast<T>(dims[kDDim] * HxW);
  std::array<int, 4> index = {0, 0, 0, 0};
  for (int i = 0; i < size; ++i) {
    const int i_mu = index[0] * dims[kGDim] + index[kGDim];
    const int i_gamma = index[kGDim] * dims[kDDim] + index[kDDim];
    const T u =
        (db[i_mu] * mu[i_mu] - ds[i_mu]) * (X[i] - mu[i_mu]) * Cube(rsig[i_mu]);
    const T v = db[i_mu] * rsig[i_mu];
    dX[i] = gamma[i_gamma] * dY[i] * rsig[i_mu] + (u - v) * denom;
    dgamma[i_gamma] += dY[i] * (X[i] - mu[i_mu]) * rsig[i_mu];
    dbeta[i_gamma] += dY[i];
    math::internal::IncreaseIndexInDims(4, dims.data(), index.data());
  }
}

} // namespace

template <typename T, class Context>
bool GroupNormOp<T, Context>::RunOnDeviceImpl(
    const int N,
    const int G,
    const int D,
    const int HxW,
    const T* X_data,
    const T* gamma_data,
    const T* beta_data,
    T* Y_data,
    T* mu_data,
    T* rsig_data) {
  const std::array<int, 4> dims = order_ == StorageOrder::NCHW
      ? std::array<int, 4>{N, G, D, HxW}
      : std::array<int, 4>{N, HxW, G, D};
  const std::array<int, 2> axes = order_ == StorageOrder::NCHW
      ? std::array<int, 2>{2, 3}
      : std::array<int, 2>{1, 3};

  // Computes mean and variance.
  math::Moments<T, Context>(
      4, dims.data(), 2, axes.data(), X_data, mu_data, rsig_data, &context_);

  // Uses rsqrt to computes 1 / std which is much faster than computes std.
  EigenArrayMap<T> rsig_array(rsig_data, G, N);
  rsig_array = (rsig_array += epsilon_).rsqrt();

  // Computes Y = gamma * (X - mu) * rsig + beta.
  if (order_ == StorageOrder::NCHW) {
    GroupNormForward<T, StorageOrder::NCHW>(
        dims, X_data, mu_data, rsig_data, gamma_data, beta_data, Y_data);
  } else {
    GroupNormForward<T, StorageOrder::NHWC>(
        dims, X_data, mu_data, rsig_data, gamma_data, beta_data, Y_data);
  }
  return true;
}

// Math:
// let: s = gamma * rsig
// let: b = beta - mu * gamma * rsig
// then: Y = s * X + b
template <typename T, class Context>
bool GroupNormGradientOp<T, Context>::RunOnDeviceImpl(
    const int N,
    const int G,
    const int D,
    const int HxW,
    const T* dY_data,
    const T* X_data,
    const T* mu_data,
    const T* rsig_data,
    const T* gamma_data,
    T* dX_data,
    T* dgamma_data,
    T* dbeta_data) {
  const std::array<int, 4> dims = order_ == StorageOrder::NCHW
      ? std::array<int, 4>{N, G, D, HxW}
      : std::array<int, 4>{N, HxW, G, D};

  // Computes dL/ds and dL/db.
  // dL/ds = Sum(dL/dY * gamma * X)
  // dL/db = Sum(dL/dY * gamma)
  const int C = G * D;
  ds_.Resize(N, G);
  db_.Resize(N, G);
  T* ds_data = ds_.template mutable_data<T>();
  T* db_data = db_.template mutable_data<T>();
  math::Set<T, Context>(N * G, T(0), ds_data, &context_);
  math::Set<T, Context>(N * G, T(0), db_data, &context_);
  if (order_ == StorageOrder::NCHW) {
    ComputeInternalGradients<T, StorageOrder::NCHW>(
        dims, dY_data, X_data, gamma_data, ds_data, db_data);
  } else {
    ComputeInternalGradients<T, StorageOrder::NHWC>(
        dims, dY_data, X_data, gamma_data, ds_data, db_data);
  }

  // Computes dL/dX, dL/dgamma and dL/dbeta.
  math::Set<T, Context>(C, T(0), dgamma_data, &context_);
  math::Set<T, Context>(C, T(0), dbeta_data, &context_);
  if (order_ == StorageOrder::NCHW) {
    GroupNormBackward<T, StorageOrder::NCHW>(
        dims,
        dY_data,
        X_data,
        mu_data,
        rsig_data,
        gamma_data,
        ds_data,
        db_data,
        dX_data,
        dgamma_data,
        dbeta_data);
  } else {
    GroupNormBackward<T, StorageOrder::NHWC>(
        dims,
        dY_data,
        X_data,
        mu_data,
        rsig_data,
        gamma_data,
        ds_data,
        db_data,
        dX_data,
        dgamma_data,
        dbeta_data);
  }
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
    .NumOutputs(3)
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
OPERATOR_SCHEMA(GroupNormGradient).NumInputs(6).NumOutputs(3);

class GetGroupNormGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "GroupNormGradient",
        "",
        vector<string>{GO(0), I(0), I(1), I(2), O(1), O(2)},
        vector<string>{GI(0), GI(1), GI(2)});
  }
};

REGISTER_GRADIENT(GroupNorm, GetGroupNormGradient);

} // namespace caffe2
