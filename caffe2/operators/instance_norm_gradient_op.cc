#include "caffe2/operators/instance_norm_op.h"

#include <string>
#include <vector>

#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

namespace {

template <typename T>
void ComputeInternalGradientsNHWC(
    const int64_t N,
    const int64_t C,
    const int64_t HxW,
    const T* dY,
    const T* X,
    T* ds,
    T* db) {
  EigenArrayMap<T> ds_arr(ds, C, N);
  EigenArrayMap<T> db_arr(db, C, N);
  for (int64_t i = 0; i < N; ++i) {
    ConstEigenArrayMap<T> dY_arr(dY + i * C * HxW, C, HxW);
    ConstEigenArrayMap<T> X_arr(X + i * C * HxW, C, HxW);
    ds_arr.col(i) = dY_arr.col(0) * X_arr.col(0);
    db_arr.col(i) = dY_arr.col(0);
    for (int j = 1; j < HxW; ++j) {
      ds_arr.col(i) += dY_arr.col(j) * X_arr.col(j);
      db_arr.col(i) += dY_arr.col(j);
    }
  }
}

template <typename T>
void InstanceNormBackwardNCHW(
    const int64_t N,
    const int64_t C,
    const int64_t HxW,
    const T* dY,
    const T* X,
    const T* mean,
    const T* rstd,
    const T* gamma,
    T* dX,
    T* ds,
    T* db) {
  const T scale = T(1) / static_cast<T>(HxW);
  ConstEigenArrayMap<T> dY_arr(dY, HxW, N * C);
  ConstEigenArrayMap<T> X_arr(X, HxW, N * C);
  for (int64_t i = 0; i < N * C; ++i) {
    const T ds_sum = (dY_arr.col(i) * X_arr.col(i)).sum();
    const T db_sum = dY_arr.col(i).sum();
    const int64_t c = i % C;
    const T c1 = rstd[i] * gamma[c];
    T c2 = ds_sum * gamma[c];
    T c3 = db_sum * gamma[c];
    c2 = (c3 * mean[i] - c2) * rstd[i] * rstd[i] * rstd[i] * scale;
    c3 = -c2 * mean[i] - c3 * rstd[i] * scale;
    for (int64_t j = 0; j < HxW; ++j) {
      const int64_t index = i * HxW + j;
      dX[index] = c1 * dY[index] + c2 * X[index] + c3;
    }
    ds[i] = ds_sum;
    db[i] = db_sum;
  }
}

template <typename T>
void InstanceNormBackwardNHWC(
    const int64_t N,
    const int64_t C,
    const int64_t HxW,
    const T* dY,
    const T* X,
    const T* ds,
    const T* db,
    const T* mean,
    const T* rstd,
    const T* gamma,
    T* dX,
    T* c1,
    T* c2,
    T* c3) {
  const T scale = T(1) / static_cast<T>(HxW);
  ConstEigenArrayMap<T> ds_arr(ds, C, N);
  ConstEigenArrayMap<T> db_arr(db, C, N);
  ConstEigenArrayMap<T> mean_arr(mean, C, N);
  ConstEigenArrayMap<T> rstd_arr(rstd, C, N);
  ConstEigenVectorArrayMap<T> gamma_arr(gamma, C);
  EigenArrayMap<T> c1_arr(c1, C, N);
  EigenArrayMap<T> c2_arr(c2, C, N);
  EigenArrayMap<T> c3_arr(c3, C, N);
  c1_arr = rstd_arr.colwise() * gamma_arr;
  c2_arr = ds_arr.colwise() * gamma_arr;
  c3_arr = db_arr.colwise() * gamma_arr;
  c2_arr = (c3_arr * mean_arr - c2_arr) * rstd_arr.cube() * scale;
  c3_arr = -c2_arr * mean_arr - c3_arr * rstd_arr * scale;
  for (int64_t i = 0; i < N; ++i) {
    ConstEigenArrayMap<T> dY_arr(dY + i * HxW * C, C, HxW);
    ConstEigenArrayMap<T> X_arr(X + i * HxW * C, C, HxW);
    EigenArrayMap<T> dX_arr(dX + i * HxW * C, C, HxW);
    dX_arr =
        (dY_arr.colwise() * c1_arr.col(i) + X_arr.colwise() * c2_arr.col(i))
            .colwise() +
        c3_arr.col(i);
  }
}

template <typename T>
void GammaBetaBackward(
    const int64_t N,
    const int64_t C,
    const T* ds,
    const T* db,
    const T* mean,
    const T* rstd,
    T* dgamma,
    T* dbeta) {
  ConstEigenArrayMap<T> ds_arr(ds, C, N);
  ConstEigenArrayMap<T> db_arr(db, C, N);
  ConstEigenArrayMap<T> mean_arr(mean, C, N);
  ConstEigenArrayMap<T> rstd_arr(rstd, C, N);
  EigenVectorArrayMap<T> dgamma_arr(dgamma, C);
  EigenVectorArrayMap<T> dbeta_arr(dbeta, C);
  dgamma_arr =
      (ds_arr.col(0) - db_arr.col(0) * mean_arr.col(0)) * rstd_arr.col(0);
  dbeta_arr = db_arr.col(0);
  for (int64_t i = 1; i < N; ++i) {
    dgamma_arr +=
        (ds_arr.col(i) - db_arr.col(i) * mean_arr.col(i)) * rstd_arr.col(i);
    dbeta_arr += db_arr.col(i);
  }
}

} // namespace

template <>
void InstanceNormGradientOp<float, CPUContext>::ComputeMoments(
    const int64_t N,
    const int64_t C,
    const int64_t HxW,
    const float* X,
    float* mean,
    float* rstd) {
  if (order_ == StorageOrder::NCHW) {
    const std::array<int, 2> X_dims = {static_cast<int>(N * C),
                                       static_cast<int>(HxW)};
    const std::array<int, 2> Y_dims = {static_cast<int>(N * C), 1};
    math::Moments<float, CPUContext>(
        2, X_dims.data(), Y_dims.data(), X, mean, rstd, &context_);
    math::InvStd<float, CPUContext>(N * C, epsilon_, rstd, rstd, &context_);
  } else {
    const float c = 1.0f / static_cast<float>(HxW);
    EigenArrayMap<float> mean_arr(mean, C, N);
    EigenArrayMap<float> rstd_arr(rstd, C, N);
    for (int64_t i = 0; i < N; ++i) {
      ConstEigenArrayMap<float> X_arr(X + i * HxW * C, C, HxW);
      mean_arr.col(i) = X_arr.col(0);
      rstd_arr.col(i) = X_arr.col(0).square();
      for (int64_t j = 1; j < HxW; ++j) {
        mean_arr.col(i) += X_arr.col(j);
        rstd_arr.col(i) += X_arr.col(j).square();
      }
    }
    mean_arr *= c;
    rstd_arr =
        ((rstd_arr * c - mean_arr.square()).max(0.0f) + epsilon_).rsqrt();
  }
}

template <>
bool InstanceNormGradientOp<float, CPUContext>::RunOnDeviceWithOrderNCHW(
    const int64_t N,
    const int64_t C,
    const int64_t HxW,
    const float* dY,
    const float* X,
    const float* mean,
    const float* rstd,
    const float* gamma,
    float* dX,
    float* dgamma,
    float* dbeta) {
  ReinitializeTensor(&ds_, {N, C}, at::dtype<float>().device(CPU));
  ReinitializeTensor(&db_, {N, C}, at::dtype<float>().device(CPU));
  float* ds_data = ds_.mutable_data<float>();
  float* db_data = db_.mutable_data<float>();
  InstanceNormBackwardNCHW<float>(
      N, C, HxW, dY, X, mean, rstd, gamma, dX, ds_data, db_data);
  GammaBetaBackward<float>(N, C, ds_data, db_data, mean, rstd, dgamma, dbeta);
  return true;
}

template <>
bool InstanceNormGradientOp<float, CPUContext>::RunOnDeviceWithOrderNHWC(
    const int64_t N,
    const int64_t C,
    const int64_t HxW,
    const float* dY,
    const float* X,
    const float* mean,
    const float* rstd,
    const float* gamma,
    float* dX,
    float* dgamma,
    float* dbeta) {
  ReinitializeTensor(&ds_, {N, C}, at::dtype<float>().device(CPU));
  ReinitializeTensor(&db_, {N, C}, at::dtype<float>().device(CPU));
  float* ds_data = ds_.mutable_data<float>();
  float* db_data = db_.mutable_data<float>();
  ComputeInternalGradientsNHWC<float>(N, C, HxW, dY, X, ds_data, db_data);
  ReinitializeTensor(&c1_, {N, C}, at::dtype<float>().device(CPU));
  ReinitializeTensor(&c2_, {N, C}, at::dtype<float>().device(CPU));
  ReinitializeTensor(&c3_, {N, C}, at::dtype<float>().device(CPU));
  float* c1_data = c1_.mutable_data<float>();
  float* c2_data = c2_.mutable_data<float>();
  float* c3_data = c3_.mutable_data<float>();
  InstanceNormBackwardNHWC<float>(
      N,
      C,
      HxW,
      dY,
      X,
      ds_data,
      db_data,
      mean,
      rstd,
      gamma,
      dX,
      c1_data,
      c2_data,
      c3_data);
  GammaBetaBackward<float>(N, C, ds_data, db_data, mean, rstd, dgamma, dbeta);
  return true;
}

namespace {

class GetInstanceNormGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    std::vector<std::string> inputs = {I(0), I(1), I(2), GO(0)};
    if (def_.output_size() >= 2) {
      inputs.push_back(O(1));
    }
    if (def_.output_size() >= 3) {
      inputs.push_back(O(2));
    }
    return SingleGradientDef(
        "InstanceNormGradient",
        "",
        inputs,
        std::vector<std::string>({GI(0), GI(1), GI(2)}));
  }
};

} // namespace

REGISTER_CPU_OPERATOR(
    InstanceNormGradient,
    InstanceNormGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(InstanceNormGradient).NumInputs(4, 6).NumOutputs(3);

REGISTER_GRADIENT(InstanceNorm, GetInstanceNormGradient);

} // namespace caffe2
