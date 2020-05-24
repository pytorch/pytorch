#include <ATen/native/group_norm.h>

#include <algorithm>
#include <array>
#include <numeric>

#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/cpu/vec256/vec256.h>

namespace at {
namespace native {

namespace {

template <typename T>
void GroupNormKernelImplInternal(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    T eps,
    Tensor* Y,
    Tensor* mean,
    Tensor* rstd) {
  DCHECK_EQ(X.numel(), N * C * HxW);
  DCHECK(!gamma.defined() || gamma.numel() == C);
  DCHECK(!beta.defined() || beta.numel() == C);
  const int64_t G = group;
  const int64_t D = C / G;
  const T* X_data = X.data_ptr<T>();
  const T* gamma_data = gamma.defined() ? gamma.data_ptr<T>() : nullptr;
  const T* beta_data = beta.defined() ? beta.data_ptr<T>() : nullptr;
  T* Y_data = Y->data_ptr<T>();
  T* mean_data = mean->data_ptr<T>();
  T* rstd_data = rstd->data_ptr<T>();
  const T s = T(1) / static_cast<T>(D * HxW);
  const bool gamma_null = gamma_data == nullptr;
  const bool beta_null = beta_data == nullptr;

  at::parallel_for(0, N * G, 1, [&](int64_t start, int64_t end) {
    constexpr int64_t K = vec256::Vec256<T>::size();
    const int64_t inner_size = D * HxW / K * K;
    std::array<T, K> mean_arr;
    std::array<T, K> rstd_arr;
    for (int64_t i = start; i < end; ++i) {
      const T* X_ptr = X_data + i * D * HxW;
      vec256::Vec256<T> mean_vec(0);
      vec256::Vec256<T> rstd_vec(0);
      for (int64_t k = 0; k < inner_size; k += K) {
        const vec256::Vec256<T> x_vec = vec256::Vec256<T>::loadu(X_ptr + k);
        mean_vec = mean_vec + x_vec;
        rstd_vec = rstd_vec + x_vec * x_vec;
      }
      mean_vec.store(mean_arr.data());
      rstd_vec.store(rstd_arr.data());
      T mean_val = std::accumulate(mean_arr.cbegin(), mean_arr.cend(), T(0));
      T rstd_val = std::accumulate(rstd_arr.cbegin(), rstd_arr.cend(), T(0));
      for (int64_t k = inner_size; k < D * HxW; ++k) {
        mean_val += X_ptr[k];
        rstd_val += X_ptr[k] * X_ptr[k];
      }
      mean_val *= s;
      rstd_val = std::max(rstd_val * s - mean_val * mean_val, T(0));
      rstd_val = T(1) / std::sqrt(rstd_val + eps);

      const int64_t n = i / G;
      const int64_t g = i % G;
      for (int64_t k = 0; k < D; ++k) {
        const int64_t c = g * D + k;
        const T scale = rstd_val * (gamma_null ? T(1) : gamma_data[c]);
        const T bias = -scale * mean_val + (beta_null ? T(0) : beta_data[c]);
        X_ptr = X_data + (n * C + c) * HxW;
        T* Y_ptr = Y_data + (n * C + c) * HxW;
        for (int64_t x = 0; x < HxW; ++x) {
          Y_ptr[x] = scale * X_ptr[x] + bias;
        }
      }
      mean_data[i] = mean_val;
      rstd_data[i] = rstd_val;
    }
  });
}

void GroupNormKernelImpl(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    double eps,
    Tensor* Y,
    Tensor* mean,
    Tensor* rstd) {
  AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "GroupNormKernelImpl", [&]() {
    GroupNormKernelImplInternal<scalar_t>(
        X,
        gamma,
        beta,
        N,
        C,
        HxW,
        group,
        static_cast<scalar_t>(eps),
        Y,
        mean,
        rstd);
  });
}

template <typename T>
void GroupNormBackwardKernelImplInternal(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    Tensor* dX,
    Tensor* dgamma,
    Tensor* dbeta) {
  DCHECK_EQ(dY.numel(), N * C * HxW);
  DCHECK_EQ(X.numel(), N * C * HxW);
  DCHECK_EQ(mean.numel(), N * group);
  DCHECK_EQ(rstd.numel(), N * group);
  DCHECK(!gamma.defined() || gamma.numel() == C);
  const int64_t G = group;
  const int64_t D = C / G;
  const T* dY_data = dY.data_ptr<T>();
  const T* X_data = X.data_ptr<T>();
  const T* mean_data = mean.data_ptr<T>();
  const T* rstd_data = rstd.data_ptr<T>();
  const T* gamma_data = gamma.defined() ? gamma.data_ptr<T>() : nullptr;
  T* dX_data = dX->defined() ? dX->data_ptr<T>() : nullptr;
  T* dgamma_data = dgamma->defined() ? dgamma->data_ptr<T>() : nullptr;
  T* dbeta_data = dbeta->defined() ? dbeta->data_ptr<T>() : nullptr;
  if (dgamma_data != nullptr) {
    std::memset(dgamma_data, 0, C * sizeof(T));
  }
  if (dbeta_data != nullptr) {
    std::memset(dbeta_data, 0, C * sizeof(T));
  }
  const T s = T(1) / static_cast<T>(D * HxW);
  const bool gamma_null = gamma_data == nullptr;
  Tensor ds = at::empty({G, D}, X.options());
  Tensor db = at::empty({G, D}, X.options());
  T* ds_data = ds.data_ptr<T>();
  T* db_data = db.data_ptr<T>();

  constexpr int64_t K = vec256::Vec256<T>::size();
  const int64_t inner_size = HxW / K * K;
  std::array<T, K> ds_arr;
  std::array<T, K> db_arr;
  for (int64_t i = 0; i < N; ++i) {
    for (int64_t j = 0; j < C; ++j) {
      const T* dY_ptr = dY_data + (i * C + j) * HxW;
      const T* X_ptr = X_data + (i * C + j) * HxW;
      vec256::Vec256<T> ds_vec(0);
      vec256::Vec256<T> db_vec(0);
      for (int64_t k = 0; k < inner_size; k += K) {
        const vec256::Vec256<T> dy_vec = vec256::Vec256<T>::loadu(dY_ptr + k);
        const vec256::Vec256<T> x_vec = vec256::Vec256<T>::loadu(X_ptr + k);
        ds_vec = ds_vec + dy_vec * x_vec;
        db_vec = db_vec + dy_vec;
      }
      ds_vec.store(ds_arr.data());
      db_vec.store(db_arr.data());
      T ds_val = std::accumulate(ds_arr.cbegin(), ds_arr.cend(), T(0));
      T db_val = std::accumulate(db_arr.cbegin(), db_arr.cend(), T(0));
      for (int64_t k = inner_size; k < HxW; ++k) {
        ds_val += dY_ptr[k] * X_ptr[k];
        db_val += dY_ptr[k];
      }
      ds_data[j] = ds_val;
      db_data[j] = db_val;
    }
    if (dX_data != nullptr) {
      const int64_t d = D / K * K;
      for (int64_t j = 0; j < G; ++j) {
        const T* ds_ptr = ds_data + j * D;
        const T* db_ptr = db_data + j * D;
        vec256::Vec256<T> ds_vec(0);
        vec256::Vec256<T> db_vec(0);
        for (int64_t k = 0; k < d; k += K) {
          const vec256::Vec256<T> gamma_vec = gamma_null
              ? vec256::Vec256<T>(1)
              : vec256::Vec256<T>::loadu(gamma_data + j * D + k);
          ds_vec = ds_vec + vec256::Vec256<T>::loadu(ds_ptr + k) * gamma_vec;
          db_vec = db_vec + vec256::Vec256<T>::loadu(db_ptr + k) * gamma_vec;
        }
        ds_vec.store(ds_arr.data());
        db_vec.store(db_arr.data());
        T ds_val = std::accumulate(ds_arr.cbegin(), ds_arr.cend(), T(0));
        T db_val = std::accumulate(db_arr.cbegin(), db_arr.cend(), T(0));
        for (int64_t k = d; k < D; ++k) {
          const T gamma_v = gamma_null ? T(1) : gamma_data[j * D + k];
          ds_val += ds_ptr[k] * gamma_v;
          db_val += db_ptr[k] * gamma_v;
        }
        const int64_t ng = i * G + j;
        const T c2 = (db_val * mean_data[ng] - ds_val) * rstd_data[ng] *
            rstd_data[ng] * rstd_data[ng] * s;
        const T c3 = -c2 * mean_data[ng] - db_val * rstd_data[ng] * s;
        for (int64_t k = 0; k < D; ++k) {
          const int64_t c = j * D + k;
          const T* dY_ptr = dY_data + (i * C + c) * HxW;
          const T* X_ptr = X_data + (i * C + c) * HxW;
          T* dX_ptr = dX_data + (i * C + c) * HxW;
          const T c1 = rstd_data[ng] * (gamma_null ? T(1) : gamma_data[c]);
          for (int64_t x = 0; x < HxW; ++x) {
            dX_ptr[x] = c1 * dY_ptr[x] + c2 * X_ptr[x] + c3;
          }
        }
      }
    }
    if (dgamma_data != nullptr) {
      for (int64_t j = 0; j < G; ++j) {
        const int64_t ng = i * G + j;
        for (int64_t k = 0; k < D; ++k) {
          const int64_t c = j * D + k;
          dgamma_data[c] +=
              (ds_data[c] - db_data[c] * mean_data[ng]) * rstd_data[ng];
        }
      }
    }
    if (dbeta_data != nullptr) {
      for (int64_t j = 0; j < C; ++j) {
        dbeta_data[j] += db_data[j];
      }
    }
  }
}

void GroupNormBackwardKernelImpl(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    Tensor* dX,
    Tensor* dgamma,
    Tensor* dbeta) {
  AT_DISPATCH_FLOATING_TYPES(
      X.scalar_type(), "GroupNormBackwardKernelImpl", [&]() {
        GroupNormBackwardKernelImplInternal<scalar_t>(
            dY, X, mean, rstd, gamma, N, C, HxW, group, dX, dgamma, dbeta);
      });
}

} // namespace

REGISTER_DISPATCH(GroupNormKernel, &GroupNormKernelImpl);
REGISTER_DISPATCH(GroupNormBackwardKernel, &GroupNormBackwardKernelImpl);

} // namespace native
} // namespace at
