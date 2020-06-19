#include <ATen/native/layer_norm.h>

#include <algorithm>
#include <cmath>
#include <tuple>
#include <utility>

#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/cpu/vec256/functional.h>
#include <ATen/cpu/vec256/vec256.h>

namespace at {
namespace native {

namespace {

template <typename T>
void AddMoments(int64_t m0_add, T m1_add, T m2_add, int64_t* m0, T* m1, T* m2) {
  const int64_t n = *m0 + m0_add;
  const T c1 = n == 0 ? 0 : static_cast<T>(*m0) / static_cast<T>(n);
  const T c2 = n == 0 ? 0 : T(1) - c1;
  const T delta = m1_add - *m1;
  *m0 = n;
  *m1 = c1 * (*m1) + c2 * m1_add;
  *m2 += m2_add + delta * delta * c1 * c2 * static_cast<T>(n);
}

template <typename T>
std::pair<T, T> WelfordMoments(int64_t N, const T* X) {
  constexpr int64_t K = vec256::Vec256<T>::size();
  const int64_t n = N / K;

  vec256::Vec256<T> m1_vec(0);
  vec256::Vec256<T> m2_vec(0);
  for (int64_t i = 0; i < n; ++i) {
    const vec256::Vec256<T> x_vec = vec256::Vec256<T>::loadu(X + i * K);
    const vec256::Vec256<T> delta_vec = x_vec - m1_vec;
    m1_vec = m1_vec + delta_vec / vec256::Vec256<T>(static_cast<T>(i + 1));
    m2_vec = m2_vec + delta_vec * (x_vec - m1_vec);
  }
  std::array<T, K> m1_arr;
  std::array<T, K> m2_arr;
  m1_vec.store(m1_arr.data());
  m2_vec.store(m2_arr.data());

  int64_t m0 = 0;
  T m1 = 0;
  T m2 = 0;
  for (int64_t i = 0; i < K; ++i) {
    AddMoments(n, m1_arr[i], m2_arr[i], &m0, &m1, &m2);
  }
  for (int64_t i = n * K; i < N; ++i) {
    const T delta = X[i] - m1;
    m1 += delta / static_cast<T>(i + 1);
    m2 += delta * (X[i] - m1);
  }

  return std::make_pair(m1, m2 / static_cast<T>(N));
}

template <typename T>
void LayerNormKernelImplInternal(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t M,
    int64_t N,
    T eps,
    Tensor* Y,
    Tensor* mean,
    Tensor* rstd) {
  TORCH_CHECK(X.numel() == M * N);
  TORCH_CHECK(!gamma.defined() || gamma.numel() == N);
  TORCH_CHECK(!beta.defined() || beta.numel() == N);
  const T* X_data = X.data_ptr<T>();
  const T* gamma_data = gamma.defined() ? gamma.data_ptr<T>() : nullptr;
  const T* beta_data = beta.defined() ? beta.data_ptr<T>() : nullptr;
  T* Y_data = Y->data_ptr<T>();
  T* mean_data = mean->data_ptr<T>();
  T* rstd_data = rstd->data_ptr<T>();
  const bool gamma_null = gamma_data == nullptr;
  const bool beta_null = beta_data == nullptr;
  at::parallel_for(0, M, 1, [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; ++i) {
      const T* X_ptr = X_data + i * N;
      T* Y_ptr = Y_data + i * N;
      T mean_val;
      T rstd_val;
      std::tie(mean_val, rstd_val) = WelfordMoments(N, X_ptr);
      rstd_val = T(1) / std::sqrt(std::max(rstd_val, T(0)) + eps);
      const T scale = rstd_val;
      const T bias = -rstd_val * mean_val;
      for (int64_t j = 0; j < N; ++j) {
        const T gamma_v = gamma_null ? T(1) : gamma_data[j];
        const T beta_v = beta_null ? T(0) : beta_data[j];
        Y_ptr[j] = (X_ptr[j] * scale + bias) * gamma_v + beta_v;
      }
      mean_data[i] = mean_val;
      rstd_data[i] = rstd_val;
    }
  });
}

void LayerNormKernelImpl(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t M,
    int64_t N,
    double eps,
    Tensor* Y,
    Tensor* mean,
    Tensor* rstd) {
  AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "LayerNormKernelImpl", [&]() {
    LayerNormKernelImplInternal<scalar_t>(
        X, gamma, beta, M, N, static_cast<scalar_t>(eps), Y, mean, rstd);
  });
}

template <typename T>
void LayerNormBackwardKernelImplInternal(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    Tensor* dX,
    Tensor* dgamma,
    Tensor* dbeta) {
  TORCH_CHECK(dY.numel() == M * N);
  TORCH_CHECK(X.numel() == M * N);
  TORCH_CHECK(mean.numel() == M);
  TORCH_CHECK(rstd.numel() == M);
  TORCH_CHECK(!gamma.defined() || gamma.numel() == N);
  const T* dY_data = dY.data_ptr<T>();
  const T* X_data = X.data_ptr<T>();
  const T* mean_data = mean.data_ptr<T>();
  const T* rstd_data = rstd.data_ptr<T>();
  const T* gamma_data = gamma.defined() ? gamma.data_ptr<T>() : nullptr;
  T* dX_data = dX->defined() ? dX->data_ptr<T>() : nullptr;
  T* dgamma_data = dgamma->defined() ? dgamma->data_ptr<T>() : nullptr;
  T* dbeta_data = dbeta->defined() ? dbeta->data_ptr<T>() : nullptr;
  const T scale = T(1) / static_cast<T>(N);
  const bool gamma_null = gamma_data == nullptr;

  if (dX_data != nullptr) {
    at::parallel_for(0, M, 1, [&](int64_t start, int64_t end) {
      constexpr int64_t K = vec256::Vec256<T>::size();
      const int64_t n = N / K * K;
      std::array<T, K> ds_arr;
      std::array<T, K> db_arr;

      for (int64_t i = start; i < end; ++i) {
        const T* dY_ptr = dY_data + i * N;
        const T* X_ptr = X_data + i * N;
        T* dX_ptr = dX_data + i * N;

        vec256::Vec256<T> ds_vec(0);
        vec256::Vec256<T> db_vec(0);
        for (int64_t j = 0; j < n; j += K) {
          const vec256::Vec256<T> dy_vec = vec256::Vec256<T>::loadu(dY_ptr + j);
          const vec256::Vec256<T> x_vec = vec256::Vec256<T>::loadu(X_ptr + j);
          const vec256::Vec256<T> gamma_vec = gamma_null
              ? vec256::Vec256<T>(1)
              : vec256::Vec256<T>::loadu(gamma_data + j);
          ds_vec = ds_vec + dy_vec * x_vec * gamma_vec;
          db_vec = db_vec + dy_vec * gamma_vec;
        }
        ds_vec.store(ds_arr.data());
        db_vec.store(db_arr.data());
        T ds = std::accumulate(ds_arr.cbegin(), ds_arr.cend(), T(0));
        T db = std::accumulate(db_arr.cbegin(), db_arr.cend(), T(0));
        for (int64_t j = n; j < N; ++j) {
          const T gamma_v = gamma_null ? T(1) : gamma_data[j];
          ds += dY_ptr[j] * X_ptr[j] * gamma_v;
          db += dY_ptr[j] * gamma_v;
        }

        const T a = rstd_data[i];
        const T b = (db * mean_data[i] - ds) * a * a * a * scale;
        const T c = -b * mean_data[i] - db * a * scale;
        for (int64_t j = 0; j < N; ++j) {
          const T gamma_v = gamma_null ? T(1) : gamma_data[j];
          dX_ptr[j] = a * dY_ptr[j] * gamma_v + b * X_ptr[j] + c;
        }
      }
    });
  }
  if (dgamma_data != nullptr) {
    constexpr int64_t K = vec256::Vec256<T>::size();
    at::parallel_for(0, N, K, [&](int64_t start, int64_t end) {
      std::memset(dgamma_data + start, 0, (end - start) * sizeof(T));
      for (int64_t i = 0; i < M; ++i) {
        const T* dY_ptr = dY_data + i * N;
        const T* X_ptr = X_data + i * N;
        const T a = rstd_data[i];
        const T b = -a * mean_data[i];
        for (int64_t j = start; j < end; ++j) {
          dgamma_data[j] += dY_ptr[j] * (a * X_ptr[j] + b);
        }
      }
    });
  }
  if (dbeta_data != nullptr) {
    constexpr int64_t K = vec256::Vec256<T>::size();
    at::parallel_for(0, N, K, [&](int64_t start, int64_t end) {
      std::memset(dbeta_data + start, 0, (end - start) * sizeof(T));
      for (int64_t i = 0; i < M; ++i) {
        const T* dY_ptr = dY_data + i * N;
        for (int64_t j = start; j < end; ++j) {
          dbeta_data[j] += dY_ptr[j];
        }
      }
    });
  }
}

void LayerNormBackwardKernelImpl(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    Tensor* dX,
    Tensor* dgamma,
    Tensor* dbeta) {
  AT_DISPATCH_FLOATING_TYPES(
      X.scalar_type(), "LayerNormBackwardKernelImpl", [&]() {
        LayerNormBackwardKernelImplInternal<scalar_t>(
            dY, X, mean, rstd, gamma, M, N, dX, dgamma, dbeta);
      });
}

} // namespace

REGISTER_DISPATCH(LayerNormKernel, &LayerNormKernelImpl);
REGISTER_DISPATCH(LayerNormBackwardKernel, &LayerNormBackwardKernelImpl);

} // namespace native
} // namespace at
