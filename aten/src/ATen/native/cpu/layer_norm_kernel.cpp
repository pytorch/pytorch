#include <ATen/native/layer_norm.h>

#include <cmath>
#include <ATen/AccumulateType.h>
#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/cpu/vec256/functional.h>
#include <ATen/cpu/vec256/vec256.h>

namespace at {
namespace native {

namespace {

template <typename T>
void LayerNormKernelImplInternal(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t M,
    int64_t N,
    double eps,
    Tensor* Y,
    Tensor* mean,
    Tensor* rstd) {
  using Vec = vec256::Vec256<T>;
  DCHECK_EQ(X.numel(), M * N);
  DCHECK(!gamma.defined() || gamma.numel() == N);
  DCHECK(!beta.defined() || beta.numel() == N);
  using ACC_T = acc_type<T, false>;
  T* X_data = X.data_ptr<T>();
  const ACC_T* gamma_data = gamma.defined() ? gamma.data_ptr<ACC_T>() : nullptr;
  const ACC_T* beta_data = beta.defined() ? beta.data_ptr<ACC_T>() : nullptr;
  T* Y_data = Y->data_ptr<T>();
  ACC_T* mean_data = mean->data_ptr<ACC_T>();
  ACC_T* rstd_data = rstd->data_ptr<ACC_T>();
  const ACC_T c = ACC_T(1) / static_cast<ACC_T>(N);
  const bool gamma_null = gamma_data == nullptr;
  const bool beta_null = beta_data == nullptr;
  at::parallel_for(0, M, 1, [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; ++i) {
      T* X_ptr = X_data + i * N;
      T* Y_ptr = Y_data + i * N;
      ACC_T mean_val = vec256::reduce_all<T>(
          [](Vec& x, Vec& y) { return x + y; },
          X_ptr,
          N);
      ACC_T rstd_val = vec256::map_reduce_all<T>(
          [](Vec x) { return x * x; },
          [](Vec x, Vec y) { return x + y; },
          X_ptr,
          N);
      mean_val *= c;
      rstd_val = std::max(rstd_val * c - mean_val * mean_val, ACC_T(0));
      rstd_val = ACC_T(1) / std::sqrt(rstd_val + eps);
      const ACC_T scale = rstd_val;
      const ACC_T bias = -rstd_val * mean_val;
      for (int64_t j = 0; j < N; ++j) {
        const ACC_T gamma_v = gamma_null ? ACC_T(1) : gamma_data[j];
        const ACC_T beta_v = beta_null ? ACC_T(0) : beta_data[j];
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
  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, X.scalar_type(), "LayerNormKernelImpl", [&]() {
    LayerNormKernelImplInternal<scalar_t>(
        X, gamma, beta, M, N, eps, Y, mean, rstd);
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
  DCHECK_EQ(dY.numel(), M * N);
  DCHECK_EQ(X.numel(), M * N);
  DCHECK_EQ(mean.numel(), M);
  DCHECK_EQ(rstd.numel(), M);
  DCHECK(!gamma.defined() || gamma.numel() == N);
  using ACC_T = acc_type<T, false>;
  const T* dY_data = dY.template data_ptr<T>();
  const T* X_data = X.template data_ptr<T>();
  const ACC_T* mean_data = mean.template data_ptr<ACC_T>();
  const ACC_T* rstd_data = rstd.template data_ptr<ACC_T>();
  const ACC_T* gamma_data =
      gamma.defined() ? gamma.template data_ptr<ACC_T>() : nullptr;
  T* dX_data = dX->defined() ? dX->template data_ptr<T>() : nullptr;
  ACC_T* dgamma_data = dgamma->defined() ? dgamma->template data_ptr<ACC_T>() : nullptr;
  if (dgamma_data != nullptr) {
    std::memset(dgamma_data, 0, N * sizeof(ACC_T));
  }
  ACC_T* dbeta_data = dbeta->defined() ? dbeta->template data_ptr<ACC_T>() : nullptr;
  if (dbeta_data != nullptr) {
    std::memset(dbeta_data, 0, N * sizeof(ACC_T));
  }
  const ACC_T scale = ACC_T(1) / static_cast<ACC_T>(N);
  const bool gamma_null = gamma_data == nullptr;
  for (int64_t i = 0; i < M; ++i) {
    const T* dY_ptr = dY_data + i * N;
    const T* X_ptr = X_data + i * N;
    if (dX_data != nullptr) {
      T* dX_ptr = dX_data + i * N;
      ACC_T ds = 0;
      ACC_T db = 0;
      for (int64_t j = 0; j < N; ++j) {
        const ACC_T gamma_v = gamma_null ? ACC_T(1) : gamma_data[j];
        ds += ACC_T(dY_ptr[j]) * ACC_T(X_ptr[j]) * gamma_v;
        db += dY_ptr[j] * gamma_v;
      }
      const ACC_T a = rstd_data[i];
      const ACC_T b = (db * mean_data[i] - ds) * a * a * a * scale;
      const ACC_T c = -b * mean_data[i] - db * a * scale;
      for (int64_t j = 0; j < N; ++j) {
        const ACC_T gamma_v = gamma_null ? ACC_T(1) : gamma_data[j];
        dX_ptr[j] = a * dY_ptr[j] * gamma_v + b * X_ptr[j] + c;
      }
    }
    if (dgamma_data != nullptr) {
      const ACC_T a = rstd_data[i];
      const ACC_T b = -a * mean_data[i];
      for (int64_t j = 0; j < N; ++j) {
        dgamma_data[j] += dY_ptr[j] * (a * X_ptr[j] + b);
      }
    }
    if (dbeta_data != nullptr) {
      for (int64_t j = 0; j < N; ++j) {
        dbeta_data[j] += dY_ptr[j];
      }
    }
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
  AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, X.scalar_type(), "LayerNormBackwardKernelImpl", [&]() {
        LayerNormBackwardKernelImplInternal<scalar_t>(
            dY, X, mean, rstd, gamma, M, N, dX, dgamma, dbeta);
      });
}

} // namespace

REGISTER_DISPATCH(LayerNormKernel, &LayerNormKernelImpl);
REGISTER_DISPATCH(LayerNormBackwardKernel, &LayerNormBackwardKernelImpl);

} // namespace native
} // namespace at
