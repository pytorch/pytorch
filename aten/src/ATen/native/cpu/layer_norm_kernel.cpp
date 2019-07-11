#include <ATen/native/cpu/layer_norm_kernel.h>

#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Dispatch.h>

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
    T eps,
    Tensor* Y,
    Tensor* mean,
    Tensor* rstd) {
  DCHECK_EQ(X.numel(), M * N);
  DCHECK(!gamma.defined() || gamma.numel() == N);
  DCHECK(!beta.defined() || beta.numel() == N);
  const T* X_data = X.data<T>();
  const T* gamma_data = gamma.defined() ? gamma.data<T>() : nullptr;
  const T* beta_data = beta.defined() ? beta.data<T>() : nullptr;
  T* Y_data = Y->data<T>();
  T* mean_data = mean->data<T>();
  T* rstd_data = rstd->data<T>();
  const T c = T(1) / static_cast<T>(N);
  const bool gamma_null = gamma_data == nullptr;
  const bool beta_null = beta_data == nullptr;
  for (int64_t i = 0; i < M; ++i) {
    const T* X_ptr = X_data + i * N;
    T* Y_ptr = Y_data + i * N;
    T mean_val = T(0);
    T rstd_val = T(0);
    for (int64_t j = 0; j < N; ++j) {
      mean_val += X_ptr[j];
      rstd_val += X_ptr[j] * X_ptr[j];
    }
    mean_val *= c;
    rstd_val = T(1) / std::sqrt(rstd_val * c - mean_val * mean_val + eps);
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

} // namespace

REGISTER_DISPATCH(LayerNormKernel, &LayerNormKernelImpl);

} // namespace native
} // namespace at
