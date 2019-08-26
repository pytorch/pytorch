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
  const T* X_data = X.data_ptr<T>();
  const T* gamma_data = gamma.defined() ? gamma.data_ptr<T>() : nullptr;
  const T* beta_data = beta.defined() ? beta.data_ptr<T>() : nullptr;
  T* Y_data = Y->data_ptr<T>();
  T* mean_data = mean->data_ptr<T>();
  T* rstd_data = rstd->data_ptr<T>();
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
    rstd_val = std::max(rstd_val * c - mean_val * mean_val, T(0));
    rstd_val = T(1) / std::sqrt(rstd_val + eps);
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
  const T* dY_data = dY.template data_ptr<T>();
  const T* X_data = X.template data_ptr<T>();
  const T* mean_data = mean.template data_ptr<T>();
  const T* rstd_data = rstd.template data_ptr<T>();
  const T* gamma_data = gamma.defined() ? gamma.template data_ptr<T>() : nullptr;
  T* dX_data = dX->defined() ? dX->template data_ptr<T>() : nullptr;
  T* dgamma_data = dgamma->defined() ? dgamma->template data_ptr<T>() : nullptr;
  if (dgamma_data != nullptr) {
    std::memset(dgamma_data, 0, N * sizeof(T));
  }
  T* dbeta_data = dbeta->defined() ? dbeta->template data_ptr<T>() : nullptr;
  if (dbeta_data != nullptr) {
    std::memset(dbeta_data, 0, N * sizeof(T));
  }
  const T scale = T(1) / static_cast<T>(N);
  const bool gamma_null = gamma_data == nullptr;
  for (int64_t i = 0; i < M; ++i) {
    const T* dY_ptr = dY_data + i * N;
    const T* X_ptr = X_data + i * N;
    if (dX_data != nullptr) {
      T* dX_ptr = dX_data + i * N;
      T ds = 0;
      T db = 0;
      for (int64_t j = 0; j < N; ++j) {
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
    if (dgamma_data != nullptr) {
      const T a = rstd_data[i];
      const T b = -a * mean_data[i];
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
  AT_DISPATCH_FLOATING_TYPES(
      X.scalar_type(), "LayerNormBackwardKernelImpl", [&]() {
        LayerNormBackwardKernelImplInternal<scalar_t>(
            dY, X, mean, rstd, gamma, M, N, dX, dgamma, dbeta);
      });
}

template <typename T>
void LayerNormOutputDoubleBackward(
    int64_t N,
    T mean,
    T rstd,
    T s_ddx_x,
    T s_ddx,
    const T* ddX,
    const T* ddgamma,
    const T* ddbeta,
    const T* gamma,
    const T* X,
    T* ddY) {
  const bool ddX_null = ddX == nullptr;
  const bool ddgamma_null = ddgamma == nullptr;
  const bool ddbeta_null = ddbeta == nullptr;
  const bool gamma_null = gamma == nullptr;
  const T scale = T(1) / static_cast<T>(N);
  T b1 = -scale * rstd * rstd * rstd;
  T b2 = -mean * b1;
  T c1 = -mean * b1 * s_ddx;
  T c2 = -(mean * b2 + scale * rstd) * s_ddx;
  b1 *= s_ddx_x;
  b2 *= s_ddx_x;
  const T u = b1 + c1;
  const T v = b2 + c2;
  for (int64_t i = 0; i < N; ++i) {
    const T ddX_v = ddX_null ? T(0) : ddX[i];
    const T ddgamma_v = ddgamma_null ? T(0) : ddgamma[i];
    const T ddbeta_v = ddbeta_null ? T(0) : ddbeta[i];
    const T gamma_v = gamma_null ? T(1) : gamma[i];
    ddY[i] = (u * X[i] + v + rstd * ddX_v) * gamma_v +
        (X[i] - mean) * rstd * ddgamma_v + ddbeta_v;
  }
}

template <typename T>
void LayerNormInputDoubleBackward(
    int64_t N,
    T mean,
    T rstd,
    T s_ddx_dy,
    T s_ddx_x,
    T s_dy_x,
    T s_ddx,
    T s_dy,
    const T* ddX,
    const T* ddgamma,
    const T* dY,
    const T* X,
    T* dX) {
  const bool ddX_null = ddX == nullptr;
  const bool ddgamma_null = ddgamma == nullptr;
  const T scale = T(1) / static_cast<T>(N);
  const T r2 = rstd * rstd;
  const T r3 = r2 * rstd;
  // dX = a * dY + b * X + c
  const T q = s_dy * mean - s_dy_x;
  const T b = scale * r3 * q;
  // d(a * dY)/dX = a1 * dY + a2 * X + a3
  // d(b * X)/dX  = b1 * dY + b2 * X + b3 + b * ddX
  // dc/dX        = c1 * dY + c2 * X + c3
  T a1 = 0;
  T a2 = -scale * r3;
  T a3 = -mean * a2;
  T b1 = T(3) * scale * r2 * q * a1 - scale * r3;
  T b2 = T(3) * scale * r2 * q * a2;
  T b3 = T(3) * scale * r2 * q * a3 + scale * scale * r3 * s_dy;
  T c1 = -(scale * s_dy * a1 + mean * b1) * s_ddx;
  T c2 = -(scale * s_dy * a2 + mean * b2) * s_ddx;
  T c3 = -(scale * s_dy * a3 + mean * b3 + scale * b) * s_ddx;
  a1 *= s_ddx_dy;
  a2 *= s_ddx_dy;
  a3 *= s_ddx_dy;
  b1 *= s_ddx_x;
  b2 *= s_ddx_x;
  b3 *= s_ddx_x;
  const T u = a1 + b1 + c1;
  const T v = a2 + b2 + c2;
  const T w = a3 + b3 + c3;
  for (int64_t i = 0; i < N; ++i) {
    const T ddX_v = ddX_null ? T(0) : ddX[i];
    dX[i] = u * dY[i] + v * X[i] + w + b * ddX_v;
  }
  if (!ddgamma_null) {
    T s_ddg_dy_x = 0;
    T s_ddg_dy = 0;
    for (int64_t i = 0; i < N; ++i) {
      dX[i] += rstd * ddgamma[i] * dY[i];
      s_ddg_dy_x += ddgamma[i] * dY[i] * X[i];
      s_ddg_dy += ddgamma[i] * dY[i];
    }
    T p1 = -scale * r3;
    T p2 = -mean * p1;
    T q1 = -mean * p1 * s_ddg_dy;
    T q2 = -(mean * p2 + scale * rstd) * s_ddg_dy;
    p1 *= s_ddg_dy_x;
    p2 *= s_ddg_dy_x;
    const T uu = p1 + q1;
    const T vv = p2 + q2;
    for (std::int64_t i = 0; i < N; ++i) {
      dX[i] += uu * X[i] + vv;
    }
  }
}

template <typename T>
void LayerNormGammaDoubleBackward(
    const std::int64_t N,
    T mean,
    T rstd,
    T s_ddx_x,
    T s_ddx,
    const T* ddX,
    const T* dY,
    const T* X,
    T* dgamma) {
  const bool ddX_null = ddX == nullptr;
  const T scale = T(1) / static_cast<T>(N);
  T b1 = -scale * rstd * rstd * rstd;
  T b2 = -mean * b1;
  T c1 = -mean * b1 * s_ddx;
  T c2 = -(mean * b2 + scale * rstd) * s_ddx;
  b1 *= s_ddx_x;
  b2 *= s_ddx_x;
  const T u = b1 + c1;
  const T v = b2 + c2;
  for (std::int64_t i = 0; i < N; ++i) {
    const T ddX_v = ddX_null ? T(0) : ddX[i];
    dgamma[i] += (u * X[i] + v + rstd * ddX_v) * dY[i];
  }
}

template <typename T>
void LayerNormDoubleBackwardKernelImplInternal(
    const Tensor& ddX,
    const Tensor& ddgamma,
    const Tensor& ddbeta,
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    Tensor* ddY,
    Tensor* dX,
    Tensor* dgamma) {
  DCHECK(!ddX.defined() || ddX.numel() == M * N);
  DCHECK(!ddgamma.defined() || ddgamma.numel() == N);
  DCHECK(!ddbeta.defined() || ddbeta.numel() == N);
  const T* ddX_data = ddX.defined() ? ddX.template data_ptr<T>() : nullptr;
  const T* ddgamma_data =
      ddgamma.defined() ? ddgamma.template data_ptr<T>() : nullptr;
  const T* ddbeta_data = ddbeta.defined() ? ddbeta.template data_ptr<T>() : nullptr;
  const T* dY_data = dY.template data_ptr<T>();
  const T* X_data = X.template data_ptr<T>();
  const T* mean_data = mean.template data_ptr<T>();
  const T* rstd_data = rstd.template data_ptr<T>();
  const T* gamma_data = gamma.defined() ? gamma.template data_ptr<T>() : nullptr;
  T* ddY_data = ddY->defined() ? ddY->data_ptr<T>() : nullptr;
  T* dX_data = dX->defined() ? dX->data_ptr<T>() : nullptr;
  T* dgamma_data = dgamma->defined() ? dgamma->data_ptr<T>() : nullptr;
  if (dgamma_data != nullptr) {
    std::memset(dgamma_data, 0, N * sizeof(dgamma_data));
  }
  const bool ddX_null = ddX_data == nullptr;
  const bool gamma_null = gamma_data == nullptr;
  for (int64_t i = 0; i < M; ++i) {
    const T* ddX_ptr = ddX_null ? nullptr : ddX_data + i * N;
    const T* dY_ptr = dY_data + i * N;
    const T* X_ptr = X_data + i * N;
    T s_ddx_dy = 0;
    T s_ddx_x = 0;
    T s_dy_x = 0;
    T s_ddx = 0;
    T s_dy = 0;
    for (int j = 0; j < N; ++j) {
      const T ddX_v = ddX_null ? T(0) : ddX_ptr[j];
      const T gamma_v = gamma_null ? T(1) : gamma_data[j];
      s_ddx_dy += ddX_v * dY_ptr[j] * gamma_v;
      s_ddx_x += ddX_v * X_ptr[j] * gamma_v;
      s_dy_x += dY_ptr[j] * X_ptr[j] * gamma_v;
      s_ddx += ddX_v * gamma_v;
      s_dy += dY_ptr[j] * gamma_v;
    }
    if (ddY_data != nullptr) {
      LayerNormOutputDoubleBackward<T>(
          N,
          mean_data[i],
          rstd_data[i],
          s_ddx_x,
          s_ddx,
          ddX_ptr,
          ddgamma_data,
          ddbeta_data,
          gamma_data,
          X_ptr,
          ddY_data + i * N);
    }
    if (dX_data != nullptr) {
      LayerNormInputDoubleBackward<T>(
          N,
          mean_data[i],
          rstd_data[i],
          s_ddx_dy,
          s_ddx_x,
          s_dy_x,
          s_ddx,
          s_dy,
          ddX_ptr,
          ddgamma_data,
          dY_ptr,
          X_ptr,
          dX_data + i * N);
    }
    if (dgamma_data != nullptr) {
      LayerNormGammaDoubleBackward<T>(
          N,
          mean_data[i],
          rstd_data[i],
          s_ddx_x,
          s_ddx,
          ddX_ptr,
          dY_ptr,
          X_ptr,
          dgamma_data);
    }
  }
}

void LayerNormDoubleBackwardKernelImpl(
    const Tensor& ddX,
    const Tensor& ddgamma,
    const Tensor& ddbeta,
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    Tensor* ddY,
    Tensor* dX,
    Tensor* dgamma) {
  AT_DISPATCH_FLOATING_TYPES(
      X.scalar_type(), "LayerNormDoubleBackwardKernelImpl", [&]() {
        LayerNormDoubleBackwardKernelImplInternal<scalar_t>(
            ddX,
            ddgamma,
            ddbeta,
            dY,
            X,
            mean,
            rstd,
            gamma,
            M,
            N,
            ddY,
            dX,
            dgamma);
      });
}

} // namespace

REGISTER_DISPATCH(LayerNormKernel, &LayerNormKernelImpl);
REGISTER_DISPATCH(LayerNormBackwardKernel, &LayerNormBackwardKernelImpl);
REGISTER_DISPATCH(
    LayerNormDoubleBackwardKernel,
    &LayerNormDoubleBackwardKernelImpl);

} // namespace native
} // namespace at
