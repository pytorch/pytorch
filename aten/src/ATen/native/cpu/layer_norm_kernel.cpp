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
    const Tensor& weight,
    const Tensor& bias,
    int64_t M,
    int64_t N,
    T eps,
    Tensor* Y,
    Tensor* mean,
    Tensor* rstd)
{
  DCHECK_EQ(X.numel(), M * N);
  DCHECK(!weight.defined() || weight.numel() == N);
  DCHECK(!bias.defined() || bias.numel() == N);
  const T* X_data = X.data_ptr<T>();
  const T* weight_data = weight.defined() ? weight.data_ptr<T>() : nullptr;
  const T* bias_data = bias.defined() ? bias.data_ptr<T>() : nullptr;
  T* Y_data = Y->data_ptr<T>();
  T* mean_data = mean->data_ptr<T>();
  T* rstd_data = rstd->data_ptr<T>();
  const T c = T(1) / static_cast<T>(N);
  const bool weight_null = weight_data == nullptr;
  const bool bias_null = bias_data == nullptr;
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
      const T weight_v = weight_null ? T(1) : weight_data[j];
      const T bias_v = bias_null ? T(0) : bias_data[j];
      Y_ptr[j] = (X_ptr[j] * scale + bias) * weight_v + bias_v;
    }
    mean_data[i] = mean_val;
    rstd_data[i] = rstd_val;
  }
}

void LayerNormKernelImpl(
    const Tensor& X,
    const Tensor& weight,
    const Tensor& bias,
    int64_t M,
    int64_t N,
    double eps,
    Tensor* Y,
    Tensor* mean,
    Tensor* rstd)
{
  AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "LayerNormKernelImpl", [&]() {
    LayerNormKernelImplInternal<scalar_t>(
        X, weight, bias, M, N, static_cast<scalar_t>(eps), Y, mean, rstd);
  });
}

template <typename T>
void LayerNormBackwardKernelImplInternal(
    const Tensor& grad_out,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& weight,
    int64_t M,
    int64_t N,
    Tensor* dX,
    Tensor* dweight,
    Tensor* dbias)
{
  DCHECK_EQ(grad_out.numel(), M * N);
  DCHECK_EQ(X.numel(), M * N);
  DCHECK_EQ(mean.numel(), M);
  DCHECK_EQ(rstd.numel(), M);
  DCHECK(!weight.defined() || weight.numel() == N);
  const T* grad_out_data = grad_out.template data_ptr<T>();
  const T* X_data = X.template data_ptr<T>();
  const T* mean_data = mean.template data_ptr<T>();
  const T* rstd_data = rstd.template data_ptr<T>();
  const T* weight_data = weight.defined() ? weight.template data_ptr<T>() : nullptr;
  T* dX_data = dX->defined() ? dX->template data_ptr<T>() : nullptr;
  T* dweight_data = dweight->defined() ? dweight->template data_ptr<T>() : nullptr;
  if (dweight_data != nullptr) {
    std::memset(dweight_data, 0, N * sizeof(T));
  }
  T* dbias_data = dbias->defined() ? dbias->template data_ptr<T>() : nullptr;
  if (dbias_data != nullptr) {
    std::memset(dbias_data, 0, N * sizeof(T));
  }
  const T scale = T(1) / static_cast<T>(N);
  const bool weight_null = weight_data == nullptr;
  for (int64_t i = 0; i < M; ++i) {
    const T* grad_out_ptr = grad_out_data + i * N;
    const T* X_ptr = X_data + i * N;
    if (dX_data != nullptr) {
      T* dX_ptr = dX_data + i * N;
      T ds = 0;
      T db = 0;
      for (int64_t j = 0; j < N; ++j) {
        const T weight_v = weight_null ? T(1) : weight_data[j];
        ds += grad_out_ptr[j] * X_ptr[j] * weight_v;
        db += grad_out_ptr[j] * weight_v;
      }
      const T a = rstd_data[i];
      const T b = (db * mean_data[i] - ds) * a * a * a * scale;
      const T c = -b * mean_data[i] - db * a * scale;
      for (int64_t j = 0; j < N; ++j) {
        const T weight_v = weight_null ? T(1) : weight_data[j];
        dX_ptr[j] = a * grad_out_ptr[j] * weight_v + b * X_ptr[j] + c;
      }
    }
    if (dweight_data != nullptr) {
      const T a = rstd_data[i];
      const T b = -a * mean_data[i];
      for (int64_t j = 0; j < N; ++j) {
        dweight_data[j] += grad_out_ptr[j] * (a * X_ptr[j] + b);
      }
    }
    if (dbias_data != nullptr) {
      for (int64_t j = 0; j < N; ++j) {
        dbias_data[j] += grad_out_ptr[j];
      }
    }
  }
}

void LayerNormBackwardKernelImpl(
    const Tensor& grad_out,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& weight,
    int64_t M,
    int64_t N,
    Tensor* dX,
    Tensor* dweight,
    Tensor* dbias)
{
  AT_DISPATCH_FLOATING_TYPES(
      X.scalar_type(), "LayerNormBackwardKernelImpl", [&]() {
        LayerNormBackwardKernelImplInternal<scalar_t>(
            grad_out, X, mean, rstd, weight, M, N, dX, dweight, dbias);
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
    const T* ddweight,
    const T* ddbias,
    const T* weight,
    const T* X,
    T* dgrad_out)
{
  const bool ddX_null = ddX == nullptr;
  const bool ddweight_null = ddweight == nullptr;
  const bool ddbias_null = ddbias == nullptr;
  const bool weight_null = weight == nullptr;
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
    const T ddweight_v = ddweight_null ? T(0) : ddweight[i];
    const T ddbias_v = ddbias_null ? T(0) : ddbias[i];
    const T weight_v = weight_null ? T(1) : weight[i];
    dgrad_out[i] = (u * X[i] + v + rstd * ddX_v) * weight_v +
        (X[i] - mean) * rstd * ddweight_v + ddbias_v;
  }
}

template <typename T>
void LayerNormInputDoubleBackward(
    int64_t N,
    T mean,
    T rstd,
    T s_ddx_grad_out,
    T s_ddx_x,
    T s_grad_out_x,
    T s_ddx,
    T s_grad_out,
    const T* ddX,
    const T* ddweight,
    const T* grad_out,
    const T* X,
    T* dX)
{
  const bool ddX_null = ddX == nullptr;
  const bool ddweight_null = ddweight == nullptr;
  const T scale = T(1) / static_cast<T>(N);
  const T r2 = rstd * rstd;
  const T r3 = r2 * rstd;
  // dX = a * grad_out + b * X + c
  const T q = s_grad_out * mean - s_grad_out_x;
  const T b = scale * r3 * q;
  // d(a * grad_out)/dX = a1 * grad_out + a2 * X + a3
  // d(b * X)/dX  = b1 * grad_out + b2 * X + b3 + b * ddX
  // dc/dX        = c1 * grad_out + c2 * X + c3
  T a1 = 0;
  T a2 = -scale * r3;
  T a3 = -mean * a2;
  T b1 = T(3) * scale * r2 * q * a1 - scale * r3;
  T b2 = T(3) * scale * r2 * q * a2;
  T b3 = T(3) * scale * r2 * q * a3 + scale * scale * r3 * s_grad_out;
  T c1 = -(scale * s_grad_out * a1 + mean * b1) * s_ddx;
  T c2 = -(scale * s_grad_out * a2 + mean * b2) * s_ddx;
  T c3 = -(scale * s_grad_out * a3 + mean * b3 + scale * b) * s_ddx;
  a1 *= s_ddx_grad_out;
  a2 *= s_ddx_grad_out;
  a3 *= s_ddx_grad_out;
  b1 *= s_ddx_x;
  b2 *= s_ddx_x;
  b3 *= s_ddx_x;
  const T u = a1 + b1 + c1;
  const T v = a2 + b2 + c2;
  const T w = a3 + b3 + c3;
  for (int64_t i = 0; i < N; ++i) {
    const T ddX_v = ddX_null ? T(0) : ddX[i];
    dX[i] = u * grad_out[i] + v * X[i] + w + b * ddX_v;
  }
  if (!ddweight_null) {
    T s_ddg_grad_out_x = 0;
    T s_ddg_grad_out = 0;
    for (int64_t i = 0; i < N; ++i) {
      dX[i] += rstd * ddweight[i] * grad_out[i];
      s_ddg_grad_out_x += ddweight[i] * grad_out[i] * X[i];
      s_ddg_grad_out += ddweight[i] * grad_out[i];
    }
    T p1 = -scale * r3;
    T p2 = -mean * p1;
    T q1 = -mean * p1 * s_ddg_grad_out;
    T q2 = -(mean * p2 + scale * rstd) * s_ddg_grad_out;
    p1 *= s_ddg_grad_out_x;
    p2 *= s_ddg_grad_out_x;
    const T uu = p1 + q1;
    const T vv = p2 + q2;
    for (std::int64_t i = 0; i < N; ++i) {
      dX[i] += uu * X[i] + vv;
    }
  }
}

template <typename T>
void LayerNormweightDoubleBackward(
    const std::int64_t N,
    T mean,
    T rstd,
    T s_ddx_x,
    T s_ddx,
    const T* ddX,
    const T* grad_out,
    const T* X,
    T* dweight)
{
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
    dweight[i] += (u * X[i] + v + rstd * ddX_v) * grad_out[i];
  }
}

template <typename T>
void LayerNormDoubleBackwardKernelImplInternal(
    const Tensor& ddX,
    const Tensor& ddweight,
    const Tensor& ddbias,
    const Tensor& grad_out,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& weight,
    int64_t M,
    int64_t N,
    Tensor* dgrad_out,
    Tensor* dX,
    Tensor* dweight)
{
  DCHECK(!ddX.defined() || ddX.numel() == M * N);
  DCHECK(!ddweight.defined() || ddweight.numel() == N);
  DCHECK(!ddbias.defined() || ddbias.numel() == N);
  const T* ddX_data = ddX.defined() ? ddX.template data_ptr<T>() : nullptr;
  const T* ddweight_data =
      ddweight.defined() ? ddweight.template data_ptr<T>() : nullptr;
  const T* ddbias_data = ddbias.defined() ? ddbias.template data_ptr<T>() : nullptr;
  const T* grad_out_data = grad_out.template data_ptr<T>();
  const T* X_data = X.template data_ptr<T>();
  const T* mean_data = mean.template data_ptr<T>();
  const T* rstd_data = rstd.template data_ptr<T>();
  const T* weight_data = weight.defined() ? weight.template data_ptr<T>() : nullptr;
  T* dgrad_out_data = dgrad_out->defined() ? dgrad_out->data_ptr<T>() : nullptr;
  T* dX_data = dX->defined() ? dX->data_ptr<T>() : nullptr;
  T* dweight_data = dweight->defined() ? dweight->data_ptr<T>() : nullptr;
  if (dweight_data != nullptr) {
    std::memset(dweight_data, 0, N * sizeof(dweight_data));
  }
  const bool ddX_null = ddX_data == nullptr;
  const bool weight_null = weight_data == nullptr;
  for (int64_t i = 0; i < M; ++i) {
    const T* ddX_ptr = ddX_null ? nullptr : ddX_data + i * N;
    const T* grad_out_ptr = grad_out_data + i * N;
    const T* X_ptr = X_data + i * N;
    T s_ddx_grad_out = 0;
    T s_ddx_x = 0;
    T s_grad_out_x = 0;
    T s_ddx = 0;
    T s_grad_out = 0;
    for (int j = 0; j < N; ++j) {
      const T ddX_v = ddX_null ? T(0) : ddX_ptr[j];
      const T weight_v = weight_null ? T(1) : weight_data[j];
      s_ddx_grad_out += ddX_v * grad_out_ptr[j] * weight_v;
      s_ddx_x += ddX_v * X_ptr[j] * weight_v;
      s_grad_out_x += grad_out_ptr[j] * X_ptr[j] * weight_v;
      s_ddx += ddX_v * weight_v;
      s_grad_out += grad_out_ptr[j] * weight_v;
    }
    if (dgrad_out_data != nullptr) {
      LayerNormOutputDoubleBackward<T>(
          N,
          mean_data[i],
          rstd_data[i],
          s_ddx_x,
          s_ddx,
          ddX_ptr,
          ddweight_data,
          ddbias_data,
          weight_data,
          X_ptr,
          dgrad_out_data + i * N);
    }
    if (dX_data != nullptr) {
      LayerNormInputDoubleBackward<T>(
          N,
          mean_data[i],
          rstd_data[i],
          s_ddx_grad_out,
          s_ddx_x,
          s_grad_out_x,
          s_ddx,
          s_grad_out,
          ddX_ptr,
          ddweight_data,
          grad_out_ptr,
          X_ptr,
          dX_data + i * N);
    }
    if (dweight_data != nullptr) {
      LayerNormweightDoubleBackward<T>(
          N,
          mean_data[i],
          rstd_data[i],
          s_ddx_x,
          s_ddx,
          ddX_ptr,
          grad_out_ptr,
          X_ptr,
          dweight_data);
    }
  }
}

void LayerNormDoubleBackwardKernelImpl(
    const Tensor& ddX,
    const Tensor& ddweight,
    const Tensor& ddbias,
    const Tensor& grad_out,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& weight,
    int64_t M,
    int64_t N,
    Tensor* dgrad_out,
    Tensor* dX,
    Tensor* dweight)
{
  AT_DISPATCH_FLOATING_TYPES(
      X.scalar_type(), "LayerNormDoubleBackwardKernelImpl", [&]() {
        LayerNormDoubleBackwardKernelImplInternal<scalar_t>(
            ddX,
            ddweight,
            ddbias,
            grad_out,
            X,
            mean,
            rstd,
            weight,
            M,
            N,
            dgrad_out,
            dX,
            dweight);
      });
}

} // namespace

REGISTER_DISPATCH(LayerNormKernel, &LayerNormKernelImpl);
REGISTER_DISPATCH(LayerNormBackwardKernel, &LayerNormBackwardKernelImpl);
REGISTER_DISPATCH(LayerNormDoubleBackwardKernel, &LayerNormDoubleBackwardKernelImpl);

} // namespace native
} // namespace at
