#include <ATen/NativeFunctions.h>

#include <algorithm>
#include <array>
#include <functional>
#include <numeric>
#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Config.h>
#include <ATen/Parallel.h>
#include <ATen/detail/CUDAHooksInterface.h>

namespace at {
namespace native {

namespace {

template <typename T>
void ElementwiseAffine(
    const std::int64_t N,
    const T mean,
    const T rstd,
    const T* X,
    const T* gamma,
    const T* beta,
    T* Y) {
  const T scale = rstd;
  const T bias = -rstd * mean;
  if (gamma != nullptr && beta != nullptr) {
    for (std::int64_t i = 0; i < N; ++i) {
      Y[i] = (X[i] * scale + bias) * gamma[i] + beta[i];
    }
  } else if (gamma != nullptr) {
    for (std::int64_t i = 0; i < N; ++i) {
      Y[i] = (X[i] * scale + bias) * gamma[i];
    }
  } else if (beta != nullptr) {
    for (std::int64_t i = 0; i < N; ++i) {
      Y[i] = X[i] * scale + bias + beta[i];
    }
  } else {
    for (std::int64_t i = 0; i < N; ++i) {
      Y[i] = X[i] * scale + bias;
    }
  }
}

template <typename T>
void LayerNormBackward(
    const std::int64_t N,
    const T mean,
    const T rstd,
    const T* dY,
    const T* X,
    const T* gamma,
    T* dX) {
  const T scale = T(1) / static_cast<T>(N);
  T ds = T(0);
  T db = T(0);
  if (gamma != nullptr) {
    for (std::int64_t i = 0; i < N; ++i) {
      ds += dY[i] * X[i] * gamma[i];
      db += dY[i] * gamma[i];
    }
  } else {
    for (std::int64_t i = 0; i < N; ++i) {
      ds += dY[i] * X[i];
      db += dY[i];
    }
  }
  const T a = rstd;
  const T b = (db * mean - ds) * rstd * rstd * rstd * scale;
  const T c = -b * mean - db * rstd * scale;
  if (gamma != nullptr) {
    for (std::int64_t i = 0; i < N; ++i) {
      dX[i] = a * dY[i] * gamma[i] + b * X[i] + c;
    }
  } else {
    for (std::int64_t i = 0; i < N; ++i) {
      dX[i] = a * dY[i] + b * X[i] + c;
    }
  }
}

template <typename T>
void GammaBetaBackward(
    const std::int64_t N,
    const T mean,
    const T rstd,
    const T* dY,
    const T* X,
    T* dgamma,
    T* dbeta) {
  const T a = rstd;
  const T b = -rstd * mean;
  if (dgamma != nullptr) {
    for (std::int64_t i = 0; i < N; ++i) {
      dgamma[i] += dY[i] * (a * X[i] + b);
    }
  }
  if (dbeta != nullptr) {
    for (std::int64_t i = 0; i < N; ++i) {
      dbeta[i] += dY[i];
    }
  }
}

template <typename T>
void ComputeInternalGradients(
    const std::int64_t N,
    const T* ddX,
    const T* dY,
    const T* X,
    const T* gamma,
    T* s_ddx_dy,
    T* s_ddx_x,
    T* s_dy_x,
    T* s_ddx,
    T* s_dy) {
  T s1 = 0;
  T s2 = 0;
  T s3 = 0;
  T s4 = 0;
  T s5 = 0;
  if (gamma != nullptr) {
    if (ddX != nullptr) {
      for (std::int64_t i = 0; i < N; ++i) {
        s1 += ddX[i] * dY[i] * gamma[i];
        s2 += ddX[i] * X[i] * gamma[i];
        s4 += ddX[i] * gamma[i];
      }
    }
    for (std::int64_t i = 0; i < N; ++i) {
      s3 += dY[i] * X[i] * gamma[i];
      s5 += dY[i] * gamma[i];
    }
  } else {
    if (ddX != nullptr) {
      for (std::int64_t i = 0; i < N; ++i) {
        s1 += ddX[i] * dY[i];
        s2 += ddX[i] * X[i];
        s4 += ddX[i];
      }
    }
    for (std::int64_t i = 0; i < N; ++i) {
      s3 += dY[i] * X[i];
      s5 += dY[i];
    }
  }
  *s_ddx_dy = s1;
  *s_ddx_x = s2;
  *s_dy_x = s3;
  *s_ddx = s4;
  *s_dy = s5;
}

template <typename T>
void LayerNormOutputDoubleBackward(
    const std::int64_t N,
    const T mean,
    const T rstd,
    const T s_ddx_x,
    const T s_ddx,
    const T* ddX,
    const T* ddgamma,
    const T* ddbeta,
    const T* gamma,
    const T* X,
    T* ddY) {
  const T scale = T(1) / static_cast<T>(N);
  T b1 = -scale * rstd * rstd * rstd;
  T b2 = -mean * b1;
  T c1 = -mean * b1 * s_ddx;
  T c2 = -(mean * b2 + scale * rstd) * s_ddx;
  b1 *= s_ddx_x;
  b2 *= s_ddx_x;
  const T u = b1 + c1;
  const T v = b2 + c2;
  if (gamma != nullptr) {
    for (std::int64_t i = 0; i < N; ++i) {
      ddY[i] = (u * X[i] + v) * gamma[i];
    }
    if (ddX != nullptr) {
      for (std::int64_t i = 0; i < N; ++i) {
        ddY[i] += rstd * ddX[i] * gamma[i];
      }
    }
  } else {
    for (std::int64_t i = 0; i < N; ++i) {
      ddY[i] = u * X[i] + v;
    }
    if (ddX != nullptr) {
      for (std::int64_t i = 0; i < N; ++i) {
        ddY[i] += rstd * ddX[i];
      }
    }
  }
  if (ddgamma != nullptr) {
    const T a = rstd;
    const T b = -mean * a;
    for (std::int64_t i = 0; i < N; ++i) {
      ddY[i] += ddgamma[i] * (a * X[i] + b);
    }
  }
  if (ddbeta != nullptr) {
    for (std::int64_t i = 0; i < N; ++i) {
      ddY[i] += ddbeta[i];
    }
  }
}

template <typename T>
void LayerNormInputDoubleBackward(
    const std::int64_t N,
    const T mean,
    const T rstd,
    const T s_ddx_dy,
    const T s_ddx_x,
    const T s_dy_x,
    const T s_ddx,
    const T s_dy,
    const T* ddX,
    const T* ddgamma,
    const T* dY,
    const T* X,
    T* dX) {
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
  for (std::int64_t i = 0; i < N; ++i) {
    dX[i] = u * dY[i] + v * X[i] + w;
  }
  if (ddX != nullptr) {
    for (std::int64_t i = 0; i < N; ++i) {
      dX[i] += b * ddX[i];
    }
  }
  if (ddgamma != nullptr) {
    T s_ddg_dy_x = 0;
    T s_ddg_dy = 0;
    for (std::int64_t i = 0; i < N; ++i) {
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
void GammaDoubleBackward(
    const std::int64_t N,
    const T mean,
    const T rstd,
    const T s_ddx_x,
    const T s_ddx,
    const T* ddX,
    const T* dY,
    const T* X,
    T* dgamma) {
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
    dgamma[i] += (u * X[i] + v) * dY[i];
  }
  if (ddX != nullptr) {
    for (std::int64_t i = 0; i < N; ++i) {
      dgamma[i] += rstd * ddX[i] * dY[i];
    }
  }
}

template <typename T>
std::tuple<Tensor, Tensor, Tensor> LayerNormForwardCPUImpl(
    const Tensor& X,
    const Tensor& gamma /* optional */,
    const Tensor& beta /* optional */,
    const int axis,
    const T eps) {
  const std::vector<std::int64_t> X_dims = X.sizes().vec();
  const std::vector<std::int64_t> outer_dims(
      X_dims.cbegin(), X_dims.cbegin() + axis);
  const std::int64_t M = std::accumulate(
      X_dims.cbegin(),
      X_dims.cbegin() + axis,
      1LL,
      std::multiplies<std::int64_t>());
  const std::int64_t N = std::accumulate(
      X_dims.cbegin() + axis,
      X_dims.cend(),
      1LL,
      std::multiplies<std::int64_t>());
  Tensor Y = at::empty_like(X);
  Tensor mean = at::empty(outer_dims, X.options());
  Tensor rstd = at::empty(outer_dims, X.options());
  const T* X_data = X.data<T>();
  const T* gamma_data = gamma.defined() ? gamma.data<T>() : nullptr;
  const T* beta_data = beta.defined() ? beta.data<T>() : nullptr;
  T* Y_data = Y.data<T>();
  T* mean_data = mean.data<T>();
  T* rstd_data = rstd.data<T>();
  const T c = T(1) / static_cast<T>(N);
  for (std::int64_t i = 0; i < M; ++i) {
    const T* X_ptr = X_data + i * N;
    T* Y_ptr = Y_data + i * N;
    T mean_val = T(0);
    T rstd_val = T(0);
    for (std::int64_t j = 0; j < N; ++j) {
      mean_val += X_ptr[j];
      rstd_val += X_ptr[j] * X_ptr[j];
    }
    mean_val *= c;
    rstd_val = T(1) / std::sqrt(rstd_val * c - mean_val * mean_val + eps);
    ElementwiseAffine<T>(
        N, mean_val, rstd_val, X_ptr, gamma_data, beta_data, Y_ptr);
    mean_data[i] = mean_val;
    rstd_data[i] = rstd_val;
  }
  return std::make_tuple(Y, mean, rstd);
}

template <typename T>
std::tuple<Tensor, Tensor, Tensor> LayerNormBackwardCPUImpl(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma /* optional */,
    const int axis,
    const std::array<bool, 3>& grad_input_mask) {
  const std::vector<std::int64_t> X_dims = X.sizes().vec();
  const std::int64_t M = std::accumulate(
      X_dims.cbegin(),
      X_dims.cbegin() + axis,
      1LL,
      std::multiplies<std::int64_t>());
  const std::int64_t N = std::accumulate(
      X_dims.cbegin() + axis,
      X_dims.cend(),
      1LL,
      std::multiplies<std::int64_t>());

  const T* dY_data = dY.template data<T>();
  const T* X_data = X.template data<T>();
  const T* mean_data = mean.template data<T>();
  const T* rstd_data = rstd.template data<T>();
  const T* gamma_data = gamma.defined() ? gamma.template data<T>() : nullptr;
  Tensor dX;
  Tensor dgamma;
  Tensor dbeta;
  T* dX_data = nullptr;
  T* dgamma_data = nullptr;
  T* dbeta_data = nullptr;
  if (grad_input_mask[0]) {
    dX = at::empty_like(X);
    dX_data = dX.template data<T>();
  }
  if (grad_input_mask[1]) {
    dgamma = at::zeros_like(gamma);
    dgamma_data = dgamma.template data<T>();
  }
  if (grad_input_mask[2]) {
    dbeta = at::zeros_like(gamma);
    dbeta_data = dbeta.template data<T>();
  }
  for (std::int64_t i = 0; i < M; ++i) {
    const T* dY_ptr = dY_data + i * N;
    const T* X_ptr = X_data + i * N;
    if (dX_data != nullptr) {
      T* dX_ptr = dX_data + i * N;
      LayerNormBackward<T>(
          N, mean_data[i], rstd_data[i], dY_ptr, X_ptr, gamma_data, dX_ptr);
    }
    GammaBetaBackward<T>(
        N, mean_data[i], rstd_data[i], dY_ptr, X_ptr, dgamma_data, dbeta_data);
  }
  return std::make_tuple(dX, dgamma, dbeta);
}

template <typename T>
std::tuple<Tensor, Tensor, Tensor> LayerNormDoubleBackwardCPUImpl(
    const Tensor& ddX,
    const Tensor& ddgamma,
    const Tensor& ddbeta,
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    const int axis,
    const std::array<bool, 3>& grad_input_mask) {
  const std::vector<std::int64_t> X_dims = X.sizes().vec();
  const std::int64_t M = std::accumulate(
      X_dims.cbegin(),
      X_dims.cbegin() + axis,
      1LL,
      std::multiplies<std::int64_t>());
  const std::int64_t N = std::accumulate(
      X_dims.cbegin() + axis,
      X_dims.cend(),
      1LL,
      std::multiplies<std::int64_t>());

  const T* ddX_data = ddX.defined() ? ddX.template data<T>() : nullptr;
  const T* ddgamma_data =
      ddgamma.defined() ? ddgamma.template data<T>() : nullptr;
  const T* ddbeta_data = ddbeta.defined() ? ddbeta.template data<T>() : nullptr;
  const T* dY_data = dY.template data<T>();
  const T* X_data = X.template data<T>();
  const T* mean_data = mean.template data<T>();
  const T* rstd_data = rstd.template data<T>();
  const T* gamma_data = gamma.defined() ? gamma.template data<T>() : nullptr;
  Tensor ddY;
  Tensor dX;
  Tensor dgamma;
  T* ddY_data = nullptr;
  T* dX_data = nullptr;
  T* dgamma_data = nullptr;
  if (grad_input_mask[0]) {
    ddY = at::empty_like(dY);
    ddY_data = ddY.template data<T>();
  }
  if (grad_input_mask[1]) {
    dX = at::empty_like(X);
    dX_data = dX.template data<T>();
  }
  if (grad_input_mask[2]) {
    dgamma = at::zeros_like(gamma);
    dgamma_data = dgamma.template data<T>();
  }
  for (std::int64_t i = 0; i < M; ++i) {
    const T* ddX_ptr = ddX_data == nullptr ? nullptr : ddX_data + i * N;
    const T* dY_ptr = dY_data + i * N;
    const T* X_ptr = X_data + i * N;
    T s_ddx_dy;
    T s_ddx_x;
    T s_dy_x;
    T s_ddx;
    T s_dy;
    ComputeInternalGradients<T>(
        N,
        ddX_ptr,
        dY_ptr,
        X_ptr,
        gamma_data,
        &s_ddx_dy,
        &s_ddx_x,
        &s_dy_x,
        &s_ddx,
        &s_dy);
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
      GammaDoubleBackward<T>(
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

  return std::make_tuple(ddY, dX, dgamma);
}

} // namespace

std::tuple<Tensor, Tensor, Tensor> layer_norm_forward_cpu(
    const Tensor& X,
    const Tensor& gamma /* optional */,
    const Tensor& beta /* optional */,
    const IntArrayRef normalized_shape,
    const double eps) {
  const int normalized_ndim = normalized_shape.size();
  AT_CHECK(
      normalized_ndim >= 1,
      "Expected normalized_shape to be at least 1-dimensional, i.e., ",
      "containing at least one element, but got normalized_shape = ",
      normalized_shape);
  AT_CHECK(
      !gamma.defined() || gamma.sizes().equals(normalized_shape),
      "Expected weight to be of same shape as normalized_shape, but got ",
      "weight of shape ",
      gamma.sizes(),
      " and normalized_shape = ",
      normalized_shape);
  AT_CHECK(
      !beta.defined() || beta.sizes().equals(normalized_shape),
      "Expected bias to be of same shape as normalized_shape, but got ",
      "bias of shape ",
      beta.sizes(),
      " and normalized_shape = ",
      normalized_shape);
  const auto X_shape = X.sizes();
  const int X_ndim = X.dim();
  if (X_ndim < normalized_ndim ||
      !X_shape.slice(X_ndim - normalized_ndim).equals(normalized_shape)) {
    std::stringstream ss;
    ss << "Given normalized_shape = " << normalized_shape
       << " , expected input with shape [*";
    for (const std::int64_t size : normalized_shape) {
      ss << ", " << size;
    }
    ss << "], but got input of size" << X_shape;
    AT_ERROR(ss.str());
  }
  const int axis = X_ndim - normalized_ndim;
  return AT_DISPATCH_FLOATING_TYPES(
      X.scalar_type(), "LayerNormForwardCPU", [&]() {
        return LayerNormForwardCPUImpl<scalar_t>(
            X, gamma, beta, axis, static_cast<scalar_t>(eps));
      });
}

std::tuple<Tensor, Tensor, Tensor> layer_norm_backward_cpu(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma /* optional */,
    const IntArrayRef normalized_shape,
    const std::array<bool, 3> grad_input_mask) {
  const int normalized_ndim = normalized_shape.size();
  const int X_ndim = X.dim();
  const int axis = X_ndim - normalized_ndim;
  return AT_DISPATCH_FLOATING_TYPES(
      X.scalar_type(), "LayerNormBackwardCPU", [&]() {
        return LayerNormBackwardCPUImpl<scalar_t>(
            dY, X, mean, rstd, gamma, axis, grad_input_mask);
      });
}

std::tuple<Tensor, Tensor, Tensor> layer_norm_double_backward_cpu(
    const Tensor& ddX,
    const Tensor& ddgamma,
    const Tensor& ddbeta,
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    const IntArrayRef normalized_shape,
    const std::array<bool, 3> grad_input_mask) {
  const int normalized_ndim = normalized_shape.size();
  const int X_ndim = X.dim();
  const int axis = X_ndim - normalized_ndim;
  return AT_DISPATCH_FLOATING_TYPES(
      X.scalar_type(), "LayerNormDoubleBackwardCPU", [&]() {
        return LayerNormDoubleBackwardCPUImpl<scalar_t>(
            ddX,
            ddgamma,
            ddbeta,
            dY,
            X,
            mean,
            rstd,
            gamma,
            axis,
            grad_input_mask);
      });
}

} // namespace native
} // namespace at
