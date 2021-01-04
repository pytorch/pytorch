#include <ATen/native/layer_norm.h>

#include <cmath>

#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/cpu/vec256/functional.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/Parallel.h>

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
  using Vec = vec256::Vec256<T>;
  DCHECK_EQ(X.numel(), M * N);
  DCHECK(!gamma.defined() || gamma.numel() == N);
  DCHECK(!beta.defined() || beta.numel() == N);
  T* X_data = X.data_ptr<T>();
  const T* gamma_data = gamma.defined() ? gamma.data_ptr<T>() : nullptr;
  const T* beta_data = beta.defined() ? beta.data_ptr<T>() : nullptr;
  T* Y_data = Y->data_ptr<T>();
  T* mean_data = mean->data_ptr<T>();
  T* rstd_data = rstd->data_ptr<T>();
  const T c = T(1) / static_cast<T>(N);
  const bool gamma_null = gamma_data == nullptr;
  const bool beta_null = beta_data == nullptr;
  at::parallel_for(0, M, 1, [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; ++i) {
      T* X_ptr = X_data + i * N;
      T* Y_ptr = Y_data + i * N;
      T mean_val = vec256::reduce_all<T>(
          [](Vec& x, Vec& y) { return x + y; },
          X_ptr,
          N);
      T rstd_val = vec256::map_reduce_all<T>(
          [](Vec x) { return x * x; },
          [](Vec x, Vec y) { return x + y; },
          X_ptr,
          N);
      mean_val *= c;
      rstd_val = std::max(rstd_val * c - mean_val * mean_val, T(0));
      rstd_val = T(1) / std::sqrt(rstd_val + eps);
      const T scale = rstd_val;
      const T bias = -rstd_val * mean_val;
      if (gamma_null || beta_null) {
        for (int64_t j = 0; j < N; ++j) {
          const T gamma_v = gamma_null ? T(1) : gamma_data[j];
          const T beta_v = beta_null ? T(0) : beta_data[j];
          Y_ptr[j] = (X_ptr[j] * scale + bias) * gamma_v + beta_v;
        }
      } else {
        vec256::map3<T>(
            [scale, bias](Vec x, Vec gamma, Vec beta) {
              return (x * Vec(scale) + Vec(bias)) * gamma + beta;
            },
            Y_ptr,
            X_ptr,
            gamma_data,
            beta_data,
            N);
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
  using Vec = vec256::Vec256<T>;
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
  T* dbeta_data = dbeta->defined() ? dbeta->template data_ptr<T>() : nullptr;
  const T scale = T(1) / static_cast<T>(N);
  const bool gamma_null = gamma_data == nullptr;
  const bool dX_null = dX_data == nullptr;
  const bool dgamma_null = dgamma_data == nullptr;
  const bool dbeta_null = dbeta_data == nullptr;

  // 1. Use two path parallel reduction for dgamma and dbeta:
  //    First path: allocate an immediate buffer of size {2, max_threads, N},
  //        dgamma_buffer = buffer[0], dbeta_buffer = buffer[1]
  //    Parallel along dim0 and reduce dY and X along dim0 to buffer.
  //    Second path: parallel along dim1 and reduce buffer to dgamma and dbeta.
  //
  // 2. Fuse first path of dgamma/dbeta with dX to reuse X[i] and dY[i] in L1 cache.
  //
  int num_threads = at::get_num_threads();
  Tensor buffer = at::empty({0}, X.options());
  T* buffer_data = nullptr;
  if (!dgamma_null || !dbeta_null) {
    // zero the immediate buffer and skip zero dgamma and dbeta
    buffer.resize_({2, num_threads, N}).zero_();
    buffer_data = buffer.template data_ptr<T>();
  }

  // First path of dgamma/dbeta and dX
  at::parallel_for(0, M, 1, [&](int64_t start, int64_t end) {
    int tid = at::get_thread_num();
    TORCH_CHECK(tid < num_threads,
                "expect thread id smaller than ", num_threads, ", got thread id ", tid);
    T* dgamma_buffer_ptr = dgamma_null ? nullptr : buffer_data + tid * N;
    T* dbeta_buffer_ptr = dbeta_null ? nullptr : buffer_data + num_threads * N + tid * N;
    for (int64_t i = start; i < end; ++i) {
      const T* dY_ptr = dY_data + i * N;
      const T* X_ptr = X_data + i * N;
      if (!dgamma_null) {
        const T a = rstd_data[i];
        const T b = -a * mean_data[i];
        // Scalar math:
        // for (int64_t j = 0; j < N; ++j) {
        //   dgamma_data[j] += dY_ptr[j] * (a * X_ptr[j] + b);
        // }
        vec256::map3<T>(
            [a, b](Vec dgamma, Vec dy, Vec x) { return dgamma + dy * (Vec(a) * x + Vec(b)); },
            dgamma_buffer_ptr,
            dgamma_buffer_ptr,
            dY_ptr,
            X_ptr,
            N);
      }
      if (!dbeta_null) {
        // Scalar math:
        // for (int64_t j = 0; j < N; ++j) {
        //   dbeta_data[j] += dY_ptr[j];
        // }
        vec256::map2<T>(
            [](Vec dbeta, Vec dy) { return dbeta + dy; },
            dbeta_buffer_ptr,
            dbeta_buffer_ptr,
            dY_ptr,
            N);
      }
      if (!dX_null) {
        T* dX_ptr = dX_data + i * N;
        T ds = T(0);
        T db = T(0);
        // Scalar math:
        // for (int64_t j = 0; j < N; ++j) {
        //   const T gamma_v = gamma_null ? T(1) : gamma_data[j];
        //   ds += dY_ptr[j] * X_ptr[j] * gamma_v;
        //   db += dY_ptr[j] * gamma_v;
        // }
        if (gamma_null) {
          ds = vec256::map2_reduce_all<T>(
              [](Vec x, Vec y) { return x * y; },
              [](Vec x, Vec y) { return x + y; },
              dY_ptr,
              X_ptr,
              N);
          db = vec256::reduce_all<T>(
              [](Vec& x, Vec& y) { return x + y; },
              dY_ptr,
              N);
        } else {
          ds = vec256::map3_reduce_all<T>(
              [](Vec x, Vec y, Vec z) { return x * y * z; },
              [](Vec x, Vec y) { return x + y; },
              dY_ptr,
              X_ptr,
              gamma_data,
              N);
          db = vec256::map2_reduce_all<T>(
              [](Vec x, Vec y) { return x * y; },
              [](Vec x, Vec y) { return x + y; },
              dY_ptr,
              gamma_data,
              N);
        }
        const T a = rstd_data[i];
        const T b = (db * mean_data[i] - ds) * a * a * a * scale;
        const T c = -b * mean_data[i] - db * a * scale;
        // Scalar math:
        // for (int64_t j = 0; j < N; ++j) {
        //   const T gamma_v = gamma_null ? T(1) : gamma_data[j];
        //   dX_ptr[j] = a * dY_ptr[j] * gamma_v + b * X_ptr[j] + c;
        // }
        if (gamma_null) {
          vec256::map2<T>(
              [a, b, c](Vec dy, Vec x) { return Vec(a) * dy + Vec(b) * x + Vec(c); },
              dX_ptr,
              dY_ptr,
              X_ptr,
              N);
        } else {
          vec256::map3<T>(
              [a, b, c](Vec dy, Vec gamma, Vec x) { return Vec(a) * dy * gamma + Vec(b) * x + Vec(c); },
              dX_ptr,
              dY_ptr,
              gamma_data,
              X_ptr,
              N);
        }
      }
    }
  });

  // Second path of dgamma/dbeta
  if (buffer_data != nullptr) {
    parallel_for(0, N, 1, [&](int64_t start, int64_t end) {
      for (int64_t j = start; j < end; ++j) {
        T dgamma_v = T(0);
        T dbeta_v = T(0);
        for (int64_t i = 0; i < num_threads; ++i) {
          dgamma_v += buffer_data[i * N + j];
          dbeta_v += buffer_data[num_threads * N + i * N + j];
        }
        if (!dgamma_null) {
          dgamma_data[j] = dgamma_v;
        }
        if (!dbeta_null) {
          dbeta_data[j] = dbeta_v;
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
