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
    Tensor& Y,
    Tensor& mean,
    Tensor& rstd) {
  TORCH_CHECK(X.numel() == N * C * HxW);
  TORCH_CHECK(!gamma.defined() || gamma.numel() == C);
  TORCH_CHECK(!beta.defined() || beta.numel() == C);
  const int64_t G = group;
  const int64_t D = C / G;
  const T* X_data = X.data_ptr<T>();
  const T* gamma_data = gamma.defined() ? gamma.data_ptr<T>() : nullptr;
  const T* beta_data = beta.defined() ? beta.data_ptr<T>() : nullptr;
  T* Y_data = Y.data_ptr<T>();
  T* mean_data = mean.data_ptr<T>();
  T* rstd_data = rstd.data_ptr<T>();
  const T s = T(1) / static_cast<T>(D * HxW);
  const bool gamma_null = (gamma_data == nullptr);
  const bool beta_null = beta_data == nullptr;

  using Vec = vec256::Vec256<T>;
  at::parallel_for(0, N * G, 1, [&](int64_t start, int64_t end) {
    constexpr int64_t K = Vec::size();
    const int64_t inner_size = D * HxW / K * K;
    std::array<T, K> mean_arr;
    std::array<T, K> rstd_arr;
    for (int64_t i = start; i < end; ++i) {
      const T* X_ptr = X_data + i * D * HxW;
      Vec mean_vec(0);
      Vec rstd_vec(0);
      for (int64_t j = 0; j < inner_size; j += K) {
        const Vec x_vec = Vec::loadu(X_ptr + j);
        mean_vec = mean_vec + x_vec;
        rstd_vec = rstd_vec + x_vec * x_vec;
      }
      mean_vec.store(mean_arr.data());
      rstd_vec.store(rstd_arr.data());
      T mean_val = std::accumulate(mean_arr.cbegin(), mean_arr.cend(), T(0));
      T rstd_val = std::accumulate(rstd_arr.cbegin(), rstd_arr.cend(), T(0));
      for (int64_t j = inner_size; j < D * HxW; ++j) {
        mean_val += X_ptr[j];
        rstd_val += X_ptr[j] * X_ptr[j];
      }
      mean_val *= s;
      rstd_val = std::max(rstd_val * s - mean_val * mean_val, T(0));
      rstd_val = T(1) / std::sqrt(rstd_val + eps);

      const int64_t g = i % G;
      const int64_t size = HxW / K * K;
      for (int64_t j = 0; j < D; ++j) {
        const int64_t c = g * D + j;
        const T scale = rstd_val * (gamma_null ? T(1) : gamma_data[c]);
        const T bias = -scale * mean_val + (beta_null ? T(0) : beta_data[c]);
        X_ptr = X_data + (i * D + j) * HxW;
        T* Y_ptr = Y_data + (i * D + j) * HxW;
        for (int64_t k = 0; k < size; k += K) {
          const Vec x_vec = Vec::loadu(X_ptr + k);
          Vec y_vec = Vec(scale) * x_vec + Vec(bias);
          y_vec.store(Y_ptr + k);
        }
        for (int64_t k = size; k < HxW; ++k) {
          Y_ptr[k] = scale * X_ptr[k] + bias;
        }
      }
      mean_data[i] = mean_val;
      rstd_data[i] = rstd_val;
    }
  });
}

template <typename T>
void GroupNormKernelImplChannelsLastInternal(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    T eps,
    Tensor& Y,
    Tensor& mean,
    Tensor& rstd) {
  TORCH_CHECK(X.numel() == N * C * HxW);
  TORCH_CHECK(!gamma.defined() || gamma.numel() == C);
  TORCH_CHECK(!beta.defined() || beta.numel() == C);
  const int64_t G = group;
  const int64_t D = C / G;
  const T* X_data = X.data_ptr<T>();
  const T* gamma_data = gamma.defined() ? gamma.data_ptr<T>() : nullptr;
  const T* beta_data = beta.defined() ? beta.data_ptr<T>() : nullptr;
  T* Y_data = Y.data_ptr<T>();
  T* mean_data = mean.data_ptr<T>();
  T* rstd_data = rstd.data_ptr<T>();
  const T s = T(1) / static_cast<T>(D * HxW);
  const bool gamma_null = (gamma_data == nullptr);
  const bool beta_null = beta_data == nullptr;

  const bool is_gamma_beta_valid = !gamma_null && !beta_null;

  // temp buffer holding x and x2
  Tensor buffer = at::empty({N, 2 * C}, X.options()).zero_();
  T* buffer_data = buffer.data_ptr<T>();

  using Vec = vec256::Vec256<T>;
  at::parallel_for(0, N, 1, [&](int64_t start, int64_t end) {
    constexpr int64_t K = Vec::size();
    const int64_t inner_size = C / K * K;
    for (int64_t n = start; n < end; ++n) {
      T* mean_ptr = buffer_data + n * 2 * C;
      T* rstd_ptr = mean_ptr + C;
      for (int64_t i = 0; i < HxW; ++i) {
        const T* X_ptr = X_data + n * HxW * C + i * C;
        for (int64_t j = 0; j < inner_size; j += K) {
          const Vec x_vec = Vec::loadu(X_ptr + j);
          Vec mean_vec = Vec::loadu(mean_ptr + j) + x_vec;
          Vec rstd_vec = Vec::loadu(rstd_ptr + j) + x_vec * x_vec;
          mean_vec.store(mean_ptr + j);
          rstd_vec.store(rstd_ptr + j);
        }
        for (int64_t j = inner_size; j < C; ++j) {
          mean_ptr[j] += X_ptr[j];
          rstd_ptr[j] += X_ptr[j] * X_ptr[j];
        }
      }

      for (int64_t g = 0; g < G; ++g) {
        T mean_val = T(0);
        T rstd_val = T(0);
        for (int64_t d = 0; d < D; ++d) {
          mean_val += mean_ptr[g * D + d];
          rstd_val += rstd_ptr[g * D + d];
        }
        mean_val *= s;
        rstd_val = std::max(rstd_val * s - mean_val * mean_val, T(0));
        rstd_val = T(1) / std::sqrt(rstd_val + eps);

        // continue to use the temp buffer for mean and rstd value,
        // so that we can vectorize the following math on entire C dimension.
        for (int64_t d = 0; d < D; ++d) {
          mean_ptr[g * D + d] = mean_val;
          rstd_ptr[g * D + d] = rstd_val;
        }

        mean_data[n * G + g] = mean_val;
        rstd_data[n * G + g] = rstd_val;
      }

      // expand gamma_null and beta_null to reduce if-else on critial path.
      if (!gamma_null && !beta_null) {
        for (int64_t i = 0; i < HxW; ++i) {
          const T* X_ptr = X_data + n * HxW * C + i * C;
          T* Y_ptr = Y_data + n * HxW * C + i * C;
          for (int64_t j = 0; j < inner_size; j += K) {
            Vec scale_vec = Vec::loadu(rstd_ptr + j) * Vec::loadu(gamma_data + j);
            Vec bias_vec = Vec::loadu(beta_data + j) - scale_vec * Vec::loadu(mean_ptr + j);
            Vec y_vec = scale_vec * Vec::loadu(X_ptr + j) + bias_vec;
            y_vec.store(Y_ptr + j);
          }
          for (int64_t j = inner_size; j < C; ++j) {
            T scale = rstd_ptr[j] * gamma_data[j];
            T bias = -scale * mean_ptr[j] + beta_data[j];
            Y_ptr[j] = scale * X_ptr[j] + bias;
          }
        }
      } else if (gamma_null && beta_null) {
        for (int64_t i = 0; i < HxW; ++i) {
          const T* X_ptr = X_data + n * HxW * C + i * C;
          T* Y_ptr = Y_data + n * HxW * C + i * C;
          for (int64_t j = 0; j < inner_size; j += K) {
            Vec scale_vec = Vec::loadu(rstd_ptr + j);
            Vec y_vec = scale_vec * Vec::loadu(X_ptr + j) - scale_vec * Vec::loadu(mean_ptr + j);
            y_vec.store(Y_ptr + j);
          }
          for (int64_t j = inner_size; j < C; ++j) {
            T scale = rstd_ptr[j];
            Y_ptr[j] = scale * X_ptr[j] -scale * mean_ptr[j];
          }
        }
      } else {
        for (int64_t i = 0; i < HxW; ++i) {
          const T* X_ptr = X_data + n * HxW * C + i * C;
          T* Y_ptr = Y_data + n * HxW * C + i * C;
          for (int64_t j = 0; j < inner_size; j += K) {
            Vec gamma_vec = gamma_null ? Vec(1) : Vec::loadu(gamma_data + j);
            Vec beta_vec = beta_null ? Vec(0) : Vec::loadu(beta_data + j);
            Vec scale_vec = Vec::loadu(rstd_ptr + j) * gamma_vec;
            Vec bias_vec = beta_vec - scale_vec * Vec::loadu(mean_ptr + j);
            Vec y_vec = scale_vec * Vec::loadu(X_ptr + j) + bias_vec;
            y_vec.store(Y_ptr + j);
          }
          for (int64_t j = inner_size; j < C; ++j) {
            T scale = rstd_ptr[j] * (gamma_null ? T(1) : gamma_data[j]);
            T bias = -scale * mean_ptr[j] + (beta_null ? T(0) : beta_data[j]);
            Y_ptr[j] = scale * X_ptr[j] + bias;
          }
        }
      }
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
    Tensor& Y,
    Tensor& mean,
    Tensor& rstd) {
  switch (X.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "GroupNormKernelImpl", [&]() {
        GroupNormKernelImplInternal<scalar_t>(
            X, gamma, beta, N, C, HxW, group, static_cast<scalar_t>(eps), Y, mean, rstd);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "GroupNormKernelImpl", [&]() {
        GroupNormKernelImplChannelsLastInternal<scalar_t>(
            X, gamma, beta, N, C, HxW, group, static_cast<scalar_t>(eps), Y, mean, rstd);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

template <typename T>
void ComputeInternalGradients(
    int64_t N,
    int64_t C,
    int64_t HxW,
    const T* dY,
    const T* X,
    T* ds,
    T* db) {
  at::parallel_for(0, N * C, 1, [=](int64_t start, int64_t end) {
    constexpr int64_t K = vec256::Vec256<T>::size();
    const int64_t inner_size = HxW / K * K;
    std::array<T, K> ds_arr;
    std::array<T, K> db_arr;
    for (int64_t i = start; i < end; ++i) {
      const T* dY_ptr = dY + i * HxW;
      const T* X_ptr = X + i * HxW;
      vec256::Vec256<T> ds_vec(0);
      vec256::Vec256<T> db_vec(0);
      for (int64_t j = 0; j < inner_size; j += K) {
        const vec256::Vec256<T> dy_vec = vec256::Vec256<T>::loadu(dY_ptr + j);
        const vec256::Vec256<T> x_vec = vec256::Vec256<T>::loadu(X_ptr + j);
        ds_vec = ds_vec + dy_vec * x_vec;
        db_vec = db_vec + dy_vec;
      }
      ds_vec.store(ds_arr.data());
      db_vec.store(db_arr.data());
      T ds_val = std::accumulate(ds_arr.cbegin(), ds_arr.cend(), T(0));
      T db_val = std::accumulate(db_arr.cbegin(), db_arr.cend(), T(0));
      for (int64_t j = inner_size; j < HxW; ++j) {
        ds_val += dY_ptr[j] * X_ptr[j];
        db_val += dY_ptr[j];
      }
      ds[i] = ds_val;
      db[i] = db_val;
    }
  });
}

template <typename T>
void GroupNormInputBackward(
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    const T* dY,
    const T* X,
    const T* mean,
    const T* rstd,
    const T* gamma,
    const T* ds,
    const T* db,
    T* dX) {
  const int64_t G = group;
  const int64_t D = C / G;
  const T s = T(1) / static_cast<T>(D * HxW);
  const bool gamma_null = (gamma == nullptr);
  at::parallel_for(0, N * G, 1, [=](int64_t start, int64_t end) {
    constexpr int64_t K = vec256::Vec256<T>::size();
    const int64_t d = D / K * K;
    std::array<T, K> ds_arr;
    std::array<T, K> db_arr;
    for (int64_t i = start; i < end; ++i) {
      const int64_t g = i % G;
      const T* ds_ptr = ds + i * D;
      const T* db_ptr = db + i * D;
      vec256::Vec256<T> ds_vec(0);
      vec256::Vec256<T> db_vec(0);
      for (int64_t j = 0; j < d; j += K) {
        const vec256::Vec256<T> gamma_vec = gamma_null
            ? vec256::Vec256<T>(1)
            : vec256::Vec256<T>::loadu(gamma + g * D + j);
        ds_vec = ds_vec + vec256::Vec256<T>::loadu(ds_ptr + j) * gamma_vec;
        db_vec = db_vec + vec256::Vec256<T>::loadu(db_ptr + j) * gamma_vec;
      }
      ds_vec.store(ds_arr.data());
      db_vec.store(db_arr.data());
      T ds_val = std::accumulate(ds_arr.cbegin(), ds_arr.cend(), T(0));
      T db_val = std::accumulate(db_arr.cbegin(), db_arr.cend(), T(0));
      for (int64_t j = d; j < D; ++j) {
        const T gamma_v = gamma_null ? T(1) : gamma[g * D + j];
        ds_val += ds_ptr[j] * gamma_v;
        db_val += db_ptr[j] * gamma_v;
      }
      const T c2 =
          (db_val * mean[i] - ds_val) * rstd[i] * rstd[i] * rstd[i] * s;
      const T c3 = -c2 * mean[i] - db_val * rstd[i] * s;
      for (int64_t j = 0; j < D; ++j) {
        const int64_t c = g * D + j;
        const T* dY_ptr = dY + (i * D + j) * HxW;
        const T* X_ptr = X + (i * D + j) * HxW;
        T* dX_ptr = dX + (i * D + j) * HxW;
        const T c1 = rstd[i] * (gamma_null ? T(1) : gamma[c]);
        for (int64_t k = 0; k < HxW; ++k) {
          dX_ptr[k] = c1 * dY_ptr[k] + c2 * X_ptr[k] + c3;
        }
      }
    }
  });
}

template <typename T>
void GammaBackward(
    int64_t N,
    int64_t C,
    int64_t group,
    const T* mean,
    const T* rstd,
    const T* ds,
    const T* db,
    T* dgamma) {
  const int64_t G = group;
  const int64_t D = C / G;
  constexpr int64_t K = vec256::Vec256<T>::size();
  at::parallel_for(0, D, K, [=](int64_t start, int64_t end) {
    for (int64_t i = 0; i < G; ++i) {
      std::memset(dgamma + i * D + start, 0, (end - start) * sizeof(T));
    }
    for (int64_t i = 0; i < N * G; ++i) {
      const T* ds_ptr = ds + i * D;
      const T* db_ptr = db + i * D;
      const int64_t g = i % G;
      for (int64_t j = start; j < end; ++j) {
        const int64_t c = g * D + j;
        dgamma[c] += (ds_ptr[j] - db_ptr[j] * mean[i]) * rstd[i];
      }
    }
  });
}

template <typename T>
void BetaBackward(int64_t N, int64_t C, const T* db, T* dbeta) {
  constexpr int64_t K = vec256::Vec256<T>::size();
  at::parallel_for(0, C, K, [=](int64_t start, int64_t end) {
    std::memset(dbeta + start, 0, (end - start) * sizeof(T));
    for (int64_t i = 0; i < N; ++i) {
      const T* db_ptr = db + i * C;
      for (int64_t j = start; j < end; ++j) {
        dbeta[j] += db_ptr[j];
      }
    }
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
    Tensor& dX,
    Tensor& dgamma,
    Tensor& dbeta) {
  TORCH_CHECK(dY.numel() == N * C * HxW);
  TORCH_CHECK(X.numel() == N * C * HxW);
  TORCH_CHECK(mean.numel() == N * group);
  TORCH_CHECK(rstd.numel() == N * group);
  TORCH_CHECK(!gamma.defined() || gamma.numel() == C);

  const T* dY_data = dY.data_ptr<T>();
  const T* X_data = X.data_ptr<T>();
  const T* mean_data = mean.data_ptr<T>();
  const T* rstd_data = rstd.data_ptr<T>();
  const T* gamma_data = gamma.defined() ? gamma.data_ptr<T>() : nullptr;
  T* dX_data = dX.defined() ? dX.data_ptr<T>() : nullptr;
  T* dgamma_data = dgamma.defined() ? dgamma.data_ptr<T>() : nullptr;
  T* dbeta_data = dbeta.defined() ? dbeta.data_ptr<T>() : nullptr;
  Tensor ds = at::empty({N, C}, X.options());
  Tensor db = at::empty({N, C}, X.options());
  T* ds_data = ds.data_ptr<T>();
  T* db_data = db.data_ptr<T>();

  ComputeInternalGradients<T>(N, C, HxW, dY_data, X_data, ds_data, db_data);

  if (dX_data != nullptr) {
    GroupNormInputBackward<T>(
        N,
        C,
        HxW,
        group,
        dY_data,
        X_data,
        mean_data,
        rstd_data,
        gamma_data,
        ds_data,
        db_data,
        dX_data);
  }
  if (dgamma_data != nullptr) {
    GammaBackward<T>(
        N, C, group, mean_data, rstd_data, ds_data, db_data, dgamma_data);
  }
  if (dbeta_data != nullptr) {
    BetaBackward<T>(N, C, db_data, dbeta_data);
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
    Tensor& dX,
    Tensor& dgamma,
    Tensor& dbeta) {
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
