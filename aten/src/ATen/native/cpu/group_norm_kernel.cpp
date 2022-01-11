#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/group_norm.h>

#include <algorithm>
#include <array>
#include <numeric>

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/cpu/moments_utils.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

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
  const bool gamma_null = (gamma_data == nullptr);
  const bool beta_null = beta_data == nullptr;
  const int64_t inner_size = D * HxW;

  at::parallel_for(0, N * G, 1, [&](int64_t start, int64_t end) {
    for (const auto i : c10::irange(start, end)) {
      const T* X_ptr = X_data + i * inner_size;
      T mean_val;
      T rstd_val;
      std::tie(mean_val, rstd_val) = utils::RowwiseMoments(X_ptr, inner_size);
      rstd_val = T(1) / std::sqrt(std::max(rstd_val, T(0)) + eps);
      if (gamma_null && beta_null) {
        T* Y_ptr = Y_data + i * inner_size;
        for (const auto j : c10::irange(inner_size)) {
          Y_ptr[j] = (X_ptr[j] - mean_val) * rstd_val;
        }
      } else {
        const int64_t g = i % G;
        for (const auto j : c10::irange(D)) {
          const int64_t c = g * D + j;
          const T scale = rstd_val * (gamma_null ? T(1) : gamma_data[c]);
          const T bias = -scale * mean_val + (beta_null ? T(0) : beta_data[c]);
          X_ptr = X_data + (i * D + j) * HxW;
          T* Y_ptr = Y_data + (i * D + j) * HxW;
          for (const auto k : c10::irange(HxW)) {
            Y_ptr[k] = scale * X_ptr[k] + bias;
          }
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

  // temp buffer holding x and x2
  Tensor buffer = at::empty({N, 2 * C}, X.options()).zero_();
  T* buffer_data = buffer.data_ptr<T>();

  using Vec = vec::Vectorized<T>;
  at::parallel_for(0, N, 1, [&](int64_t start, int64_t end) {
    constexpr int64_t K = Vec::size();
    const int64_t inner_size = C / K * K;
    for (const auto n : c10::irange(start, end)) {
      T* mean_ptr = buffer_data + n * 2 * C;
      T* rstd_ptr = mean_ptr + C;
      for (const auto i : c10::irange(HxW)) {
        const T* X_ptr = X_data + n * HxW * C + i * C;
        for (int64_t j = 0; j < inner_size; j += K) {
          const Vec x_vec = Vec::loadu(X_ptr + j);
          Vec mean_vec = Vec::loadu(mean_ptr + j) + x_vec;
          Vec rstd_vec = Vec::loadu(rstd_ptr + j) + x_vec * x_vec;
          mean_vec.store(mean_ptr + j);
          rstd_vec.store(rstd_ptr + j);
        }
        for (const auto j : c10::irange(inner_size, C)) {
          mean_ptr[j] += X_ptr[j];
          rstd_ptr[j] += X_ptr[j] * X_ptr[j];
        }
      }

      for (const auto g : c10::irange(G)) {
        T mean_val = T(0);
        T rstd_val = T(0);
        for (const auto d : c10::irange(D)) {
          mean_val += mean_ptr[g * D + d];
          rstd_val += rstd_ptr[g * D + d];
        }
        mean_val *= s;
        rstd_val = std::max(rstd_val * s - mean_val * mean_val, T(0));
        rstd_val = T(1) / std::sqrt(rstd_val + eps);

        // continue to use the temp buffer for mean and rstd value,
        // so that we can vectorize the following math on entire C dimension.
        for (const auto d : c10::irange(D)) {
          mean_ptr[g * D + d] = mean_val;
          rstd_ptr[g * D + d] = rstd_val;
        }

        mean_data[n * G + g] = mean_val;
        rstd_data[n * G + g] = rstd_val;
      }

      // expand gamma_null and beta_null to reduce if-else on critial path.
      if (!gamma_null && !beta_null) {
        for (const auto i : c10::irange(HxW)) {
          const T* X_ptr = X_data + n * HxW * C + i * C;
          T* Y_ptr = Y_data + n * HxW * C + i * C;
          for (int64_t j = 0; j < inner_size; j += K) {
            Vec scale_vec = Vec::loadu(rstd_ptr + j) * Vec::loadu(gamma_data + j);
            Vec bias_vec = Vec::loadu(beta_data + j) - scale_vec * Vec::loadu(mean_ptr + j);
            Vec y_vec = scale_vec * Vec::loadu(X_ptr + j) + bias_vec;
            y_vec.store(Y_ptr + j);
          }
          for (const auto j : c10::irange(inner_size, C)) {
            T scale = rstd_ptr[j] * gamma_data[j];
            T bias = -scale * mean_ptr[j] + beta_data[j];
            Y_ptr[j] = scale * X_ptr[j] + bias;
          }
        }
      } else if (gamma_null && beta_null) {
        for (const auto i : c10::irange(HxW)) {
          const T* X_ptr = X_data + n * HxW * C + i * C;
          T* Y_ptr = Y_data + n * HxW * C + i * C;
          for (int64_t j = 0; j < inner_size; j += K) {
            Vec scale_vec = Vec::loadu(rstd_ptr + j);
            Vec y_vec = scale_vec * Vec::loadu(X_ptr + j) - scale_vec * Vec::loadu(mean_ptr + j);
            y_vec.store(Y_ptr + j);
          }
          for (const auto j : c10::irange(inner_size, C)) {
            T scale = rstd_ptr[j];
            Y_ptr[j] = scale * X_ptr[j] -scale * mean_ptr[j];
          }
        }
      } else {
        for (const auto i : c10::irange(HxW)) {
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
          for (const auto j : c10::irange(inner_size, C)) {
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
    constexpr int64_t K = vec::Vectorized<T>::size();
    const int64_t inner_size = HxW / K * K;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    std::array<T, K> ds_arr;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    std::array<T, K> db_arr;
    for (const auto i : c10::irange(start, end)) {
      const T* dY_ptr = dY + i * HxW;
      const T* X_ptr = X + i * HxW;
      vec::Vectorized<T> ds_vec(0);
      vec::Vectorized<T> db_vec(0);
      for (int64_t j = 0; j < inner_size; j += K) {
        const vec::Vectorized<T> dy_vec = vec::Vectorized<T>::loadu(dY_ptr + j);
        const vec::Vectorized<T> x_vec = vec::Vectorized<T>::loadu(X_ptr + j);
        ds_vec = ds_vec + dy_vec * x_vec;
        db_vec = db_vec + dy_vec;
      }
      ds_vec.store(ds_arr.data());
      db_vec.store(db_arr.data());
      T ds_val = std::accumulate(ds_arr.cbegin(), ds_arr.cend(), T(0));
      T db_val = std::accumulate(db_arr.cbegin(), db_arr.cend(), T(0));
      for (const auto j : c10::irange(inner_size, HxW)) {
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
    constexpr int64_t K = vec::Vectorized<T>::size();
    const int64_t d = D / K * K;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    std::array<T, K> ds_arr;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    std::array<T, K> db_arr;
    for (const auto i : c10::irange(start, end)) {
      const int64_t g = i % G;
      const T* ds_ptr = ds + i * D;
      const T* db_ptr = db + i * D;
      vec::Vectorized<T> ds_vec(0);
      vec::Vectorized<T> db_vec(0);
      for (int64_t j = 0; j < d; j += K) {
        const vec::Vectorized<T> gamma_vec = gamma_null
            ? vec::Vectorized<T>(1)
            : vec::Vectorized<T>::loadu(gamma + g * D + j);
        ds_vec = ds_vec + vec::Vectorized<T>::loadu(ds_ptr + j) * gamma_vec;
        db_vec = db_vec + vec::Vectorized<T>::loadu(db_ptr + j) * gamma_vec;
      }
      ds_vec.store(ds_arr.data());
      db_vec.store(db_arr.data());
      T ds_val = std::accumulate(ds_arr.cbegin(), ds_arr.cend(), T(0));
      T db_val = std::accumulate(db_arr.cbegin(), db_arr.cend(), T(0));
      for (const auto j : c10::irange(d, D)) {
        const T gamma_v = gamma_null ? T(1) : gamma[g * D + j];
        ds_val += ds_ptr[j] * gamma_v;
        db_val += db_ptr[j] * gamma_v;
      }
      const T c2 =
          (db_val * mean[i] - ds_val) * rstd[i] * rstd[i] * rstd[i] * s;
      const T c3 = -c2 * mean[i] - db_val * rstd[i] * s;
      for (const auto j : c10::irange(D)) {
        const int64_t c = g * D + j;
        const T* dY_ptr = dY + (i * D + j) * HxW;
        const T* X_ptr = X + (i * D + j) * HxW;
        T* dX_ptr = dX + (i * D + j) * HxW;
        const T c1 = rstd[i] * (gamma_null ? T(1) : gamma[c]);
        for (const auto k : c10::irange(HxW)) {
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
  constexpr int64_t K = vec::Vectorized<T>::size();
  at::parallel_for(0, D, K, [=](int64_t start, int64_t end) {
    for (const auto i : c10::irange(G)) {
      std::memset(dgamma + i * D + start, 0, (end - start) * sizeof(T));
    }
    for (int64_t i = 0; i < N * G; ++i) {
      const T* ds_ptr = ds + i * D;
      const T* db_ptr = db + i * D;
      const int64_t g = i % G;
      for (const auto j : c10::irange(start, end)) {
        const int64_t c = g * D + j;
        dgamma[c] += (ds_ptr[j] - db_ptr[j] * mean[i]) * rstd[i];
      }
    }
  });
}

template <typename T>
void BetaBackward(int64_t N, int64_t C, const T* db, T* dbeta) {
  constexpr int64_t K = vec::Vectorized<T>::size();
  at::parallel_for(0, C, K, [=](int64_t start, int64_t end) {
    std::memset(dbeta + start, 0, (end - start) * sizeof(T));
    for (const auto i : c10::irange(N)) {
      const T* db_ptr = db + i * C;
      for (const auto j : c10::irange(start, end)) {
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
