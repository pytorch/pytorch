#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/group_norm.h>

#include <algorithm>
#include <array>
#include <numeric>

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/native/cpu/moments_utils.h>
#include <ATen/native/cpu/mixed_data_type.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

namespace at {
namespace native {

namespace {

template <typename T, typename PT>
void GroupNormKernelImplInternal(
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
  TORCH_CHECK(X.numel() == N * C * HxW);
  TORCH_CHECK(!gamma.defined() || gamma.numel() == C);
  TORCH_CHECK(!beta.defined() || beta.numel() == C);
  const int64_t G = group;
  const int64_t D = C / G;
  const T* X_data = X.data_ptr<T>();
  const PT* gamma_data = gamma.defined() ? gamma.data_ptr<PT>() : nullptr;
  const PT* beta_data = beta.defined() ? beta.data_ptr<PT>() : nullptr;
  T* Y_data = Y.data_ptr<T>();
  PT* mean_data = mean.data_ptr<PT>();
  PT* rstd_data = rstd.data_ptr<PT>();
  const bool gamma_null = (gamma_data == nullptr);
  const bool beta_null = beta_data == nullptr;
  const int64_t inner_size = D * HxW;

  using T_ACC = vec::vec_scalar_t<T>;

  at::parallel_for(0, N * G, 1, [&](int64_t start, int64_t end) {
    for (const auto i : c10::irange(start, end)) {
      const T* X_ptr = X_data + i * inner_size;
      T_ACC mean_val;
      T_ACC rstd_val;
      std::tie(mean_val, rstd_val) = RowwiseMoments(X_ptr, inner_size);
      rstd_val = T_ACC(1) / std::sqrt(std::max(rstd_val, T_ACC(0)) + eps);
      if (gamma_null && beta_null) {
        T* Y_ptr = Y_data + i * inner_size;
        for (const auto j : c10::irange(inner_size)) {
          Y_ptr[j] = (X_ptr[j] - mean_val) * rstd_val;
        }
      } else {
        const int64_t g = i % G;
        for (const auto j : c10::irange(D)) {
          const int64_t c = g * D + j;
          const T_ACC scale = rstd_val * (gamma_null ? PT(1) : gamma_data[c]);
          const T_ACC bias = -scale * mean_val + (beta_null ? PT(0) : beta_data[c]);
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
std::tuple<T, T> ColumnwiseMoments(
    const T* X_data,
    int64_t HxW,
    int64_t C,
    int64_t D) {
  using Vec = vec::Vectorized<T>;
  constexpr int64_t K = Vec::size();
  const int64_t inner_size = D / K * K;
  Vec acc0_vec{0}, acc1_vec{0};
  for (const auto m : c10::irange(HxW)) {
    const T* X_ptr = X_data + m * C;
    int64_t d = 0;
    for (; d < inner_size; d += K) {
      Vec x_vec = Vec::loadu(X_ptr + d);
      acc0_vec += x_vec;
      acc1_vec += x_vec * x_vec;
    }
    if (D - d > 0) {
      Vec x_vec = Vec::loadu(X_ptr + d, D - d);
      acc0_vec += x_vec;
      acc1_vec += x_vec * x_vec;
    }
  }
  T mean_val = vec::vec_reduce_all([](Vec& x, Vec& y) { return x + y; }, acc0_vec);
  T rstd_val = vec::vec_reduce_all([](Vec& x, Vec& y) { return x + y; }, acc1_vec);
  return std::tuple<T, T>(mean_val, rstd_val);
}

template <typename T = BFloat16>
std::tuple<float, float> ColumnwiseMoments(
    const BFloat16* X_data,
    int64_t HxW,
    int64_t C,
    int64_t D) {
  using bVec = vec::Vectorized<BFloat16>;
  using fVec = vec::Vectorized<float>;
  constexpr int64_t K = bVec::size();
  const int64_t inner_size = D / K * K;
  fVec acc0_fvec{0}, acc1_fvec{0}, zero{0};
  for (const auto m : c10::irange(HxW)) {
    const BFloat16* X_ptr = X_data + m * C;
    int64_t d = 0;
    for (; d < inner_size; d += K) {
      bVec x_bvec = bVec::loadu(X_ptr + d);
      fVec x_fvec0, x_fvec1;
      std::tie(x_fvec0, x_fvec1) = convert_bfloat16_float(x_bvec);
      acc0_fvec += x_fvec0 + x_fvec1;
      acc1_fvec += x_fvec0 * x_fvec0 + x_fvec1 * x_fvec1;
    }
    if (D - d > 0) {
      bVec x_bvec = bVec::loadu(X_ptr + d, D - d);
      fVec x_fvec0, x_fvec1;
      std::tie(x_fvec0, x_fvec1) = convert_bfloat16_float(x_bvec);
      if (D - d > fVec::size()) {
        x_fvec1 = fVec::set(zero, x_fvec1, D - d - fVec::size());
        acc0_fvec += x_fvec0 + x_fvec1;
        acc1_fvec += x_fvec0 * x_fvec0 + x_fvec1 * x_fvec1;
      } else {
        x_fvec0 = fVec::set(zero, x_fvec0, D - d);
        acc0_fvec += x_fvec0;
        acc1_fvec += x_fvec0 * x_fvec0;
      }
    }
  }
  float mean_val = vec::vec_reduce_all([](fVec& x, fVec& y) { return x + y; }, acc0_fvec);
  float rstd_val = vec::vec_reduce_all([](fVec& x, fVec& y) { return x + y; }, acc1_fvec);
  return std::tuple<float, float>(mean_val, rstd_val);
}

template <typename T, typename PT>
void GroupNormKernelImplChannelsLastInternal(
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
  TORCH_CHECK(X.numel() == N * C * HxW);
  TORCH_CHECK(!gamma.defined() || gamma.numel() == C);
  TORCH_CHECK(!beta.defined() || beta.numel() == C);
  const int64_t G = group;
  const int64_t D = C / G;
  const T* X_data = X.data_ptr<T>();
  const PT* gamma_data = gamma.defined() ? gamma.data_ptr<PT>() : nullptr;
  const PT* beta_data = beta.defined() ? beta.data_ptr<PT>() : nullptr;
  T* Y_data = Y.data_ptr<T>();
  PT* mean_data = mean.data_ptr<PT>();
  PT* rstd_data = rstd.data_ptr<PT>();

  using T_ACC = vec::vec_scalar_t<T>;
  using Vec = vec::Vectorized<T_ACC>;

  const T s = T_ACC(1) / static_cast<T_ACC>(D * HxW);
  const bool gamma_null = (gamma_data == nullptr);
  const bool beta_null = beta_data == nullptr;

  // NB: About algorithm choosen:
  //
  // On channels last, GroupNorm has a input shape of {N, H, W, GD},
  // Mean and rstd are collected per each n and g, which involves reduction
  // on non-adjacent dimensions. We can parallel in the following 2 impls:
  //
  // impl-1: parallel on N * G. Only need one omp session but memory access
  //   per thread is non-contiguous.
  //
  // impl-2: parallel on N * HxW. Memory access per thread is contiguous,
  //   but requires help of extra temp buffer of size {T, N, 2C}.
  //
  // Generally impl-2 has better performance when HxW is large enough, so that
  //   data per thread {NHWC / T} is much larger then temp buffer per thread {2NC}
  //
  constexpr int64_t feature_map_threshold = 1024;
  if (HxW < feature_map_threshold) {
    // impl-1: parallel on N * G.
    //
    // for each plain of HxW, scale and bias is calculated only once
    Tensor buffer = at::empty({N * G, 2 * D}, X.options());
    T* buffer_data = buffer.data_ptr<T>();

    at::parallel_for(0, N * G, 1, [&](int64_t begin, int64_t end) {
      int64_t n{0}, g{0};
      data_index_init(begin, n, N, g, G);
      for (const auto i : c10::irange(begin, end)) {
        // step-1: for each n and g, collect sum of x and x2
        //
        // Note that using vec::map_reduce_all here is simpler to write
        // but it is slower since horizontal reduce from vec to scalar is slow.
        // So it is better to reduce with a vec across all HxW plain,
        // and do a horizontal add just once for each {n, g}.
        //
        T_ACC mean_val, rstd_val;
        std::tie(mean_val, rstd_val) = ColumnwiseMoments(
                X_data + n * HxW * C + g * D,
                HxW,
                C,
                D);

        mean_val *= s;
        rstd_val = std::max(rstd_val * s - mean_val * mean_val, T_ACC(0));
        rstd_val = T_ACC(1) / std::sqrt(rstd_val + eps);
        mean_data[i] = mean_val;
        rstd_data[i] = rstd_val;

        // step-2: calculate scale and bias
        T* scale_ptr = buffer_data + i * 2 * D;
        T* bias_ptr = scale_ptr + D;
        for (const auto d : c10::irange(D)) {
          const int64_t c = g * D + d;
          scale_ptr[d] = rstd_val * (gamma_null ? PT(1) : gamma_data[c]);
          bias_ptr[d] = -scale_ptr[d] * mean_val + (beta_null ? PT(0) : beta_data[c]);
        }

        // step-3: apply scale and bias
        for (const auto m : c10::irange(HxW)) {
          const T* X_ptr = X_data + n * HxW * C + m * C + g * D;
          T* Y_ptr = Y_data + n * HxW * C + m * C + g * D;
          vec::map3<T>(
              [](Vec x, Vec scale, Vec bias) { return x * scale + bias; },
              Y_ptr,
              X_ptr,
              scale_ptr,
              bias_ptr,
              D);
        }

        data_index_step(n, N, g, G);
      }
    });
  } else {
    // impl-2: parallel on N * HxW.
    //
    // temp buffer holding x and x2
    int num_threads = at::get_num_threads();
    Tensor buffer = at::empty({num_threads, N, 2 * C}, X.options()).zero_();
    T* buffer_data = buffer.data_ptr<T>();

    // step-1: accumulate on dimension of C
    //
    // In order to improve multi-core performance when N=1,
    // we parallel on the all the outer dimensions of N and HxW,
    // leaving the most inner dimension C for vectorization.
    //
    // Note that parallel on {N, HxW, G} is not feasible for some common configs,
    // e.g. say input shape is {1, 32, h, w} and G = 8,
    //   this will give D = 4 which is unable to take full SIMD length.
    //
    // To avoid thread conflict, we make use of a temp buffer of {T, N, 2C},
    //   firstly, reduce from {N, HxW, C} to {T, N, 2C}
    //
    at::parallel_for(0, N * HxW, 1, [&](int64_t begin, int64_t end) {
      int tid = at::get_thread_num();
      T* buffer_ptr = buffer_data + tid * N * 2 * C;

      int64_t n{0}, m{0};
      data_index_init(begin, n, N, m, HxW);
      for (const auto i : c10::irange(begin, end)) {
        T* mean_ptr = buffer_ptr + n * 2 * C;
        T* rstd_ptr = mean_ptr + C;
        const T* X_ptr = X_data + i * C;

        vec::map2<T>(
            [](Vec x, Vec y) { return x + y; },
            mean_ptr,
            X_ptr,
            mean_ptr,
            C);

        vec::map2<T>(
            [](Vec x, Vec y) { return x * x + y; },
            rstd_ptr,
            X_ptr,
            rstd_ptr,
            C);

        data_index_step(n, N, m, HxW);
      }
    });

    // step-2: compute mean and rstd
    for (const auto n : c10::irange(N)) {
      for (const auto g : c10::irange(G)) {
        T_ACC mean_val{0}, rstd_val{0};
        for (const auto d : c10::irange(D)) {
          for (const auto t : c10::irange(num_threads)) {
            T* buffer_ptr = buffer_data + t * N * 2 * C + n * 2 * C;
            mean_val += buffer_ptr[g * D + d];
            rstd_val += buffer_ptr[g * D + d + C];
           }
        }
        mean_val *= s;
        rstd_val = std::max(rstd_val * s - mean_val * mean_val, T_ACC(0));
        rstd_val = T_ACC(1) / std::sqrt(rstd_val + eps);
        mean_data[n * G + g] = T(mean_val);
        rstd_data[n * G + g] = T(rstd_val);
      }
    }

    // step-3: compute scale and bias
    //
    // mean/rstd have shape of {N, G}, gamma/beta have shape of {G, D}.
    // And scale/bias have shape of {N, C} so that we can directly vectorize on
    // dimension of C in the final step.
    //
    // We could fuse step 3 and 4 into a single session but this way is better:
    //   a. D might be too small for vectorization;
    //   b. Avoid duplicate caculation of scale/bias, each HxW plain share the same scale/bias
    //
    for (const auto n : c10::irange(N)) {
      for (const auto g : c10::irange(G)) {
        T* scale_ptr = buffer_data + n * 2 * C;
        T* bias_ptr = scale_ptr + C;
        T mean_val = mean_data[n * G + g];
        T rstd_val = rstd_data[n * G + g];
        for (const auto d : c10::irange(D)) {
          const int64_t c = g * D + d;
          scale_ptr[c] = rstd_val * (gamma_null ? PT(1) : gamma_data[c]);
          bias_ptr[c] = -scale_ptr[c] * mean_val + (beta_null ? PT(0) : beta_data[c]);
        }
      }
    }

    // step-4: apply scale and bias
    //
    // Parallel on on the all the outer dimensions of N and HxW
    // and vectorize on C.
    //
    at::parallel_for(0, N * HxW, 1, [&](int64_t begin, int64_t end) {
      int64_t n{0}, m{0};
      data_index_init(begin, n, N, m, HxW);
      for (const auto i : c10::irange(begin, end)) {
        const T* X_ptr = X_data + i * C;
        T* Y_ptr = Y_data + i * C;
        T* scale_ptr = buffer_data + n * 2 * C;
        T* bias_ptr = scale_ptr + C;
        vec::map3<T>(
            [](Vec x, Vec scale, Vec bias) { return x * scale + bias; },
            Y_ptr,
            X_ptr,
            scale_ptr,
            bias_ptr,
            C);

        data_index_step(n, N, m, HxW);
      }
    });
  }
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
  const bool mixed_type = is_mixed_type(X, gamma, beta);
  switch (X.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, X.scalar_type(), "GroupNormKernelImpl", [&]() {
        if (mixed_type) {
          GroupNormKernelImplInternal<BFloat16, float>(X, gamma, beta, N, C, HxW, group, eps, Y, mean, rstd);
        } else {
          GroupNormKernelImplInternal<scalar_t, scalar_t>(X, gamma, beta, N, C, HxW, group, eps, Y, mean, rstd);
        }
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast:
    case at::MemoryFormat::ChannelsLast3d: {
      AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, X.scalar_type(), "GroupNormKernelImpl", [&]() {
        if (mixed_type) {
          GroupNormKernelImplChannelsLastInternal<BFloat16, float>(X, gamma, beta, N, C, HxW, group, eps, Y, mean, rstd);
        } else {
          GroupNormKernelImplChannelsLastInternal<scalar_t, scalar_t>(X, gamma, beta, N, C, HxW, group, eps, Y, mean, rstd);
        }
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, ChannelsLast3d, Contiguous");
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
  AT_DISPATCH_FLOATING_TYPES_AND(
      ScalarType::BFloat16, X.scalar_type(), "GroupNormBackwardKernelImpl", [&]() {
        GroupNormBackwardKernelImplInternal<scalar_t>(
            dY, X, mean, rstd, gamma, N, C, HxW, group, dX, dgamma, dbeta);
      });
}

} // namespace

REGISTER_DISPATCH(GroupNormKernel, &GroupNormKernelImpl);
REGISTER_DISPATCH(GroupNormBackwardKernel, &GroupNormBackwardKernelImpl);

} // namespace native
} // namespace at
