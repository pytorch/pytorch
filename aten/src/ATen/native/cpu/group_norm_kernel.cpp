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
          const T_ACC scale = rstd_val * (gamma_null ? T_ACC(1) : T_ACC(gamma_data[c]));
          const T_ACC bias = -scale * mean_val + (beta_null ? T_ACC(0) : T_ACC(beta_data[c]));
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

template <typename scalar_t, typename param_t>
inline void calc_mean_var(
  const scalar_t* X_ptr,
  param_t* mean_ptr,
  param_t* rstd_ptr,
  int64_t C) {
  using Vec = vec::Vectorized<vec::vec_scalar_t<param_t>>;
  vec::map2<scalar_t>(
          [](Vec x, Vec y) { return x + y; },
          mean_ptr,
          X_ptr,
          mean_ptr,
          C);
  vec::map2<scalar_t>(
      [](Vec x, Vec y) { return x * x + y; },
      rstd_ptr,
      X_ptr,
      rstd_ptr,
      C);
}

template <>
inline void calc_mean_var(
  const BFloat16* X_ptr,
  float* mean_ptr,
  float* rstd_ptr,
  int64_t C) {
  using fVec = vec::Vectorized<float>;
  using bVec = vec::Vectorized<BFloat16>;
  int64_t d = 0;
  for (; d < C - (C % bVec::size()); d += bVec::size()) {
    bVec data_bvec = bVec::loadu(X_ptr + d);
    fVec mean_fvec0 = fVec::loadu(mean_ptr + d);
    fVec mean_fvec1 = fVec::loadu(mean_ptr + d + fVec::size());
    fVec rstd_fvec0 = fVec::loadu(rstd_ptr + d);
    fVec rstd_fvec1 = fVec::loadu(rstd_ptr + d + fVec::size());
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    mean_fvec0 = data_fvec0 + mean_fvec0;
    mean_fvec1 = data_fvec1 + mean_fvec1;
    rstd_fvec0 = data_fvec0 * data_fvec0 + rstd_fvec0;
    rstd_fvec1 = data_fvec1 * data_fvec1 + rstd_fvec1;
    mean_fvec0.store(mean_ptr + d);
    mean_fvec1.store(mean_ptr + d + fVec::size());
    rstd_fvec0.store(rstd_ptr + d);
    rstd_fvec1.store(rstd_ptr + d + fVec::size());
  }
  if (C - d > 0) {
    bVec data_bvec = bVec::loadu(X_ptr + d, C - d);
    fVec mean_fvec0 = fVec::loadu(mean_ptr + d, (C - d) > fVec::size() ? fVec::size() : (C - d));
    fVec mean_fvec1 = fVec::loadu(mean_ptr + d + fVec::size(), (C - d) > fVec::size() ? (C - d - fVec::size()) : 0);
    fVec rstd_fvec0 = fVec::loadu(rstd_ptr + d, (C - d) > fVec::size() ? fVec::size() : (C - d));
    fVec rstd_fvec1 = fVec::loadu(rstd_ptr + d + fVec::size(), (C - d) > fVec::size() ? (C - d - fVec::size()) : 0);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    mean_fvec0 = data_fvec0 + mean_fvec0;
    mean_fvec1 = data_fvec1 + mean_fvec1;
    rstd_fvec0 = data_fvec0 * data_fvec0 + rstd_fvec0;
    rstd_fvec1 = data_fvec1 * data_fvec1 + rstd_fvec1;
    mean_fvec0.store(mean_ptr + d, (C - d) > fVec::size() ? fVec::size() : (C - d));
    mean_fvec1.store(mean_ptr + d + fVec::size(), (C - d) > fVec::size() ? (C - d - fVec::size()) : 0);
    rstd_fvec0.store(rstd_ptr + d, (C - d) > fVec::size() ? fVec::size() : (C - d));
    rstd_fvec1.store(rstd_ptr + d + fVec::size(), (C - d) > fVec::size() ? (C - d - fVec::size()) : 0);
  }
}

template <typename scalar_t, typename param_t>
inline void apply_scale_bias(
  scalar_t* Y_ptr,
  const scalar_t* X_ptr,
  const param_t* scale_ptr,
  const param_t* bias_ptr,
  int64_t C) {
  using Vec = vec::Vectorized<vec::vec_scalar_t<param_t>>;
  vec::map3<scalar_t>(
    [](Vec x, Vec scale, Vec bias) { return x * scale + bias; },
    Y_ptr,
    X_ptr,
    scale_ptr,
    bias_ptr,
    C);
}

template <>
inline void apply_scale_bias(
  BFloat16* Y_ptr,
  const BFloat16* X_ptr,
  const float* scale_ptr,
  const float* bias_ptr,
  int64_t C) {
  using fVec = vec::Vectorized<float>;
  using bVec = vec::Vectorized<BFloat16>;
  int64_t d = 0;
  for (; d < C - (C % bVec::size()); d += bVec::size()) {
    bVec data_bvec = bVec::loadu(X_ptr + d);
    fVec scale_fvec0 = fVec::loadu(scale_ptr + d);
    fVec scale_fvec1 = fVec::loadu(scale_ptr + d + fVec::size());
    fVec bias_fvec0 = fVec::loadu(bias_ptr + d);
    fVec bias_fvec1 = fVec::loadu(bias_ptr + d + fVec::size());
    fVec data_fvec0, data_fvec1, out0, out1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    out0 = data_fvec0 * scale_fvec0 + bias_fvec0;
    out1 = data_fvec1 * scale_fvec1 + bias_fvec1;
    convert_float_bfloat16(out0, out1).store(Y_ptr + d);
  }
  if (C - d > 0) {
    bVec data_bvec = bVec::loadu(X_ptr + d, C - d);
    fVec scale_fvec0 = fVec::loadu(scale_ptr + d, (C - d) > fVec::size() ? fVec::size() : (C - d));
    fVec scale_fvec1 = fVec::loadu(scale_ptr + d + fVec::size(), (C - d) > fVec::size() ? (C - d - fVec::size()) : 0);
    fVec bias_fvec0 = fVec::loadu(bias_ptr + d, (C - d) > fVec::size() ? fVec::size() : (C - d));
    fVec bias_fvec1 = fVec::loadu(bias_ptr + d + fVec::size(), (C - d) > fVec::size() ? (C - d - fVec::size()) : 0);
    fVec data_fvec0, data_fvec1, out0, out1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    out0 = data_fvec0 * scale_fvec0 + bias_fvec0;
    out1 = data_fvec1 * scale_fvec1 + bias_fvec1;
    convert_float_bfloat16(out0, out1).store(Y_ptr + d, C - d);
  }
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

  const T_ACC s = T_ACC(1) / static_cast<T_ACC>(D * HxW);
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
    Tensor buffer = at::empty({N * G, 2 * D}, X.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));
    T_ACC* buffer_data = buffer.data_ptr<T_ACC>();

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
        T_ACC* scale_ptr = buffer_data + i * 2 * D;
        T_ACC* bias_ptr = scale_ptr + D;
        for (const auto d : c10::irange(D)) {
          const int64_t c = g * D + d;
          scale_ptr[d] = rstd_val * (gamma_null ? T_ACC(1) : T_ACC(gamma_data[c]));
          bias_ptr[d] = -scale_ptr[d] * mean_val + (beta_null ? T_ACC(0) : T_ACC(beta_data[c]));
        }

        // step-3: apply scale and bias
        for (const auto m : c10::irange(HxW)) {
          const T* X_ptr = X_data + n * HxW * C + m * C + g * D;
          T* Y_ptr = Y_data + n * HxW * C + m * C + g * D;
          apply_scale_bias<T, T_ACC>(Y_ptr, X_ptr, scale_ptr, bias_ptr, D);
        }

        data_index_step(n, N, g, G);
      }
    });
  } else {
    // impl-2: parallel on N * HxW.
    //
    // temp buffer holding x and x2
    int num_threads = at::get_num_threads();
    Tensor buffer = at::empty({num_threads, N, 2 * C},
      X.options().dtype(c10::CppTypeToScalarType<T_ACC>::value)).zero_();
    T_ACC* buffer_data = buffer.data_ptr<T_ACC>();
    Tensor tmp_buffer = at::empty({N, 2 * G},
      X.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));
    T_ACC* tmp_buffer_data = tmp_buffer.data_ptr<T_ACC>();
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
      T_ACC* buffer_ptr = buffer_data + tid * N * 2 * C;

      int64_t n{0}, m{0};
      data_index_init(begin, n, N, m, HxW);
      for (const auto i : c10::irange(begin, end)) {
        T_ACC* mean_ptr = buffer_ptr + n * 2 * C;
        T_ACC* rstd_ptr = mean_ptr + C;
        const T* X_ptr = X_data + i * C;
        calc_mean_var<T, T_ACC>(X_ptr, mean_ptr, rstd_ptr, C);
        data_index_step(n, N, m, HxW);
      }
    });

    // step-2: compute mean and rstd
    for (const auto n : c10::irange(N)) {
      for (const auto g : c10::irange(G)) {
        T_ACC mean_val{0}, rstd_val{0};
        for (const auto d : c10::irange(D)) {
          for (const auto t : c10::irange(num_threads)) {
            T_ACC* buffer_ptr = buffer_data + t * N * 2 * C + n * 2 * C;
            mean_val += buffer_ptr[g * D + d];
            rstd_val += buffer_ptr[g * D + d + C];
           }
        }
        mean_val *= s;
        rstd_val = std::max(rstd_val * s - mean_val * mean_val, T_ACC(0));
        rstd_val = T_ACC(1) / std::sqrt(rstd_val + eps);
        tmp_buffer_data[n * 2 * G + 2 * g] = mean_val;
        tmp_buffer_data[n * 2 * G + 2 * g + 1] = rstd_val;
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
        T_ACC* scale_ptr = buffer_data + n * 2 * C;
        T_ACC* bias_ptr = scale_ptr + C;
        T_ACC mean_val = tmp_buffer_data[n * 2 * G + 2 * g];
        T_ACC rstd_val = tmp_buffer_data[n * 2 * G + 2 * g + 1];
        mean_data[n * G + g] = mean_val;
        rstd_data[n * G + g] = rstd_val;

        for (const auto d : c10::irange(D)) {
          const int64_t c = g * D + d;
          scale_ptr[c] = rstd_val * (gamma_null ? T_ACC(1) : T_ACC(gamma_data[c]));
          bias_ptr[c] = -scale_ptr[c] * mean_val + (beta_null ? T_ACC(0) : T_ACC(beta_data[c]));
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
        T_ACC* scale_ptr = buffer_data + n * 2 * C;
        T_ACC* bias_ptr = scale_ptr + C;
        apply_scale_bias<T, T_ACC>(Y_ptr, X_ptr, scale_ptr, bias_ptr, C);
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

template <typename T, typename T_ACC>
void ComputeInternalGradients(
    int64_t N,
    int64_t C,
    int64_t HxW,
    const T* dY,
    const T* X,
    T_ACC* ds,
    T_ACC* db) {
  at::parallel_for(0, N * C, 1, [=](int64_t start, int64_t end) {
    constexpr int64_t K = vec::Vectorized<T>::size();
    const int64_t inner_size = HxW / K * K;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    std::array<T_ACC, K> ds_arr;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    std::array<T_ACC, K> db_arr;
    for (const auto i : c10::irange(start, end)) {
      const T* dY_ptr = dY + i * HxW;
      const T* X_ptr = X + i * HxW;
      vec::Vectorized<T_ACC> ds_vec(0);
      vec::Vectorized<T_ACC> db_vec(0);
      for (int64_t j = 0; j < inner_size; j += K) {
        const vec::Vectorized<T> dy_vec = vec::Vectorized<T>::loadu(dY_ptr + j);
        const vec::Vectorized<T> x_vec = vec::Vectorized<T>::loadu(X_ptr + j);
        ds_vec = ds_vec + dy_vec * x_vec;
        db_vec = db_vec + dy_vec;
      }
      ds_vec.store(ds_arr.data());
      db_vec.store(db_arr.data());
      T_ACC ds_val = std::accumulate(ds_arr.cbegin(), ds_arr.cend(), T_ACC(0));
      T_ACC db_val = std::accumulate(db_arr.cbegin(), db_arr.cend(), T_ACC(0));
      for (const auto j : c10::irange(inner_size, HxW)) {
        ds_val += dY_ptr[j] * X_ptr[j];
        db_val += dY_ptr[j];
      }
      ds[i] = ds_val;
      db[i] = db_val;
    }
  });
}

template <>
void ComputeInternalGradients(
    int64_t N,
    int64_t C,
    int64_t HxW,
    const BFloat16* dY,
    const BFloat16* X,
    float* ds,
    float* db) {
  using bVec = vec::Vectorized<BFloat16>;
  using fVec = vec::Vectorized<float>;
  at::parallel_for(0, N * C, 1, [=](int64_t start, int64_t end) {
    constexpr int64_t K = bVec::size();
    const int64_t inner_size = HxW / K * K;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    std::array<float, K/2> ds_arr;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    std::array<float, K/2> db_arr;
    for (const auto i : c10::irange(start, end)) {
      const BFloat16* dY_ptr = dY + i * HxW;
      const BFloat16* X_ptr = X + i * HxW;
      fVec ds_vec(0);
      fVec db_vec(0);
      for (int64_t j = 0; j < inner_size; j += K) {
        const bVec dy_bvec = bVec::loadu(dY_ptr + j);
        const bVec x_bvec = bVec::loadu(X_ptr + j);
        fVec x_fvec0, x_fvec1, dy_fvec0, dy_fvec1;
        std::tie(x_fvec0, x_fvec1) = convert_bfloat16_float(x_bvec);
        std::tie(dy_fvec0, dy_fvec1) = convert_bfloat16_float(dy_bvec);
        ds_vec = ds_vec + dy_fvec0 * x_fvec0;
        ds_vec = ds_vec + dy_fvec1 * x_fvec1;
        db_vec = db_vec + dy_fvec0 + dy_fvec1;
      }
      ds_vec.store(ds_arr.data());
      db_vec.store(db_arr.data());
      float ds_val = std::accumulate(ds_arr.cbegin(), ds_arr.cend(), float(0));
      float db_val = std::accumulate(db_arr.cbegin(), db_arr.cend(), float(0));
      for (const auto j : c10::irange(inner_size, HxW)) {
        ds_val += float(dY_ptr[j]) * float(X_ptr[j]);
        db_val += float(dY_ptr[j]);
      }
      ds[i] = ds_val;
      db[i] = db_val;
    }
  });
}

template <typename PT, typename T_ACC>
inline void calc_db_ds(
    const T_ACC* ds_ptr,
    const T_ACC* db_ptr,
    bool gamma_null,
    const PT* gamma_ptr,
    const int64_t d,
    const int64_t K,
    void* ds_arr,
    void* db_arr) {
    vec::Vectorized<T_ACC> ds_vec(0);
    vec::Vectorized<T_ACC> db_vec(0);
    for (int64_t j = 0; j < d; j += K) {
      const vec::Vectorized<PT> gamma_vec = gamma_null
          ? vec::Vectorized<PT>(1)
          : vec::Vectorized<PT>::loadu(gamma_ptr + j);
      ds_vec = ds_vec + vec::Vectorized<T_ACC>::loadu(ds_ptr + j) * gamma_vec;
      db_vec = db_vec + vec::Vectorized<T_ACC>::loadu(db_ptr + j) * gamma_vec;
    }
    ds_vec.store(ds_arr);
    db_vec.store(db_arr);
}

template <>
inline void calc_db_ds(
    const float* ds_ptr,
    const float* db_ptr,
    bool gamma_null,
    const BFloat16* gamma_ptr,
    const int64_t d,
    const int64_t K,
    void* ds_arr,
    void* db_arr) {
    vec::Vectorized<float> ds_vec(0);
    vec::Vectorized<float> db_vec(0);
    for (int64_t j = 0; j < d; j += K) {
      const vec::Vectorized<BFloat16> gamma_vec = gamma_null
          ? vec::Vectorized<BFloat16>(1)
          : vec::Vectorized<BFloat16>::loadu(gamma_ptr + j);
      vec::Vectorized<float> gamma_vec0, gamma_vec1;
      std::tie(gamma_vec0, gamma_vec1) = convert_bfloat16_float(gamma_vec);
      ds_vec = ds_vec + vec::Vectorized<float>::loadu(ds_ptr + j) * gamma_vec0 +
        vec::Vectorized<float>::loadu(ds_ptr + j + K / 2) * gamma_vec1;
      db_vec = db_vec + vec::Vectorized<float>::loadu(db_ptr + j) * gamma_vec0 +
        vec::Vectorized<float>::loadu(db_ptr + j + K / 2) * gamma_vec1;
    }
    ds_vec.store(ds_arr);
    db_vec.store(db_arr);
}

template <typename T, typename PT, typename T_ACC>
void GroupNormInputBackward(
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    const T* dY,
    const T* X,
    const PT* mean,
    const PT* rstd,
    const PT* gamma,
    const T_ACC* ds,
    const T_ACC* db,
    T* dX) {
  const int64_t G = group;
  const int64_t D = C / G;
  const T_ACC s = T_ACC(1) / static_cast<T_ACC>(D * HxW);
  const bool gamma_null = (gamma == nullptr);
  at::parallel_for(0, N * G, 1, [=](int64_t start, int64_t end) {
    constexpr int64_t K = vec::Vectorized<PT>::size();
    constexpr int64_t scale = sizeof(T_ACC) / sizeof (PT);
    const int64_t d = D / K * K;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    std::array<T_ACC, K / scale> ds_arr;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    std::array<T_ACC, K / scale> db_arr;
    for (const auto i : c10::irange(start, end)) {
      const int64_t g = i % G;
      const T_ACC* ds_ptr = ds + i * D;
      const T_ACC* db_ptr = db + i * D;
      const PT* gamma_ptr = gamma + g * D;
      calc_db_ds(ds_ptr, db_ptr, gamma_null, gamma_ptr, d, K, ds_arr.data(), db_arr.data());
      T_ACC ds_val = std::accumulate(ds_arr.cbegin(), ds_arr.cend(), T_ACC(0));
      T_ACC db_val = std::accumulate(db_arr.cbegin(), db_arr.cend(), T_ACC(0));
      for (const auto j : c10::irange(d, D)) {
        const T_ACC gamma_v = gamma_null ? T_ACC(1) : T_ACC(gamma[g * D + j]);
        ds_val += ds_ptr[j] * gamma_v;
        db_val += db_ptr[j] * gamma_v;
      }
      const T_ACC c2 =
          (db_val * T_ACC(mean[i]) - ds_val) * T_ACC(rstd[i]) * T_ACC(rstd[i]) * T_ACC(rstd[i]) * s;
      // std::cout << "n " << i / G << " g " << g << " c2 " << c2 << " db_val " << db_val << " ds_val " << ds_val << " T_ACC(mean[i]) " << T_ACC(mean[i]) << " T_ACC(rstd[i]) " << T_ACC(rstd[i]) << " s " << s << "\n";
      const T_ACC c3 = -c2 * T_ACC(mean[i]) - db_val * T_ACC(rstd[i]) * s;
      // std::cout << "c3 " << c3 << "\n";
      for (const auto j : c10::irange(D)) {
        const int64_t c = g * D + j;
        const T* dY_ptr = dY + (i * D + j) * HxW;
        const T* X_ptr = X + (i * D + j) * HxW;
        T* dX_ptr = dX + (i * D + j) * HxW;
        const T_ACC c1 = T_ACC(rstd[i]) * (gamma_null ? T_ACC(1) : T_ACC(gamma[c]));
        // std::cout << "c1 " << c1 << " T_ACC(rstd[i]) " << T_ACC(rstd[i]) << " T_ACC(gamma[c]) " << T_ACC(gamma[c]) << "\n";
        for (const auto k : c10::irange(HxW)) {
          dX_ptr[k] = c1 * T_ACC(dY_ptr[k]) + c2 * T_ACC(X_ptr[k]) + c3;
          // std::cout << "dX_ptr[k] " << T_ACC(dX_ptr[k]) << "\n";
          // std::cout << "T_ACC(dY_ptr[k]) " << T_ACC(dY_ptr[k]) << " T_ACC(X_ptr[k]) " << T_ACC(X_ptr[k]) << "\n";
        }
      }
    }
  });
}

template <typename PT, typename T_ACC>
void GammaBackward(
    int64_t N,
    int64_t C,
    int64_t group,
    const PT* mean,
    const PT* rstd,
    const T_ACC* ds,
    const T_ACC* db,
    PT* dgamma) {
  const int64_t G = group;
  const int64_t D = C / G;
  constexpr int64_t K = vec::Vectorized<PT>::size();
  at::parallel_for(0, D, K, [=](int64_t start, int64_t end) {
    for (const auto i : c10::irange(G)) {
      std::memset(dgamma + i * D + start, 0, (end - start) * sizeof(PT));
    }
    for (int64_t i = 0; i < N * G; ++i) {
      const T_ACC* ds_ptr = ds + i * D;
      const T_ACC* db_ptr = db + i * D;
      const int64_t g = i % G;

      for (const auto j : c10::irange(start, end)) {
        const int64_t c = g * D + j;
        dgamma[c] += (ds_ptr[j] - db_ptr[j] * T_ACC(mean[i])) * T_ACC(rstd[i]);
      }
    }
  });
}

template <>
void GammaBackward(
    int64_t N,
    int64_t C,
    int64_t group,
    const BFloat16* mean,
    const BFloat16* rstd,
    const float* ds,
    const float* db,
    BFloat16* dgamma) {
  using fVec = vec::Vectorized<float>;
  using bVec = vec::Vectorized<BFloat16>;
  const int64_t G = group;
  const int64_t D = C / G;
  constexpr int64_t K = bVec::size();
  float* acc_buffer = new float[C];
  at::parallel_for(0, D, K, [=](int64_t start, int64_t end) {
    for (const auto i : c10::irange(G)) {
      std::memset(acc_buffer + i * D + start, 0, (end - start) * sizeof(float));
    }
    for (int64_t i = 0; i < N * G; ++i) {
      const float* ds_ptr = ds + i * D;
      const float* db_ptr = db + i * D;
      const int64_t g = i % G;
      for (const auto j : c10::irange(start, end)) {
        const int64_t c = g * D + j;
        acc_buffer[c] += (ds_ptr[j] - db_ptr[j] * float(mean[i])) * float(rstd[i]);
      }
    }
  });

  int64_t d = 0;
  for (; d < C / K * K; d += K) {
    fVec data_fvec0 = fVec::loadu(acc_buffer + d);
    fVec data_fvec1 = fVec::loadu(acc_buffer + d + K / 2);
    convert_float_bfloat16(data_fvec0, data_fvec1).store(dgamma + d);

  }
  if (C - d > 0) {
    fVec data_fvec0 = fVec::loadu(acc_buffer + d, (C - d) > fVec::size() ? fVec::size() : (C - d));
    fVec data_fvec1 = fVec::loadu(acc_buffer + d + fVec::size(), (C - d) > fVec::size() ? (C - d - fVec::size()) : 0);
    convert_float_bfloat16(data_fvec0, data_fvec1).store(dgamma + d, C - d);
  }
  if (acc_buffer != nullptr) {
    delete[] acc_buffer;
  }
}

template <typename PT, typename T_ACC>
void BetaBackward(int64_t N, int64_t C, const T_ACC* db, PT* dbeta) {
  constexpr int64_t K = vec::Vectorized<PT>::size();
  at::parallel_for(0, C, K, [=](int64_t start, int64_t end) {
    std::memset(dbeta + start, 0, (end - start) * sizeof(PT));
    for (const auto i : c10::irange(N)) {
      const T_ACC* db_ptr = db + i * C;
      for (const auto j : c10::irange(start, end)) {
        dbeta[j] += db_ptr[j];
      }
    }
  });
}

template <>
void BetaBackward(int64_t N, int64_t C, const float* db, BFloat16* dbeta) {
  using fVec = vec::Vectorized<float>;
  using bVec = vec::Vectorized<BFloat16>;
  constexpr int64_t K = bVec::size();
  float* acc_buffer = new float[C];
  at::parallel_for(0, C, K, [=](int64_t start, int64_t end) {
    std::memset(acc_buffer + start, 0, (end - start) * sizeof(float));
    for (const auto i : c10::irange(N)) {
      const float* db_ptr = db + i * C;
      for (const auto j : c10::irange(start, end)) {
        acc_buffer[j] += db_ptr[j];
      }
    }
  });
  int64_t d = 0;
  for (; d < C / K * K; d += K) {
    fVec data_fvec0 = fVec::loadu(acc_buffer + d);
    fVec data_fvec1 = fVec::loadu(acc_buffer + d + K / 2);
    convert_float_bfloat16(data_fvec0, data_fvec1).store(dbeta + d);

  }
  if (C - d > 0) {
    fVec data_fvec0 = fVec::loadu(acc_buffer + d, (C - d) > fVec::size() ? fVec::size() : (C - d));
    fVec data_fvec1 = fVec::loadu(acc_buffer + d + fVec::size(), (C - d) > fVec::size() ? (C - d - fVec::size()) : 0);
    convert_float_bfloat16(data_fvec0, data_fvec1).store(dbeta + d, C - d);
  }
  if (acc_buffer != nullptr) {
    delete[] acc_buffer;
  }
}

template <typename T, typename PT>
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
  using T_ACC = vec::vec_scalar_t<T>;
  const T* dY_data = dY.data_ptr<T>();
  const T* X_data = X.data_ptr<T>();
  const PT* mean_data = mean.data_ptr<PT>();
  const PT* rstd_data = rstd.data_ptr<PT>();
  const PT* gamma_data = gamma.defined() ? gamma.data_ptr<PT>() : nullptr;
  T* dX_data = dX.defined() ? dX.data_ptr<T>() : nullptr;
  PT* dgamma_data = dgamma.defined() ? dgamma.data_ptr<PT>() : nullptr;
  PT* dbeta_data = dbeta.defined() ? dbeta.data_ptr<PT>() : nullptr;
  Tensor ds = at::empty({N, C}, X.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));
  Tensor db = at::empty({N, C}, X.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));
  T_ACC* ds_data = ds.data_ptr<T_ACC>();
  T_ACC* db_data = db.data_ptr<T_ACC>();
  ComputeInternalGradients<T, T_ACC>(N, C, HxW, dY_data, X_data, ds_data, db_data);
  // std::cout << "ds " << ds << "\n";
  // std::cout << "db " << db << "\n";
  if (dX_data != nullptr) {
    GroupNormInputBackward<T, PT, T_ACC>(
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
        dX_data);\
    // std::cout << "dX " << dX << "\n";
  }
  if (dgamma_data != nullptr) {
    GammaBackward<PT, T_ACC>(
        N, C, group, mean_data, rstd_data, ds_data, db_data, dgamma_data);
    // std::cout << "dgamma " << dgamma << "\n";
  }
  if (dbeta_data != nullptr) {
    BetaBackward<PT, T_ACC>(N, C, db_data, dbeta_data);
    // std::cout << "dbeta " << dbeta << "\n";
  }
}

template <typename scalar_t, typename param_t>
inline void calc_ds_db_channels_last(
  const scalar_t* dY_ptr,
  const scalar_t* X_ptr,
  param_t* ds_ptr,
  param_t* db_ptr,
  int64_t C) {
  using Vec = vec::Vectorized<vec::vec_scalar_t<param_t>>;
  constexpr int64_t K = vec::Vectorized<scalar_t>::size();
  const int64_t inner_size = C / K * K;
  int64_t d = 0;
  for (; d < inner_size; d+= K) {
    Vec ds_dev = Vec::loadu(ds_ptr + d);
    Vec db_vec = Vec::loadu(db_ptr + d);
    Vec x_vec = Vec::loadu(X_ptr + d);
    Vec dy_vec = Vec::loadu(dY_ptr + d);

    ds_dev += x_vec * dy_vec;
    db_vec += dy_vec;
    ds_dev.store(ds_ptr + d);
    db_vec.store(db_ptr + d);
  }
  if (C - d > 0) {
    Vec ds_dev = Vec::loadu(ds_ptr + d, C - d);
    Vec db_vec = Vec::loadu(db_ptr + d, C - d);
    Vec x_vec = Vec::loadu(X_ptr + d, C - d);
    Vec dy_vec = Vec::loadu(dY_ptr + d, C - d);
    ds_dev += x_vec * dy_vec;
    db_vec += dy_vec;
    ds_dev.store(ds_ptr + d, C - d);
    db_vec.store(db_ptr + d, C - d);
  }
}

template <>
inline void calc_ds_db_channels_last(
  const BFloat16* dY_ptr,
  const BFloat16* X_ptr,
  float* ds_ptr,
  float* db_ptr,
  int64_t C) {
  using fVec = vec::Vectorized<float>;
  using bVec = vec::Vectorized<BFloat16>;
  int64_t d = 0;
  for (; d < C - (C % bVec::size()); d += bVec::size()) {
    fVec ds_dev0 = fVec::loadu(ds_ptr + d);
    fVec ds_dev1 = fVec::loadu(ds_ptr + d + fVec::size());
    fVec db_vec0 = fVec::loadu(db_ptr + d);
    fVec db_vec1 = fVec::loadu(db_ptr + d + fVec::size());
    bVec x_vec = bVec::loadu(X_ptr + d);
    bVec dy_vec = bVec::loadu(dY_ptr + d);
    fVec x_vec0, x_vec1, dy_vec0, dy_vec1;
    std::tie(x_vec0, x_vec1) = convert_bfloat16_float(x_vec);
    std::tie(dy_vec0, dy_vec1) = convert_bfloat16_float(dy_vec);
    ds_dev0 += x_vec0 * dy_vec0;
    ds_dev1 += x_vec1 * dy_vec1;
    db_vec0 += dy_vec0;
    db_vec1 += dy_vec1;

    ds_dev0.store(ds_ptr + d);
    ds_dev1.store(ds_ptr + d + fVec::size());
    db_vec0.store(db_ptr + d);
    db_vec1.store(db_ptr + d + fVec::size());

  }
  if (C - d > 0) {
    fVec ds_dev0 = fVec::loadu(ds_ptr + d, (C - d) > fVec::size() ? fVec::size() : (C - d));
    fVec ds_dev1 = fVec::loadu(ds_ptr + d + fVec::size(), (C - d) > fVec::size() ? (C - d - fVec::size()) : 0);
    fVec db_vec0 = fVec::loadu(db_ptr + d, (C - d) > fVec::size() ? fVec::size() : (C - d));
    fVec db_vec1 = fVec::loadu(db_ptr + d + fVec::size(), (C - d) > fVec::size() ? (C - d - fVec::size()) : 0);
    bVec x_vec = bVec::loadu(X_ptr + d, C - d);
    bVec dy_vec = bVec::loadu(dY_ptr + d, C - d);
    fVec x_vec0, x_vec1, dy_vec0, dy_vec1;
    std::tie(x_vec0, x_vec1) = convert_bfloat16_float(x_vec);
    std::tie(dy_vec0, dy_vec1) = convert_bfloat16_float(dy_vec);
    ds_dev0 += x_vec0 * dy_vec0;
    ds_dev1 += x_vec1 * dy_vec1;
    db_vec0 += dy_vec0;
    db_vec1 += dy_vec1;

    ds_dev0.store(ds_ptr + d, (C - d) > fVec::size() ? fVec::size() : (C - d));
    ds_dev1.store(ds_ptr + d + fVec::size(), (C - d) > fVec::size() ? (C - d - fVec::size()) : 0);
    db_vec0.store(db_ptr + d, (C - d) > fVec::size() ? fVec::size() : (C - d));
    db_vec1.store(db_ptr + d + fVec::size(), (C - d) > fVec::size() ? (C - d - fVec::size()) : 0);
  }
}

template <typename T, typename PT, typename T_ACC>
void apply_input_backward_channles_last(
  const T* dY_data,
  const T* X_data,
  T* dX_data,
  const PT* rstd,
  const PT* gamma,
  T_ACC c2,
  T_ACC c3,
  int64_t HxW,
  int64_t C,
  int64_t D,
  bool ColMov) {
  const bool gamma_null = (gamma == nullptr);
  int64_t d = 0;
  auto K = vec::Vectorized<T>::size();
  for (; d < D / K * K; d += K) {
    if (ColMov) {
      for (const auto m : c10::irange(HxW)) {
        const T* X_ptr = X_data + m * C;
        const T* dY_ptr = dY_data + m * C;
        T* dX_ptr = dX_data + m * C;
        auto c1 = vec::Vectorized<T>(*rstd) *
          (gamma_null ? vec::Vectorized<T>(1) : vec::Vectorized<T>::loadu(gamma + d));
        auto dy_vec = vec::Vectorized<T>::loadu(dY_ptr + d);
        auto x_vec = vec::Vectorized<T>::loadu(X_ptr + d);
        auto dx_vec = c1 * dy_vec +
          vec::Vectorized<T>(c2) * x_vec + vec::Vectorized<T>(c3);
        dx_vec.store(dX_ptr + d);
      }
    } else {
      auto c1 = vec::Vectorized<T>(*rstd) *
        (gamma_null ? vec::Vectorized<T>(1) : vec::Vectorized<T>::loadu(gamma + d));
      auto dy_vec = vec::Vectorized<T>::loadu(dY_data + d);
      auto x_vec = vec::Vectorized<T>::loadu(X_data + d);
      auto dx_vec = c1 * dy_vec +
        vec::Vectorized<T>(c2) * x_vec + vec::Vectorized<T>(c3);
      dx_vec.store(dX_data + d);
    }
  }
  if (D - d > 0) {
    if (ColMov) {
      for (const auto m : c10::irange(HxW)) {
        const T* X_ptr = X_data + m * C;
        const T* dY_ptr = dY_data + m * C;
        T* dX_ptr = dX_data + m * C;
        auto c1 = vec::Vectorized<T>(*rstd) *
        (gamma_null ? vec::Vectorized<T>(1) : vec::Vectorized<T>::loadu(gamma + d, D - d));
      auto dy_vec = vec::Vectorized<T>::loadu(dY_ptr + d, D - d);
      auto x_vec = vec::Vectorized<T>::loadu(X_ptr + d, D - d);
      auto dx_vec = c1 * dy_vec +
        vec::Vectorized<T>(c2) * x_vec + vec::Vectorized<T>(c3);
      dx_vec.store(dX_ptr + d, D - d);
      }
    } else {
      auto c1 = vec::Vectorized<T>(*rstd) *
        (gamma_null ? vec::Vectorized<T>(1) : vec::Vectorized<T>::loadu(gamma + d, D - d));
      auto dy_vec = vec::Vectorized<T>::loadu(dY_data + d, D - d);
      auto x_vec = vec::Vectorized<T>::loadu(X_data + d, D - d);
      auto dx_vec = c1 * dy_vec +
        vec::Vectorized<T>(c2) * x_vec + vec::Vectorized<T>(c3);
      dx_vec.store(dX_data + d, D - d);
    }
  }
}

// TODO add specialization for T = BFloat16, PT = BFloat16
template <>
void apply_input_backward_channles_last(
  const BFloat16* dY_data,
  const BFloat16* X_data,
  BFloat16* dX_data,
  const BFloat16* rstd,
  const BFloat16* gamma,
  float c2,
  float c3,
  int64_t HxW,
  int64_t C,
  int64_t D,
  bool ColMov) {
  using bVec = vec::Vectorized<BFloat16>;
  using fVec = vec::Vectorized<float>;
  const bool gamma_null = (gamma == nullptr);
  auto K = bVec::size();
  int64_t d = 0;
  for (; d < D / K * K; d += K) {
    if (ColMov) {
      for (const auto m : c10::irange(HxW)) {
        const BFloat16* X_ptr = X_data + m * C;
        const BFloat16* dY_ptr = dY_data + m * C;
        BFloat16* dX_ptr = dX_data + m * C;
        // fVec c1_0 = fVec(*rstd) *
        //   (gamma_null ? fVec(1) : fVec::loadu(gamma + d));
        // fVec c1_1 = fVec(*rstd) *
        //   (gamma_null ? fVec(1) : fVec::loadu(gamma + d + fVec::size()));
        bVec gamma_vec = bVec::loadu(gamma + d);
        fVec rstd_vec = fVec(float(*rstd));
        bVec dy_vec = bVec::loadu(dY_ptr + d);
        bVec x_vec = bVec::loadu(X_ptr + d);
        fVec dy_vec0, dy_vec1, x_vec0, x_vec1, c1_vec0, c1_vec1;
        fVec gamma_vec0, gamma_vec1;
        std::tie(gamma_vec0, gamma_vec1) = convert_bfloat16_float(gamma_vec);
        std::tie(x_vec0, x_vec1) = convert_bfloat16_float(x_vec);
        std::tie(dy_vec0, dy_vec1) = convert_bfloat16_float(dy_vec);
        c1_vec0 = rstd_vec * (gamma_null ? fVec(1) : gamma_vec0);
        c1_vec1 = rstd_vec * (gamma_null ? fVec(1) : gamma_vec1);
        fVec dx_vec0 = c1_vec0 * dy_vec0 + fVec(c2) * x_vec0 + fVec(c3);
        fVec dx_vec1 = c1_vec1 * dy_vec1 + fVec(c2) * x_vec1 + fVec(c3);
        convert_float_bfloat16(dx_vec0, dx_vec1).store(dX_ptr + d);
      }
    } else {
      bVec gamma_vec = bVec::loadu(gamma + d);
      fVec rstd_vec = fVec(float(*rstd));
      bVec dy_vec = bVec::loadu(dY_data + d);
      bVec x_vec = bVec::loadu(X_data + d);
      fVec dy_vec0, dy_vec1, x_vec0, x_vec1, c1_vec0, c1_vec1;
      fVec gamma_vec0, gamma_vec1;
      std::tie(gamma_vec0, gamma_vec1) = convert_bfloat16_float(gamma_vec);
      std::tie(x_vec0, x_vec1) = convert_bfloat16_float(x_vec);
      std::tie(dy_vec0, dy_vec1) = convert_bfloat16_float(dy_vec);
      c1_vec0 = rstd_vec * (gamma_null ? fVec(1) : gamma_vec0);
      c1_vec1 = rstd_vec * (gamma_null ? fVec(1) : gamma_vec1);
      fVec dx_vec0 = c1_vec0 * dy_vec0 + fVec(c2) * x_vec0 + fVec(c3);
      fVec dx_vec1 = c1_vec1 * dy_vec1 + fVec(c2) * x_vec1 + fVec(c3);
      convert_float_bfloat16(dx_vec0, dx_vec1).store(dX_data + d);
    }
  }
  if (D - d > 0) {
    if (ColMov) {
      for (const auto m : c10::irange(HxW)) {
        const BFloat16* X_ptr = X_data + m * C;
        const BFloat16* dY_ptr = dY_data + m * C;
        BFloat16* dX_ptr = dX_data + m * C;
        // fVec c1_0 = fVec(*rstd) *
        //   (gamma_null ? fVec(1) : fVec::loadu(gamma + d, (D - d) > fVec::size() ? fVec::size() : (D - d)));
        // fVec c1_1 = fVec(*rstd) *
        //   (gamma_null ? fVec(1) : fVec::loadu(gamma + d + fVec::size(), (D - d) > fVec::size() ? (D - d - fVec::size()) : 0));
        bVec gamma_vec = bVec::loadu(gamma + d, D - d);
        fVec rstd_vec = fVec(float(*rstd));
        bVec dy_vec = bVec::loadu(dY_ptr + d, D - d);
        bVec x_vec = bVec::loadu(X_ptr + d, D - d);
        fVec dy_vec0, dy_vec1, x_vec0, x_vec1, c1_vec0, c1_vec1;
        fVec gamma_vec0, gamma_vec1;
        std::tie(gamma_vec0, gamma_vec1) = convert_bfloat16_float(gamma_vec);
        std::tie(x_vec0, x_vec1) = convert_bfloat16_float(x_vec);
        std::tie(dy_vec0, dy_vec1) = convert_bfloat16_float(dy_vec);
        c1_vec0 = rstd_vec * (gamma_null ? fVec(1) : gamma_vec0);
        c1_vec1 = rstd_vec * (gamma_null ? fVec(1) : gamma_vec1);
        fVec dx_vec0 = c1_vec0 * dy_vec0 + fVec(c2) * x_vec0 + fVec(c3);
        fVec dx_vec1 = c1_vec1 * dy_vec1 + fVec(c2) * x_vec1 + fVec(c3);
        convert_float_bfloat16(dx_vec0, dx_vec1).store(dX_ptr + d, D - d);
      }
    } else {
      bVec gamma_vec = bVec::loadu(gamma + d, D - d);
      fVec rstd_vec = fVec(float(*rstd));
      bVec dy_vec = bVec::loadu(dY_data + d, D - d);
      bVec x_vec = bVec::loadu(X_data + d, D - d);
      fVec dy_vec0, dy_vec1, x_vec0, x_vec1, c1_vec0, c1_vec1;
      fVec gamma_vec0, gamma_vec1;
      std::tie(gamma_vec0, gamma_vec1) = convert_bfloat16_float(gamma_vec);
      std::tie(x_vec0, x_vec1) = convert_bfloat16_float(x_vec);
      std::tie(dy_vec0, dy_vec1) = convert_bfloat16_float(dy_vec);
      c1_vec0 = rstd_vec * (gamma_null ? fVec(1) : gamma_vec0);
      c1_vec1 = rstd_vec * (gamma_null ? fVec(1) : gamma_vec1);
      fVec dx_vec0 = c1_vec0 * dy_vec0 + fVec(c2) * x_vec0 + fVec(c3);
      fVec dx_vec1 = c1_vec1 * dy_vec1 + fVec(c2) * x_vec1 + fVec(c3);
      convert_float_bfloat16(dx_vec0, dx_vec1).store(dX_data + d, D - d);
    }
  }
}

template <>
void apply_input_backward_channles_last(
  const BFloat16* dY_data,
  const BFloat16* X_data,
  BFloat16* dX_data,
  const float* rstd,
  const float* gamma,
  float c2,
  float c3,
  int64_t HxW,
  int64_t C,
  int64_t D,
  bool ColMov) {
  using bVec = vec::Vectorized<BFloat16>;
  using fVec = vec::Vectorized<float>;
  const bool gamma_null = (gamma == nullptr);
  auto K = bVec::size();
  int64_t d = 0;
  for (; d < D / K * K; d += K) {
    if (ColMov) {
      for (const auto m : c10::irange(HxW)) {
        const BFloat16* X_ptr = X_data + m * C;
        const BFloat16* dY_ptr = dY_data + m * C;
        BFloat16* dX_ptr = dX_data + m * C;
        fVec c1_0 = fVec(*rstd) *
          (gamma_null ? fVec(1) : fVec::loadu(gamma + d));
        fVec c1_1 = fVec(*rstd) *
          (gamma_null ? fVec(1) : fVec::loadu(gamma + d + fVec::size()));
        bVec dy_vec = bVec::loadu(dY_ptr + d);
        bVec x_vec = bVec::loadu(X_ptr + d);
        fVec dy_vec0, dy_vec1, x_vec0, x_vec1;
        std::tie(x_vec0, x_vec1) = convert_bfloat16_float(x_vec);
        std::tie(dy_vec0, dy_vec1) = convert_bfloat16_float(dy_vec);
        fVec dx_vec0 = c1_0 * dy_vec0 + fVec(c2) * x_vec0 + fVec(c3);
        fVec dx_vec1 = c1_1 * dy_vec1 + fVec(c2) * x_vec1 + fVec(c3);
        convert_float_bfloat16(dx_vec0, dx_vec1).store(dX_ptr + d);
      }
    } else {
      fVec c1_0 = fVec(*rstd) *
        (gamma_null ? fVec(1) : fVec::loadu(gamma + d));
      fVec c1_1 = fVec(*rstd) *
        (gamma_null ? fVec(1) : fVec::loadu(gamma + d + fVec::size()));
      bVec dy_vec = bVec::loadu(dY_data + d);
      bVec x_vec = bVec::loadu(X_data + d);
      fVec dy_vec0, dy_vec1, x_vec0, x_vec1;
      std::tie(x_vec0, x_vec1) = convert_bfloat16_float(x_vec);
      std::tie(dy_vec0, dy_vec1) = convert_bfloat16_float(dy_vec);
      fVec dx_vec0 = c1_0 * dy_vec0 + fVec(c2) * x_vec0 + fVec(c3);
      fVec dx_vec1 = c1_1 * dy_vec1 + fVec(c2) * x_vec1 + fVec(c3);
      convert_float_bfloat16(dx_vec0, dx_vec1).store(dX_data + d);
    }
  }
  if (D - d > 0) {
    if (ColMov) {
      for (const auto m : c10::irange(HxW)) {
        const BFloat16* X_ptr = X_data + m * C;
        const BFloat16* dY_ptr = dY_data + m * C;
        BFloat16* dX_ptr = dX_data + m * C;
        fVec c1_0 = fVec(*rstd) *
          (gamma_null ? fVec(1) : fVec::loadu(gamma + d, (D - d) > fVec::size() ? fVec::size() : (D - d)));
        fVec c1_1 = fVec(*rstd) *
          (gamma_null ? fVec(1) : fVec::loadu(gamma + d + fVec::size(), (D - d) > fVec::size() ? (D - d - fVec::size()) : 0));
        bVec dy_vec = bVec::loadu(dY_ptr + d, D - d);
        bVec x_vec = bVec::loadu(X_ptr + d, D - d);
        fVec dy_vec0, dy_vec1, x_vec0, x_vec1;
        std::tie(x_vec0, x_vec1) = convert_bfloat16_float(x_vec);
        std::tie(dy_vec0, dy_vec1) = convert_bfloat16_float(dy_vec);
        fVec dx_vec0 = c1_0 * dy_vec0 + fVec(c2) * x_vec0 + fVec(c3);
        fVec dx_vec1 = c1_1 * dy_vec1 + fVec(c2) * x_vec1 + fVec(c3);
        convert_float_bfloat16(dx_vec0, dx_vec1).store(dX_ptr + d, D - d);
      }
    } else {
      // float c1 = (*rstd) * (gamma_null ? float(1) : gamma[d]);
      // dX_ptr[d] = c1 * float(dY_ptr[d]) + c2 * float(X_ptr[d]) + c3;
      fVec c1_0 = fVec(*rstd) *
        (gamma_null ? fVec(1) : fVec::loadu(gamma + d, (D - d) > fVec::size() ? fVec::size() : (D - d)));
      fVec c1_1 = fVec(*rstd) *
        (gamma_null ? fVec(1) : fVec::loadu(gamma + d + fVec::size(), (D - d) > fVec::size() ? (D - d - fVec::size()) : 0));
      bVec dy_vec = bVec::loadu(dY_data + d, D - d);
      bVec x_vec = bVec::loadu(X_data + d, D - d);
      fVec dy_vec0, dy_vec1, x_vec0, x_vec1;
      std::tie(x_vec0, x_vec1) = convert_bfloat16_float(x_vec);
      std::tie(dy_vec0, dy_vec1) = convert_bfloat16_float(dy_vec);
      fVec dx_vec0 = c1_0 * dy_vec0 + fVec(c2) * x_vec0 + fVec(c3);
      fVec dx_vec1 = c1_1 * dy_vec1 + fVec(c2) * x_vec1 + fVec(c3);
      convert_float_bfloat16(dx_vec0, dx_vec1).store(dX_data + d, D - d);
    }

  }
}

template <typename T, typename PT, typename T_ACC>
std::tuple<T_ACC, T_ACC> DsDbColumnwiseMoments(
    const T* X_data,
    const T* dY_data,
    const PT* gamma_ptr,
    T_ACC* ds_ptr,
    T_ACC* db_ptr,
    int64_t HxW,
    int64_t C,
    int64_t D) {
  using Vec = vec::Vectorized<T>;
  constexpr int64_t K = Vec::size();
  const int64_t inner_size = D / K * K;
  int64_t d = 0;
  T_ACC ds_gamma{0}, db_gamma{0};
  for (; d < inner_size; d += K) {
    Vec acc0_vec{0}, acc1_vec{0};
    for (const auto m : c10::irange(HxW)) {
      const T* X_ptr = X_data + m * C;
      const T* dY_ptr = dY_data + m * C;
      Vec x_vec = Vec::loadu(X_ptr + d);
      Vec dy_vec = Vec::loadu(dY_ptr + d);
      acc0_vec += x_vec * dy_vec;
      acc1_vec += dy_vec;
    }
    acc0_vec.store(ds_ptr + d);
    acc1_vec.store(db_ptr + d);
    ds_gamma += vec::vec_reduce_all([](Vec& x, Vec& y) { return x + y; },
      acc0_vec * Vec::loadu(gamma_ptr + d));
    db_gamma += vec::vec_reduce_all([](Vec& x, Vec& y) { return x + y; },
      acc1_vec * Vec::loadu(gamma_ptr + d));
  }
  // std::cout << "DsDbColumnwiseMoments 1 " << " ds_gamma " << ds_gamma << " db_gamma " << db_gamma << "\n";
  if (D - d > 0) {
    Vec acc0_vec{0}, acc1_vec{0};
    for (const auto m : c10::irange(HxW)) {
      const T* X_ptr = X_data + m * C;
      const T* dY_ptr = dY_data + m * C;
      Vec x_vec = Vec::loadu(X_ptr + d, D - d);
      Vec dy_vec = Vec::loadu(dY_ptr + d, D - d);
      acc0_vec += x_vec * dy_vec;
      acc1_vec += dy_vec;
    }
    acc0_vec.store(ds_ptr + d, D - d);
    acc1_vec.store(db_ptr + d, D - d);
    ds_gamma += vec::vec_reduce_all([](Vec& x, Vec& y) { return x + y; },
      acc0_vec * Vec::loadu(gamma_ptr + d, D - d));
    db_gamma += vec::vec_reduce_all([](Vec& x, Vec& y) { return x + y; },
      acc1_vec * Vec::loadu(gamma_ptr + d, D - d));
  }
  // std::cout << "DsDbColumnwiseMoments 2 " << " ds_gamma " << ds_gamma << " db_gamma " << db_gamma << "\n";
  return std::tuple<T_ACC, T_ACC>(ds_gamma, db_gamma);
}

template <>
std::tuple<float, float> DsDbColumnwiseMoments(
  const BFloat16* X_data,
  const BFloat16* dY_data,
  const float* gamma_ptr,
  float* ds_ptr,
  float* db_ptr,
  int64_t HxW,
  int64_t C,
  int64_t D) {
  using bVec = vec::Vectorized<BFloat16>;
  using fVec = vec::Vectorized<float>;
  constexpr int64_t K = bVec::size();
  const int64_t inner_size = D / K * K;
  float ds_gamma{0}, db_gamma{0};
  int64_t d = 0;
  for (; d < inner_size; d += K) {
    fVec acc0_vec0{0}, acc0_vec1{0}, acc1_vec0{0}, acc1_vec1{0};
    for (const auto m : c10::irange(HxW)) {
      const BFloat16* X_ptr = X_data + m * C;
      const BFloat16* dY_ptr = dY_data + m * C;
      bVec x_vec = bVec::loadu(X_ptr + d);
      bVec dy_vec = bVec::loadu(dY_ptr + d);
      fVec x_vec0, x_vec1, dy_vec0, dy_vec1;
      std::tie(x_vec0, x_vec1) = convert_bfloat16_float(x_vec);
      std::tie(dy_vec0, dy_vec1) = convert_bfloat16_float(dy_vec);
      acc0_vec0 += x_vec0 * dy_vec0;
      acc0_vec1 += x_vec1 * dy_vec1;
      acc1_vec0 += dy_vec0;
      acc1_vec1 += dy_vec1;
    }
    acc0_vec0.store(ds_ptr + d);
    acc0_vec1.store(ds_ptr + d + fVec::size());
    acc1_vec0.store(db_ptr + d);
    acc1_vec1.store(db_ptr + d + fVec::size());
    ds_gamma += vec::vec_reduce_all([](fVec& x, fVec& y) { return x + y; },
      acc0_vec0 * fVec::loadu(gamma_ptr + d));
    ds_gamma += vec::vec_reduce_all([](fVec& x, fVec& y) { return x + y; },
      acc0_vec1 * fVec::loadu(gamma_ptr + d + fVec::size()));
    db_gamma += vec::vec_reduce_all([](fVec& x, fVec& y) { return x + y; },
      acc1_vec0 * fVec::loadu(gamma_ptr + d));
    db_gamma += vec::vec_reduce_all([](fVec& x, fVec& y) { return x + y; },
      acc1_vec1 * fVec::loadu(gamma_ptr + d + fVec::size()));
  }
  for (; d < D; d++) {
    float acc0{0}, acc1{0};
    for (const auto m : c10::irange(HxW)) {
      const BFloat16* X_ptr = X_data + m * C;
      const BFloat16* dY_ptr = dY_data + m * C;
      acc0 += float(X_ptr[d]) * float(dY_ptr[d]);
      acc1 += float(dY_ptr[d]);
    }
    ds_ptr[d] = acc0;
    db_ptr[d] = acc1;
    ds_gamma += acc0 * gamma_ptr[d];
    db_gamma += acc1 * gamma_ptr[d];
  }

  return std::tuple<float, float>(ds_gamma, db_gamma);
}

template <>
std::tuple<float, float> DsDbColumnwiseMoments(
  const BFloat16* X_data,
  const BFloat16* dY_data,
  const BFloat16* gamma_ptr,
  float* ds_ptr,
  float* db_ptr,
  int64_t HxW,
  int64_t C,
  int64_t D) {
  using bVec = vec::Vectorized<BFloat16>;
  using fVec = vec::Vectorized<float>;
  constexpr int64_t K = bVec::size();
  const int64_t inner_size = D / K * K;
  float ds_gamma{0}, db_gamma{0};
  int64_t d = 0;
  for (; d < inner_size; d += K) {
    fVec acc0_vec0{0}, acc0_vec1{0}, acc1_vec0{0}, acc1_vec1{0};
    for (const auto m : c10::irange(HxW)) {
      const BFloat16* X_ptr = X_data + m * C;
      const BFloat16* dY_ptr = dY_data + m * C;
      bVec x_vec = bVec::loadu(X_ptr + d);
      bVec dy_vec = bVec::loadu(dY_ptr + d);
      fVec x_vec0, x_vec1, dy_vec0, dy_vec1;
      std::tie(x_vec0, x_vec1) = convert_bfloat16_float(x_vec);
      std::tie(dy_vec0, dy_vec1) = convert_bfloat16_float(dy_vec);
      acc0_vec0 += x_vec0 * dy_vec0;
      acc0_vec1 += x_vec1 * dy_vec1;
      acc1_vec0 += dy_vec0;
      acc1_vec1 += dy_vec1;
    }
    acc0_vec0.store(ds_ptr + d);
    acc0_vec1.store(ds_ptr + d + fVec::size());
    acc1_vec0.store(db_ptr + d);
    acc1_vec1.store(db_ptr + d + fVec::size());
    bVec gamma_vec = bVec::loadu(gamma_ptr + d);
    fVec gamma_vec0, gamma_vec1;
    std::tie(gamma_vec0, gamma_vec1) = convert_bfloat16_float(gamma_vec);

    ds_gamma += vec::vec_reduce_all([](fVec& x, fVec& y) { return x + y; },
      acc0_vec0 * gamma_vec0);
    ds_gamma += vec::vec_reduce_all([](fVec& x, fVec& y) { return x + y; },
      acc0_vec1 * gamma_vec1);
    db_gamma += vec::vec_reduce_all([](fVec& x, fVec& y) { return x + y; },
      acc1_vec0 * gamma_vec0);
    db_gamma += vec::vec_reduce_all([](fVec& x, fVec& y) { return x + y; },
      acc1_vec1 * gamma_vec1);
  }
  for (; d < D; d++) {
    float acc0{0}, acc1{0};
    for (const auto m : c10::irange(HxW)) {
      const BFloat16* X_ptr = X_data + m * C;
      const BFloat16* dY_ptr = dY_data + m * C;
      acc0 += float(X_ptr[d]) * float(dY_ptr[d]);
      acc1 += float(dY_ptr[d]);
    }
    ds_ptr[d] = acc0;
    db_ptr[d] = acc1;
    ds_gamma += acc0 * float(gamma_ptr[d]);
    db_gamma += acc1 * float(gamma_ptr[d]);
  }

  return std::tuple<float, float>(ds_gamma, db_gamma);
}

template <typename T, typename PT>
void GroupNormBackwardKernelImplChannelsLastInternal(
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
  int64_t D = C / group;
  int64_t G = group;
  using T_ACC = vec::vec_scalar_t<T>;
  const T* dY_data = dY.data_ptr<T>();
  const T* X_data = X.data_ptr<T>();
  const PT* mean_data = mean.data_ptr<PT>();
  const PT* rstd_data = rstd.data_ptr<PT>();
  const PT* gamma_data = gamma.defined() ? gamma.data_ptr<PT>() : nullptr;
  T* dX_data = dX.defined() ? dX.data_ptr<T>() : nullptr;
  PT* dgamma_data = dgamma.defined() ? dgamma.data_ptr<PT>() : nullptr;
  PT* dbeta_data = dbeta.defined() ? dbeta.data_ptr<PT>() : nullptr;
  const bool gamma_null = (gamma_data == nullptr);
  Tensor ds = at::empty({N, C}, X.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));
  Tensor db = at::empty({N, C}, X.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));
  T_ACC* ds_data = ds.data_ptr<T_ACC>();
  T_ACC* db_data = db.data_ptr<T_ACC>();
  const T_ACC s = T_ACC(1) / static_cast<T_ACC>(D * HxW);

  constexpr int64_t feature_map_threshold = 1024;
  if (HxW < feature_map_threshold) {
    at::parallel_for(0, N * G, 1, [=](int64_t begin, int64_t end) {
      int64_t n{0}, g{0};
      data_index_init(begin, n, N, g, G);
      for (const auto i : c10::irange(begin, end)) {
        // step 1 compute ds & db
        T_ACC* ds_ptr = ds_data + i * D;
        T_ACC* db_ptr = db_data + i * D;
        T_ACC ds_gamma, db_gamma;
        const T* X_ptr = X_data + n * HxW * C + g * D;
        const T* dY_ptr = dY_data + n * HxW * C + g * D;
        const PT* gamma_ptr = gamma_null ? gamma_data : (gamma_data + g * D);
        std::tie(ds_gamma, db_gamma) = DsDbColumnwiseMoments<T, PT, T_ACC>(
          X_ptr,
          dY_ptr,
          gamma_ptr,
          ds_ptr,
          db_ptr,
          HxW,
          C,
          D);
        // std::cout << "ds_gamma " << ds_gamma << " db_gamma " << db_gamma << "\n";
        // step2 compute dX
        T* dX_ptr = dX_data + n * HxW * C + g * D;
        const PT* rstd_ptr = rstd_data + i;
        const T_ACC c2 =
            (db_gamma * T_ACC(mean_data[i]) - ds_gamma) * T_ACC(rstd_data[i]) * T_ACC(rstd_data[i]) * T_ACC(rstd_data[i]) * s;
        const T_ACC c3 = -c2 * T_ACC(mean_data[i]) - db_gamma * T_ACC(rstd_data[i]) * s;
        apply_input_backward_channles_last<T, PT, T_ACC>(dY_ptr, X_ptr, dX_ptr, rstd_ptr, gamma_ptr, c2, c3, HxW, C, D, true);
        data_index_step(n, N, g, G);
      }
    });
    // std::cout << "ds " << ds << "\n";
    // std::cout << "db " << db << "\n";
    // std::cout << "dX " << dX << "\n";
    // step 3 compute dgamma & dbeta
    if (dgamma_data != nullptr) {
      GammaBackward<PT, T_ACC>(
          N, C, group, mean_data, rstd_data, ds_data, db_data, dgamma_data);
    }
    if (dbeta_data != nullptr) {
      BetaBackward<PT, T_ACC>(N, C, db_data, dbeta_data);
    }
  } else {
    // parallel on N * HxW.
    int num_threads = at::get_num_threads();
    Tensor buffer = at::empty({num_threads, N, 2 * C},
      X.options().dtype(c10::CppTypeToScalarType<T_ACC>::value)).zero_();
    T_ACC* buffer_data = buffer.data_ptr<T_ACC>();

    Tensor tmp_buffer = at::empty({N, 2 * G},
      X.options().dtype(c10::CppTypeToScalarType<T_ACC>::value));
    T_ACC* tmp_buffer_data = tmp_buffer.data_ptr<T_ACC>();

    // step 1
    at::parallel_for(0, N * HxW, 1, [&](int64_t begin, int64_t end) {
      int tid = at::get_thread_num();
      T_ACC* buffer_ptr = buffer_data + tid * N * 2 * C;
      int64_t n{0}, m{0};
      data_index_init(begin, n, N, m, HxW);
      for (const auto i : c10::irange(begin, end)) {
        T_ACC* ds_ptr = buffer_ptr +  n * 2 * C;
        T_ACC* db_ptr = ds_ptr + C;
        const T* X_ptr = X_data + i * C;
        const T* dY_ptr = dY_data + i * C;

        calc_ds_db_channels_last<T, T_ACC>(dY_ptr, X_ptr, ds_ptr, db_ptr, C);
        data_index_step(n, N, m, HxW);
      }
    });

    // step 2
    for (const auto n : c10::irange(N)) {
      for (const auto g : c10::irange(G)) {
        T_ACC ds_gamma{0}, db_gamma{0};
        for (const auto d : c10::irange(D)) {
          T_ACC ds_val{0}, db_val{0};
          for (const auto t : c10::irange(num_threads)) {
            T_ACC* buffer_ptr = buffer_data + t * N * 2 * C + n * 2 * C;
            ds_gamma += buffer_ptr[g * D + d] * (gamma_null ? T_ACC(1) : T_ACC(gamma_data[g * D + d]));
            db_gamma += buffer_ptr[g * D + d + C] * (gamma_null ? T_ACC(1) : T_ACC(gamma_data[g * D + d]));
            ds_val += buffer_ptr[g * D + d];
            db_val += buffer_ptr[g * D + d + C];

            }
          ds_data[n * C + g * D + d] = ds_val;
          db_data[n * C + g * D + d] = db_val;
        }
        tmp_buffer_data[n * 2 * G + 2 * g] = ds_gamma;
        tmp_buffer_data[n * 2 * G + 2 * g + 1] = db_gamma;
      }
    }

    // step 3 compute dx
    if (dX_data != nullptr) {
      at::parallel_for(0, N * HxW, 1, [&](int64_t begin, int64_t end) {
        int64_t n{0}, m{0};
        data_index_init(begin, n, N, m, HxW);
        for (const auto i : c10::irange(begin, end)) {
          for (const auto g : c10::irange(0, G)) {
            const T* X_ptr = X_data + i * C + g * D;
            const T* dY_ptr = dY_data + i * C + g * D;
            T* dX_ptr = dX_data + i * C + g * D;
            const PT* mean_ptr = mean_data + n * G + g;
            const PT* rstd_ptr = rstd_data + n * G + g;
            const PT* gamma_ptr = gamma_null ? gamma_data : (gamma_data + g * D);
            T_ACC ds_val = tmp_buffer_data[n * 2 * G + 2 * g];
            T_ACC db_val = tmp_buffer_data[n * 2 * G + 2 * g + 1];
            // std::cout << "ds_val " << ds_val << " db_val " << db_val << "\n";
            const T_ACC c2 =
            (db_val * T_ACC(*mean_ptr) - ds_val) * T_ACC(*rstd_ptr) * T_ACC(*rstd_ptr)* T_ACC(*rstd_ptr) * s;
            const T_ACC c3 = -c2 * T_ACC(*mean_ptr) - db_val * T_ACC(*rstd_ptr) * s;
            apply_input_backward_channles_last<T, PT, T_ACC>(dY_ptr, X_ptr, dX_ptr, rstd_ptr, gamma_ptr, c2, c3, HxW, C, D, false);
          }

          data_index_step(n, N, m, HxW);
        }
      });
    }
    // std::cout << "ds " << ds << "\n";
    // std::cout << "db " << db << "\n";
    // std::cout << "dX " << dX << "\n";
    // step 4 compute dgamma & dbeta
    if (dgamma_data != nullptr) {
      GammaBackward<PT, T_ACC>(
          N, C, group, mean_data, rstd_data, ds_data, db_data, dgamma_data);
    }
    if (dbeta_data != nullptr) {
      BetaBackward<PT, T_ACC>(N, C, db_data, dbeta_data);
    }
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
  const bool mixed_type = is_mixed_type(dY, gamma);
  switch (X.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      // std::cout << "Contiguous\n";
      AT_DISPATCH_FLOATING_TYPES_AND(
        ScalarType::BFloat16, X.scalar_type(), "GroupNormBackwardKernelImpl", [&]() {
        if(mixed_type) {
          GroupNormBackwardKernelImplInternal<BFloat16, float>(
              dY, X, mean, rstd, gamma, N, C, HxW, group, dX, dgamma, dbeta);
        } else {
          GroupNormBackwardKernelImplInternal<scalar_t, scalar_t>(
              dY, X, mean, rstd, gamma, N, C, HxW, group, dX, dgamma, dbeta);
        }
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast:
    case at::MemoryFormat::ChannelsLast3d: {
      // std::cout << "ChannelsLast nd\n";
      AT_DISPATCH_FLOATING_TYPES_AND(
        ScalarType::BFloat16, X.scalar_type(), "GroupNormBackwardKernelImpl", [&]() {
        if(mixed_type) {
          GroupNormBackwardKernelImplChannelsLastInternal<BFloat16, float>(
              dY, X, mean, rstd, gamma, N, C, HxW, group, dX, dgamma, dbeta);
        } else {
          GroupNormBackwardKernelImplChannelsLastInternal<scalar_t, scalar_t>(
              dY, X, mean, rstd, gamma, N, C, HxW, group, dX, dgamma, dbeta);
        }
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, ChannelsLast3d, Contiguous");
  }

}

} // namespace

REGISTER_DISPATCH(GroupNormKernel, &GroupNormKernelImpl);
REGISTER_DISPATCH(GroupNormBackwardKernel, &GroupNormBackwardKernelImpl);

} // namespace native
} // namespace at
