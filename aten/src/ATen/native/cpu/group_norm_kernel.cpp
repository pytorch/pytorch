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
#include <ATen/OpMathType.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

namespace at::native {

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
  const T* X_data = X.const_data_ptr<T>();
  const PT* gamma_data = gamma.defined() ? gamma.const_data_ptr<PT>() : nullptr;
  const PT* beta_data = beta.defined() ? beta.const_data_ptr<PT>() : nullptr;
  T* Y_data = Y.data_ptr<T>();
  PT* mean_data = mean.data_ptr<PT>();
  PT* rstd_data = rstd.data_ptr<PT>();
  const bool gamma_null = (gamma_data == nullptr);
  const bool beta_null = beta_data == nullptr;
  const int64_t inner_size = D * HxW;

  using opmath_t = at::opmath_type<T>;

  at::parallel_for(0, N * G, 1, [&](int64_t start, int64_t end) {
    for (const auto i : c10::irange(start, end)) {
      const T* X_ptr = X_data + i * inner_size;
      auto [mean_val, rstd_val] = RowwiseMoments(X_ptr, inner_size);
      rstd_val = opmath_t(1) / std::sqrt(std::max(rstd_val, opmath_t(0)) + eps);
      if (gamma_null && beta_null) {
        T* Y_ptr = Y_data + i * inner_size;
        for (const auto j : c10::irange(inner_size)) {
          Y_ptr[j] = (X_ptr[j] - mean_val) * rstd_val;
        }
      } else {
        const int64_t g = i % G;
        for (const auto j : c10::irange(D)) {
          const int64_t c = g * D + j;
          const opmath_t scale = rstd_val * (gamma_null ? opmath_t(1) : opmath_t(gamma_data[c]));
          const opmath_t bias = -scale * mean_val + (beta_null ? opmath_t(0) : opmath_t(beta_data[c]));
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
typename std::enable_if<std::is_same<T, at::opmath_type<T>>::value,
  std::tuple<T, T>>::type
ColumnwiseMoments(
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


// std::is_same<T, at::BFloat16> || std::is_same<T, at::Half>
template <typename T>
typename std::enable_if<!std::is_same<T, at::opmath_type<T>>::value,
  std::tuple<at::opmath_type<T>, at::opmath_type<T>>>::type
ColumnwiseMoments(
    const T* X_data,
    int64_t HxW,
    int64_t C,
    int64_t D) {
  using opmath_t = at::opmath_type<T>;
  using Vec = vec::Vectorized<T>;
  using fVec = vec::Vectorized<opmath_t>;
  constexpr int64_t K = Vec::size();
  const int64_t inner_size = D / K * K;
  fVec acc0_fvec{0}, acc1_fvec{0}, zero{0};
  for (const auto m : c10::irange(HxW)) {
    const T* X_ptr = X_data + m * C;
    int64_t d = 0;
    for (; d < inner_size; d += K) {
      Vec x_bvec = Vec::loadu(X_ptr + d);
      auto [x_fvec0, x_fvec1] = convert_to_float<T>(x_bvec);
      acc0_fvec += x_fvec0 + x_fvec1;
      acc1_fvec += x_fvec0 * x_fvec0 + x_fvec1 * x_fvec1;
    }
    if (D - d > 0) {
      Vec x_bvec = Vec::loadu(X_ptr + d, D - d);
      auto [x_fvec0, x_fvec1] = convert_to_float<T>(x_bvec);
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
  opmath_t mean_val = vec::vec_reduce_all([](fVec& x, fVec& y) { return x + y; }, acc0_fvec);
  opmath_t rstd_val = vec::vec_reduce_all([](fVec& x, fVec& y) { return x + y; }, acc1_fvec);
  return std::tuple<opmath_t, opmath_t>(mean_val, rstd_val);
}

template <typename T, typename opmath_t>
inline typename std::enable_if<std::is_same<T, opmath_t>::value, void>::type
CalcMeanVar(
  const T* X_ptr,
  opmath_t* mean_ptr,
  opmath_t* rstd_ptr,
  int64_t C) {
  using Vec = vec::Vectorized<T>;
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
}

// std::is_same<T, at::BFloat16> || std::is_same<T, at::Half>
template <typename T, typename opmath_t>
inline typename std::enable_if<!std::is_same<T, opmath_t>::value, void>::type
CalcMeanVar(
  const T* X_ptr,
  opmath_t* mean_ptr,
  opmath_t* rstd_ptr,
  int64_t C) {
  using fVec = vec::Vectorized<opmath_t>;
  using Vec = vec::Vectorized<T>;
  int64_t d = 0;
  for (; d < C - (C % Vec::size()); d += Vec::size()) {
    Vec data_bvec = Vec::loadu(X_ptr + d);
    fVec mean_fvec0 = fVec::loadu(mean_ptr + d);
    fVec mean_fvec1 = fVec::loadu(mean_ptr + d + fVec::size());
    fVec rstd_fvec0 = fVec::loadu(rstd_ptr + d);
    fVec rstd_fvec1 = fVec::loadu(rstd_ptr + d + fVec::size());
    auto [data_fvec0, data_fvec1] = convert_to_float<T>(data_bvec);
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
    Vec data_bvec = Vec::loadu(X_ptr + d, C - d);
    fVec mean_fvec0 = fVec::loadu(mean_ptr + d, (C - d) > fVec::size() ? fVec::size() : (C - d));
    fVec mean_fvec1 = fVec::loadu(mean_ptr + d + fVec::size(), (C - d) > fVec::size() ? (C - d - fVec::size()) : 0);
    fVec rstd_fvec0 = fVec::loadu(rstd_ptr + d, (C - d) > fVec::size() ? fVec::size() : (C - d));
    fVec rstd_fvec1 = fVec::loadu(rstd_ptr + d + fVec::size(), (C - d) > fVec::size() ? (C - d - fVec::size()) : 0);
    auto [data_fvec0, data_fvec1] = convert_to_float<T>(data_bvec);
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

template <typename T, typename opmath_t>
inline typename std::enable_if<std::is_same<T, opmath_t>::value, void>::type
ApplyScaleBias(
  T* Y_ptr,
  const T* X_ptr,
  const opmath_t* scale_ptr,
  const opmath_t* bias_ptr,
  int64_t C) {
  using Vec = vec::Vectorized<T>;
  vec::map3<T>(
    [](Vec x, Vec scale, Vec bias) { return x * scale + bias; },
    Y_ptr,
    X_ptr,
    scale_ptr,
    bias_ptr,
    C);
}

// std::is_same<T, at::BFloat16> || std::is_same<T, at::Half>
template <typename T, typename opmath_t>
inline typename std::enable_if<!std::is_same<T, opmath_t>::value, void>::type
ApplyScaleBias(
  T* Y_ptr,
  const T* X_ptr,
  const opmath_t* scale_ptr,
  const opmath_t* bias_ptr,
  int64_t C) {
  using fVec = vec::Vectorized<opmath_t>;
  using Vec = vec::Vectorized<T>;
  int64_t d = 0;
  for (; d < C - (C % Vec::size()); d += Vec::size()) {
    Vec data_bvec = Vec::loadu(X_ptr + d);
    fVec scale_fvec0 = fVec::loadu(scale_ptr + d);
    fVec scale_fvec1 = fVec::loadu(scale_ptr + d + fVec::size());
    fVec bias_fvec0 = fVec::loadu(bias_ptr + d);
    fVec bias_fvec1 = fVec::loadu(bias_ptr + d + fVec::size());
    auto [data_fvec0, data_fvec1] = convert_to_float<T>(data_bvec);
    fVec out0 = data_fvec0 * scale_fvec0 + bias_fvec0;
    fVec out1 = data_fvec1 * scale_fvec1 + bias_fvec1;
    convert_from_float<T>(out0, out1).store(Y_ptr + d);
  }
  if (C - d > 0) {
    Vec data_bvec = Vec::loadu(X_ptr + d, C - d);
    fVec scale_fvec0 = fVec::loadu(scale_ptr + d, (C - d) > fVec::size() ? fVec::size() : (C - d));
    fVec scale_fvec1 = fVec::loadu(scale_ptr + d + fVec::size(), (C - d) > fVec::size() ? (C - d - fVec::size()) : 0);
    fVec bias_fvec0 = fVec::loadu(bias_ptr + d, (C - d) > fVec::size() ? fVec::size() : (C - d));
    fVec bias_fvec1 = fVec::loadu(bias_ptr + d + fVec::size(), (C - d) > fVec::size() ? (C - d - fVec::size()) : 0);
    auto [data_fvec0, data_fvec1] = convert_to_float<T>(data_bvec);
    fVec out0 = data_fvec0 * scale_fvec0 + bias_fvec0;
    fVec out1 = data_fvec1 * scale_fvec1 + bias_fvec1;
    convert_from_float<T>(out0, out1).store(Y_ptr + d, C - d);
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
  const T* X_data = X.const_data_ptr<T>();
  const PT* gamma_data = gamma.defined() ? gamma.const_data_ptr<PT>() : nullptr;
  const PT* beta_data = beta.defined() ? beta.const_data_ptr<PT>() : nullptr;
  T* Y_data = Y.data_ptr<T>();
  PT* mean_data = mean.data_ptr<PT>();
  PT* rstd_data = rstd.data_ptr<PT>();

  using opmath_t = at::opmath_type<T>;

  const opmath_t s = opmath_t(1) / static_cast<opmath_t>(D * HxW);
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
    Tensor buffer = at::empty({N * G, 2 * D}, X.options().dtype(c10::CppTypeToScalarType<opmath_t>::value));
    opmath_t* buffer_data = buffer.data_ptr<opmath_t>();

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
        auto [mean_val, rstd_val] = ColumnwiseMoments(
                X_data + n * HxW * C + g * D,
                HxW,
                C,
                D);

        mean_val *= s;
        rstd_val = std::max(rstd_val * s - mean_val * mean_val, opmath_t(0));
        rstd_val = opmath_t(1) / std::sqrt(rstd_val + eps);
        mean_data[i] = mean_val;
        rstd_data[i] = rstd_val;

        // step-2: calculate scale and bias
        opmath_t* scale_ptr = buffer_data + i * 2 * D;
        opmath_t* bias_ptr = scale_ptr + D;
        for (const auto d : c10::irange(D)) {
          const int64_t c = g * D + d;
          scale_ptr[d] = rstd_val * (gamma_null ? opmath_t(1) : opmath_t(gamma_data[c]));
          bias_ptr[d] = -scale_ptr[d] * mean_val + (beta_null ? opmath_t(0) : opmath_t(beta_data[c]));
        }

        // step-3: apply scale and bias
        for (const auto m : c10::irange(HxW)) {
          const T* X_ptr = X_data + n * HxW * C + m * C + g * D;
          T* Y_ptr = Y_data + n * HxW * C + m * C + g * D;
          ApplyScaleBias<T, opmath_t>(Y_ptr, X_ptr, scale_ptr, bias_ptr, D);
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
      X.options().dtype(c10::CppTypeToScalarType<opmath_t>::value)).zero_();
    opmath_t* buffer_data = buffer.data_ptr<opmath_t>();
    Tensor tmp_buffer = at::empty({N, 2 * G},
      X.options().dtype(c10::CppTypeToScalarType<opmath_t>::value));
    opmath_t* tmp_buffer_data = tmp_buffer.data_ptr<opmath_t>();
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
      opmath_t* buffer_ptr = buffer_data + tid * N * 2 * C;

      int64_t n{0}, m{0};
      data_index_init(begin, n, N, m, HxW);
      for (const auto i : c10::irange(begin, end)) {
        opmath_t* mean_ptr = buffer_ptr + n * 2 * C;
        opmath_t* rstd_ptr = mean_ptr + C;
        const T* X_ptr = X_data + i * C;
        CalcMeanVar<T, opmath_t>(X_ptr, mean_ptr, rstd_ptr, C);
        data_index_step(n, N, m, HxW);
      }
    });

    // step-2: compute mean and rstd
    for (const auto n : c10::irange(N)) {
      for (const auto g : c10::irange(G)) {
        opmath_t mean_val{0}, rstd_val{0};
        for (const auto d : c10::irange(D)) {
          for (const auto t : c10::irange(num_threads)) {
            opmath_t* buffer_ptr = buffer_data + t * N * 2 * C + n * 2 * C;
            mean_val += buffer_ptr[g * D + d];
            rstd_val += buffer_ptr[g * D + d + C];
           }
        }
        mean_val *= s;
        rstd_val = std::max(rstd_val * s - mean_val * mean_val, opmath_t(0));
        rstd_val = opmath_t(1) / std::sqrt(rstd_val + eps);
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
    //   b. Avoid duplicate calculation of scale/bias, each HxW plain share the same scale/bias
    //
    for (const auto n : c10::irange(N)) {
      for (const auto g : c10::irange(G)) {
        opmath_t* scale_ptr = buffer_data + n * 2 * C;
        opmath_t* bias_ptr = scale_ptr + C;
        opmath_t mean_val = tmp_buffer_data[n * 2 * G + 2 * g];
        opmath_t rstd_val = tmp_buffer_data[n * 2 * G + 2 * g + 1];
        mean_data[n * G + g] = mean_val;
        rstd_data[n * G + g] = rstd_val;

        for (const auto d : c10::irange(D)) {
          const int64_t c = g * D + d;
          scale_ptr[c] = rstd_val * (gamma_null ? opmath_t(1) : opmath_t(gamma_data[c]));
          bias_ptr[c] = -scale_ptr[c] * mean_val + (beta_null ? opmath_t(0) : opmath_t(beta_data[c]));
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
        opmath_t* scale_ptr = buffer_data + n * 2 * C;
        opmath_t* bias_ptr = scale_ptr + C;
        ApplyScaleBias<T, opmath_t>(Y_ptr, X_ptr, scale_ptr, bias_ptr, C);
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
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, X.scalar_type(), "GroupNormKernelImpl", [&]() {
        using param_t = at::opmath_type<scalar_t>;
        if (mixed_type) {
          GroupNormKernelImplInternal<scalar_t, param_t>(
              X, gamma, beta, N, C, HxW, group, eps, Y, mean, rstd);
        } else {
          GroupNormKernelImplInternal<scalar_t, scalar_t>(
              X, gamma, beta, N, C, HxW, group, eps, Y, mean, rstd);
        }
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast:
    case at::MemoryFormat::ChannelsLast3d: {
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, X.scalar_type(), "GroupNormKernelImpl", [&]() {
        using param_t = at::opmath_type<scalar_t>;
        if (mixed_type) {
          GroupNormKernelImplChannelsLastInternal<scalar_t, param_t>(
              X, gamma, beta, N, C, HxW, group, eps, Y, mean, rstd);
        } else {
          GroupNormKernelImplChannelsLastInternal<scalar_t, scalar_t>(
              X, gamma, beta, N, C, HxW, group, eps, Y, mean, rstd);
        }
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, ChannelsLast3d, Contiguous");
  }
}


template <typename T, typename opmath_t>
typename std::enable_if<std::is_same<T, opmath_t>::value, void>::type
ComputeInternalGradients(
    int64_t N,
    int64_t C,
    int64_t HxW,
    const T* dY,
    const T* X,
    opmath_t* ds,
    opmath_t* db) {
  using Vec = at::vec::Vectorized<opmath_t>;
  at::parallel_for(0, N * C, 1, [=](int64_t start, int64_t end) {
    for (const auto i : c10::irange(start, end)) {
      const T* dY_ptr = dY + i * HxW;
      const T* X_ptr = X + i * HxW;
      ds[i] = at::vec::map2_reduce_all<T>(
          [](Vec x, Vec y) { return x * y; },
          [](Vec x, Vec y) { return x + y; },
          dY_ptr,
          X_ptr,
          HxW);
      db[i] = at::vec::reduce_all<T>(
          [](Vec& x, Vec& y) { return x + y; }, dY_ptr, HxW);
    }
  });
}

template <typename T, typename opmath_t>
typename std::enable_if<!std::is_same<T, opmath_t>::value, void>::type
ComputeInternalGradients(
    int64_t N,
    int64_t C,
    int64_t HxW,
    const T* dY,
    const T* X,
    opmath_t* ds,
    opmath_t* db) {
  using Vec = vec::Vectorized<T>;
  using fVec = vec::Vectorized<opmath_t>;
  at::parallel_for(0, N * C, 1, [=](int64_t start, int64_t end) {
    constexpr int64_t K = Vec::size();
    const int64_t inner_size = HxW / K * K;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    std::array<opmath_t, K / 2> ds_arr;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    std::array<opmath_t, K / 2> db_arr;
    for (const auto i : c10::irange(start, end)) {
      const T* dY_ptr = dY + i * HxW;
      const T* X_ptr = X + i * HxW;
      fVec ds_vec(0);
      fVec db_vec(0);
      for (int64_t j = 0; j < inner_size; j += K) {
        const Vec dy_bvec = Vec::loadu(dY_ptr + j);
        const Vec x_bvec = Vec::loadu(X_ptr + j);
        auto [x_fvec0, x_fvec1] = convert_to_float<T>(x_bvec);
        auto [dy_fvec0, dy_fvec1] = convert_to_float<T>(dy_bvec);
        ds_vec = ds_vec + dy_fvec0 * x_fvec0;
        ds_vec = ds_vec + dy_fvec1 * x_fvec1;
        db_vec = db_vec + dy_fvec0 + dy_fvec1;
      }
      ds_vec.store(ds_arr.data());
      db_vec.store(db_arr.data());
      opmath_t ds_val = std::accumulate(ds_arr.cbegin(), ds_arr.cend(), opmath_t(0));
      opmath_t db_val = std::accumulate(db_arr.cbegin(), db_arr.cend(), opmath_t(0));
      for (const auto j : c10::irange(inner_size, HxW)) {
        ds_val += opmath_t(dY_ptr[j]) * opmath_t(X_ptr[j]);
        db_val += opmath_t(dY_ptr[j]);
      }
      ds[i] = ds_val;
      db[i] = db_val;
    }
  });
}

template <typename PT, typename opmath_t>
inline typename std::enable_if<std::is_same<PT, opmath_t>::value, void>::type
CalcDsDb(
    const opmath_t* ds_ptr,
    const opmath_t* db_ptr,
    const PT* gamma_ptr,
    const int64_t d,
    const int64_t K,
    void* ds_arr,
    void* db_arr) {
    vec::Vectorized<opmath_t> ds_vec(0);
    vec::Vectorized<opmath_t> db_vec(0);
    for (int64_t j = 0; j < d; j += K) {
      const vec::Vectorized<PT> gamma_vec = (gamma_ptr == nullptr)
          ? vec::Vectorized<PT>(1)
          : vec::Vectorized<PT>::loadu(gamma_ptr + j);
      ds_vec = ds_vec + vec::Vectorized<PT>::loadu(ds_ptr + j) * gamma_vec;
      db_vec = db_vec + vec::Vectorized<PT>::loadu(db_ptr + j) * gamma_vec;
    }
    ds_vec.store(ds_arr);
    db_vec.store(db_arr);
}

template <typename PT, typename opmath_t>
inline typename std::enable_if<!std::is_same<PT, opmath_t>::value, void>::type
CalcDsDb(
    const opmath_t* ds_ptr,
    const opmath_t* db_ptr,
    const PT* gamma_ptr,
    const int64_t d,
    const int64_t K,
    void* ds_arr,
    void* db_arr) {
  using fVec = at::vec::Vectorized<opmath_t>;
  using Vec = at::vec::Vectorized<PT>;
  fVec ds_acc(0);
  fVec db_acc(0);
  for (int64_t j = 0; j < d; j += K) {
    const Vec gamma_vec = (gamma_ptr == nullptr) ? Vec(1) : Vec::loadu(gamma_ptr + j);
    auto [gamma_vec0, gamma_vec1] = convert_to_float<PT>(gamma_vec);
    ds_acc += fVec::loadu(ds_ptr + j) * gamma_vec0;
    ds_acc += fVec::loadu(ds_ptr + j + fVec::size()) * gamma_vec1;
    db_acc += fVec::loadu(db_ptr + j) * gamma_vec0;
    db_acc += fVec::loadu(db_ptr + j + fVec::size()) * gamma_vec1;
  }
  ds_acc.store(ds_arr);
  db_acc.store(db_arr);
}

template <typename T, typename PT, typename opmath_t>
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
    const opmath_t* ds,
    const opmath_t* db,
    T* dX) {
  const int64_t G = group;
  const int64_t D = C / G;
  const opmath_t s = opmath_t(1) / static_cast<opmath_t>(D * HxW);
  const bool gamma_null = (gamma == nullptr);
  at::parallel_for(0, N * G, 1, [=](int64_t start, int64_t end) {
    constexpr int64_t K = vec::Vectorized<PT>::size();
    const int64_t d = D / K * K;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    std::array<opmath_t, at::vec::Vectorized<opmath_t>::size()> ds_arr;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    std::array<opmath_t, at::vec::Vectorized<opmath_t>::size()> db_arr;
    for (const auto i : c10::irange(start, end)) {
      const int64_t g = i % G;
      const opmath_t* ds_ptr = ds + i * D;
      const opmath_t* db_ptr = db + i * D;
      const PT* gamma_ptr = gamma_null ? nullptr : (gamma + g * D);
      CalcDsDb(ds_ptr, db_ptr, gamma_ptr, d, K, ds_arr.data(), db_arr.data());
      opmath_t ds_val = std::accumulate(ds_arr.cbegin(), ds_arr.cend(), opmath_t(0));
      opmath_t db_val = std::accumulate(db_arr.cbegin(), db_arr.cend(), opmath_t(0));
      for (const auto j : c10::irange(d, D)) {
        const opmath_t gamma_v = gamma_null ? opmath_t(1) : opmath_t(gamma[g * D + j]);
        ds_val += ds_ptr[j] * gamma_v;
        db_val += db_ptr[j] * gamma_v;
      }
      const opmath_t c2 =
          (db_val * opmath_t(mean[i]) - ds_val) * opmath_t(rstd[i]) * opmath_t(rstd[i]) * opmath_t(rstd[i]) * s;
      const opmath_t c3 = -c2 * opmath_t(mean[i]) - db_val * opmath_t(rstd[i]) * s;

      for (const auto j : c10::irange(D)) {
        const int64_t c = g * D + j;
        const T* dY_ptr = dY + (i * D + j) * HxW;
        const T* X_ptr = X + (i * D + j) * HxW;
        T* dX_ptr = dX + (i * D + j) * HxW;
        const opmath_t c1 = opmath_t(rstd[i]) * (gamma_null ? opmath_t(1) : opmath_t(gamma[c]));
        for (const auto k : c10::irange(HxW)) {
          dX_ptr[k] = c1 * opmath_t(dY_ptr[k]) + c2 * opmath_t(X_ptr[k]) + c3;
        }
      }
    }
  });
}

template <typename PT, typename opmath_t>
typename std::enable_if<std::is_same<PT, opmath_t>::value, void>::type
GammaBackward(
    int64_t N,
    int64_t C,
    int64_t group,
    const PT* mean,
    const PT* rstd,
    const opmath_t* ds,
    const opmath_t* db,
    PT* dgamma) {
  const int64_t G = group;
  const int64_t D = C / G;
  constexpr int64_t K = at::vec::Vectorized<PT>::size();
  using Vec = at::vec::Vectorized<PT>;
  const int64_t inner_size = D / K * K;
  for (const auto g : c10::irange(G)) {
    int64_t i = 0;
    for (; i < inner_size; i += K) {
      Vec acc_vec{0};
      for (const auto n : c10::irange(N)) {
        const PT* ds_ptr = ds + n * C + g * D + i;
        const PT* db_ptr = db + n * C + g * D + i;
        auto ds_vec = Vec::loadu(ds_ptr);
        auto db_vec = Vec::loadu(db_ptr);
        auto mean_vec = Vec(mean[n * G + g]);
        auto rstd_vec = Vec(rstd[n * G + g]);
        acc_vec += (ds_vec - db_vec * mean_vec) * rstd_vec;
      }
      acc_vec.store(dgamma + g * D + i);
    }
    if (D - i > 0) {
      Vec acc_vec{0};
      for (const auto n : c10::irange(N)) {
        const PT* ds_ptr = ds + n * C + g * D + i;
        const PT* db_ptr = db + n * C + g * D + i;
        auto ds_vec = Vec::loadu(ds_ptr, D - i);
        auto db_vec = Vec::loadu(db_ptr, D - i);
        auto mean_vec = Vec(mean[n * G + g]);
        auto rstd_vec = Vec(rstd[n * G + g]);
        acc_vec += (ds_vec - db_vec * mean_vec) * rstd_vec;
      }
      acc_vec.store(dgamma + g * D + i, D - i);
    }
  }
}

template <typename PT, typename opmath_t>
typename std::enable_if<!std::is_same<PT, opmath_t>::value, void>::type
GammaBackward(
    int64_t N,
    int64_t C,
    int64_t group,
    const PT* mean,
    const PT* rstd,
    const opmath_t* ds,
    const opmath_t* db,
    PT* dgamma) {
  const int64_t G = group;
  const int64_t D = C / G;
  using Vec = at::vec::Vectorized<PT>;
  using fVec = at::vec::Vectorized<opmath_t>;
  constexpr int64_t K = Vec::size();
  const int64_t inner_size = D / K * K;
  for (const auto g : c10::irange(G)) {
    int64_t i = 0;
    for (; i < inner_size; i += K) {
      fVec acc0_vec{0}, acc1_vec{0};
      for (const auto n : c10::irange(N)) {
        const opmath_t* ds_ptr = ds + n * C + g * D + i;
        const opmath_t* db_ptr = db + n * C + g * D + i;
        fVec ds_vec0, ds_vec1, db_vec0, db_vec1;
        ds_vec0 = fVec::loadu(ds_ptr);
        ds_vec1 = fVec::loadu(ds_ptr + fVec::size());
        db_vec0 = fVec::loadu(db_ptr);
        db_vec1 = fVec::loadu(db_ptr + fVec::size());
        fVec mean_vec = fVec(opmath_t(mean[n * G + g]));
        fVec rstd_vec = fVec(opmath_t(rstd[n * G + g]));
        acc0_vec += (ds_vec0 - db_vec0 * mean_vec) * rstd_vec;
        acc1_vec += (ds_vec1 - db_vec1 * mean_vec) * rstd_vec;
      }
      convert_from_float<PT>(acc0_vec, acc1_vec).store(dgamma + g * D + i);
    }
    if (D - i > 0) {
      fVec acc0_vec{0}, acc1_vec{0};
      for (const auto n : c10::irange(N)) {
        const opmath_t* ds_ptr = ds + n * C + g * D + i;
        const opmath_t* db_ptr = db + n * C + g * D + i;
        fVec ds_vec0, ds_vec1, db_vec0, db_vec1;
        ds_vec0 = fVec::loadu(
            ds_ptr, (D - i) > fVec::size() ? fVec::size() : (D - i));
        ds_vec1 = fVec::loadu(
            ds_ptr + fVec::size(),
            (D - i) > fVec::size() ? (D - i - fVec::size()) : 0);
        db_vec0 = fVec::loadu(
            db_ptr, (D - i) > fVec::size() ? fVec::size() : (D - i));
        db_vec1 = fVec::loadu(
            db_ptr + fVec::size(),
            (D - i) > fVec::size() ? (D - i - fVec::size()) : 0);
        fVec mean_vec = fVec(opmath_t(mean[n * G + g]));
        fVec rstd_vec = fVec(opmath_t(rstd[n * G + g]));
        acc0_vec += (ds_vec0 - db_vec0 * mean_vec) * rstd_vec;
        acc1_vec += (ds_vec1 - db_vec1 * mean_vec) * rstd_vec;
      }
      convert_from_float<PT>(acc0_vec, acc1_vec).store(dgamma + g * D + i, D - i);
    }
  }
}

template <typename PT, typename opmath_t>
typename std::enable_if<std::is_same<PT, opmath_t>::value, void>::type
BetaBackward(int64_t N, int64_t C, const opmath_t* db, PT* dbeta) {
  using Vec = at::vec::Vectorized<PT>;
  constexpr int64_t K = Vec::size();
  Vec acc_vec{0}, zero{0};
  const int64_t inner_size = C / K * K;
  int64_t i = 0;
  for (; i < inner_size; i += K) {
    for (const auto n : c10::irange(N)) {
      acc_vec += Vec::loadu(db + n * C + i);
    }
    acc_vec.store(dbeta + i);
    acc_vec = Vec::set(acc_vec, zero);
  }
  if (C - i > 0) {
    for (const auto n : c10::irange(N)) {
      acc_vec += Vec::loadu(db + n * C + i, C - i);
    }
    acc_vec.store(dbeta + i, C - i);
    acc_vec = Vec::set(acc_vec, zero, C - i);
  }
}

template <typename PT, typename opmath_t>
typename std::enable_if<!std::is_same<PT, opmath_t>::value, void>::type
BetaBackward(int64_t N, int64_t C, const opmath_t* db, PT* dbeta) {
  using Vec = at::vec::Vectorized<PT>;
  using fVec = at::vec::Vectorized<opmath_t>;
  constexpr int64_t K = Vec::size();
  fVec acc0_vec{0}, acc1_vec{0}, zero{0};
  const int64_t inner_size = C / K * K;
  int64_t i = 0;
  for (; i < inner_size; i += K) {
    for (const auto n : c10::irange(N)) {
      fVec db_vec0, db_vec1;
      db_vec0 = fVec::loadu(db + n * C + i);
      db_vec1 = fVec::loadu(db + n * C + i + fVec::size());
      acc0_vec += db_vec0;
      acc1_vec += db_vec1;
    }
    convert_from_float<PT>(acc0_vec, acc1_vec).store(dbeta + i);
    acc0_vec = fVec::set(acc0_vec, zero);
    acc1_vec = fVec::set(acc1_vec, zero);
  }
  if (C - i > 0) {
    for (const auto n : c10::irange(N)) {
      fVec db_vec0, db_vec1;
      db_vec0 = fVec::loadu(
          db + n * C + i, (C - i) > fVec::size() ? fVec::size() : (C - i));
      db_vec1 = fVec::loadu(
          db + n * C + i + fVec::size(),
          (C - i) > fVec::size() ? (C - i - fVec::size()) : 0);
      acc0_vec += db_vec0;
      acc1_vec += db_vec1;
    }
    convert_from_float<PT>(acc0_vec, acc1_vec).store(dbeta + i, C - i);
    acc0_vec = fVec::set(acc0_vec, zero, C - i);
    acc1_vec = fVec::set(acc1_vec, zero, C - i);
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
  const T* dY_data = dY.const_data_ptr<T>();
  const T* X_data = X.const_data_ptr<T>();
  const PT* mean_data = mean.const_data_ptr<PT>();
  const PT* rstd_data = rstd.const_data_ptr<PT>();
  const PT* gamma_data = gamma.defined() ? gamma.const_data_ptr<PT>() : nullptr;
  T* dX_data = dX.defined() ? dX.data_ptr<T>() : nullptr;
  PT* dgamma_data = dgamma.defined() ? dgamma.data_ptr<PT>() : nullptr;
  PT* dbeta_data = dbeta.defined() ? dbeta.data_ptr<PT>() : nullptr;
  using opmath_t = at::opmath_type<T>;
  Tensor ds = at::empty({N, C}, X.options().dtype(c10::CppTypeToScalarType<opmath_t>::value));
  Tensor db = at::empty({N, C}, X.options().dtype(c10::CppTypeToScalarType<opmath_t>::value));
  opmath_t* ds_data = ds.data_ptr<opmath_t>();
  opmath_t* db_data = db.data_ptr<opmath_t>();
  ComputeInternalGradients<T, opmath_t>(N, C, HxW, dY_data, X_data, ds_data, db_data);

  if (dX_data != nullptr) {
    GroupNormInputBackward<T, PT, opmath_t>(
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
    GammaBackward(
        N, C, group, mean_data, rstd_data, ds_data, db_data, dgamma_data);
  }
  if (dbeta_data != nullptr) {
    BetaBackward(N, C, db_data, dbeta_data);
  }
}

template <typename T, typename opmath_t>
inline typename std::enable_if<std::is_same<T, opmath_t>::value, void>::type
DsDbRowwiseMomentsChannelsLast(
  const T* dY_ptr,
  const T* X_ptr,
  opmath_t* ds_ptr,
  opmath_t* db_ptr,
  int64_t C) {
  using Vec = vec::Vectorized<T>;
  constexpr int64_t K = vec::Vectorized<T>::size();
  const int64_t inner_size = C / K * K;
  int64_t d = 0;
  for (; d < inner_size; d += K) {
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

template <typename T, typename opmath_t>
inline typename std::enable_if<!std::is_same<T, opmath_t>::value, void>::type
DsDbRowwiseMomentsChannelsLast(
  const T* dY_ptr,
  const T* X_ptr,
  opmath_t* ds_ptr,
  opmath_t* db_ptr,
  int64_t C) {
  using fVec = vec::Vectorized<opmath_t>;
  using Vec = vec::Vectorized<T>;
  int64_t d = 0;
  for (; d < C - (C % Vec::size()); d += Vec::size()) {
    fVec ds_dev0 = fVec::loadu(ds_ptr + d);
    fVec ds_dev1 = fVec::loadu(ds_ptr + d + fVec::size());
    fVec db_vec0 = fVec::loadu(db_ptr + d);
    fVec db_vec1 = fVec::loadu(db_ptr + d + fVec::size());
    Vec x_vec = Vec::loadu(X_ptr + d);
    Vec dy_vec = Vec::loadu(dY_ptr + d);
    auto [x_vec0, x_vec1] = convert_to_float<T>(x_vec);
    auto [dy_vec0, dy_vec1] = convert_to_float<T>(dy_vec);
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
    Vec x_vec = Vec::loadu(X_ptr + d, C - d);
    Vec dy_vec = Vec::loadu(dY_ptr + d, C - d);
    auto [x_vec0, x_vec1] = convert_to_float<T>(x_vec);
    auto [dy_vec0, dy_vec1] = convert_to_float<T>(dy_vec);
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

template <typename T>
inline typename std::enable_if<std::is_same<T, at::opmath_type<T>>::value,
  std::tuple<
  vec::Vectorized<T>,
  vec::Vectorized<T>>>::type
load_util(const T* data_ptr, int64_t n) {
  using Vec = vec::Vectorized<T>;
  auto vec0 = Vec::loadu(data_ptr, n > Vec::size() ? Vec::size() : n);
  auto vec1 = Vec::loadu(
      data_ptr + Vec::size(), n > Vec::size() ? (n - Vec::size()) : 0);
  return std::tuple<Vec, Vec>(vec0, vec1);
}

template <typename T>
inline typename std::enable_if<!std::is_same<T, at::opmath_type<T>>::value,
  std::tuple<
    vec::Vectorized<at::opmath_type<T>>,
    vec::Vectorized<at::opmath_type<T>>>
    >::type
load_util(const T* data_ptr, int64_t n) {
  using Vec = vec::Vectorized<T>;
  auto vec = Vec::loadu(data_ptr, n);
  return convert_to_float<T>(vec);
}

template <typename T, typename PT, typename opmath_t>
inline typename std::enable_if<std::is_same<T, opmath_t>::value, void>::type
ApplyInputGradientsChannelsLastColMov(
  const T* dY_data,
  const T* X_data,
  T* dX_data,
  const PT* rstd,
  const PT* gamma,
  opmath_t c2,
  opmath_t c3,
  int64_t HxW,
  int64_t C,
  int64_t D) {
  const bool gamma_null = (gamma == nullptr);
  int64_t d = 0;
  auto K = vec::Vectorized<T>::size();
  for (; d < D / K * K; d += K) {
    auto c1 = vec::Vectorized<T>(*rstd) *
        (gamma_null ? vec::Vectorized<T>(1)
                    : vec::Vectorized<T>::loadu(gamma + d));
    for (const auto m : c10::irange(HxW)) {
      const T* X_ptr = X_data + m * C;
      const T* dY_ptr = dY_data + m * C;
      T* dX_ptr = dX_data + m * C;
      auto dy_vec = vec::Vectorized<T>::loadu(dY_ptr + d);
      auto x_vec = vec::Vectorized<T>::loadu(X_ptr + d);
      auto dx_vec = c1 * dy_vec +
        vec::Vectorized<T>(c2) * x_vec + vec::Vectorized<T>(c3);
      dx_vec.store(dX_ptr + d);
    }
  }
  if (D - d > 0) {
    auto c1 = vec::Vectorized<T>(*rstd) *
        (gamma_null ? vec::Vectorized<T>(1)
                    : vec::Vectorized<T>::loadu(gamma + d, D - d));
    for (const auto m : c10::irange(HxW)) {
      const T* X_ptr = X_data + m * C;
      const T* dY_ptr = dY_data + m * C;
      T* dX_ptr = dX_data + m * C;
    auto dy_vec = vec::Vectorized<T>::loadu(dY_ptr + d, D - d);
    auto x_vec = vec::Vectorized<T>::loadu(X_ptr + d, D - d);
    auto dx_vec = c1 * dy_vec +
      vec::Vectorized<T>(c2) * x_vec + vec::Vectorized<T>(c3);
    dx_vec.store(dX_ptr + d, D - d);
    }
  }
}

template <typename T, typename PT, typename opmath_t>
inline typename std::enable_if<!std::is_same<T, opmath_t>::value, void>::type
ApplyInputGradientsChannelsLastColMov(
    const T* dY_data,
    const T* X_data,
    T* dX_data,
    const PT* rstd,
    const PT* gamma,
    opmath_t c2,
    opmath_t c3,
    int64_t HxW,
    int64_t C,
    int64_t D) {
  using Vec = vec::Vectorized<T>;
  using fVec = vec::Vectorized<opmath_t>;
  const bool gamma_null = (gamma == nullptr);
  auto K = Vec::size();
  int64_t d = 0;
  for (; d < D / K * K; d += K) {
    auto [c1_0, c1_1] = gamma_null ? std::tuple<fVec, fVec>(fVec(1), fVec(1))
                                      : load_util(gamma + d, K);
    c1_0 = c1_0 * fVec(opmath_t(*rstd));
    c1_1 = c1_1 * fVec(opmath_t(*rstd));
    for (const auto m : c10::irange(HxW)) {
      const T* X_ptr = X_data + m * C;
      const T* dY_ptr = dY_data + m * C;
      T* dX_ptr = dX_data + m * C;

      Vec dy_vec = Vec::loadu(dY_ptr + d);
      Vec x_vec = Vec::loadu(X_ptr + d);
      auto [x_vec0, x_vec1] = convert_to_float<T>(x_vec);
      auto [dy_vec0, dy_vec1] = convert_to_float<T>(dy_vec);
      fVec dx_vec0 = c1_0 * dy_vec0 + fVec(c2) * x_vec0 + fVec(c3);
      fVec dx_vec1 = c1_1 * dy_vec1 + fVec(c2) * x_vec1 + fVec(c3);
      convert_from_float<T>(dx_vec0, dx_vec1).store(dX_ptr + d);
    }
  }
  if (D - d > 0) {
    auto [c1_0, c1_1] = gamma_null ? std::tuple<fVec, fVec>(fVec(1), fVec(1))
                                      : load_util(gamma + d, D - d);
    c1_0 = c1_0 * fVec(opmath_t(*rstd));
    c1_1 = c1_1 * fVec(opmath_t(*rstd));
    for (const auto m : c10::irange(HxW)) {
      const T* X_ptr = X_data + m * C;
      const T* dY_ptr = dY_data + m * C;
      T* dX_ptr = dX_data + m * C;
      Vec dy_vec = Vec::loadu(dY_ptr + d, D - d);
      Vec x_vec = Vec::loadu(X_ptr + d, D - d);
      auto [x_vec0, x_vec1] = convert_to_float<T>(x_vec);
      auto [dy_vec0, dy_vec1] = convert_to_float<T>(dy_vec);
      fVec dx_vec0 = c1_0 * dy_vec0 + fVec(c2) * x_vec0 + fVec(c3);
      fVec dx_vec1 = c1_1 * dy_vec1 + fVec(c2) * x_vec1 + fVec(c3);
      convert_from_float<T>(dx_vec0, dx_vec1).store(dX_ptr + d, D - d);
    }
  }
}

template <typename T, typename PT, typename opmath_t>
inline typename std::enable_if<std::is_same<T, opmath_t>::value, void>::type
ApplyInputGradientsChannelsLastRowMov(
  const T* dY_data,
  const T* X_data,
  T* dX_data,
  const PT* rstd,
  const PT* gamma,
  opmath_t c2,
  opmath_t c3,
  int64_t HxW,
  int64_t C,
  int64_t D) {
  const bool gamma_null = (gamma == nullptr);
  int64_t d = 0;
  auto K = vec::Vectorized<T>::size();
  for (; d < D / K * K; d += K) {
    auto c1 = vec::Vectorized<T>(*rstd) *
      (gamma_null ? vec::Vectorized<T>(1) : vec::Vectorized<T>::loadu(gamma + d));
    auto dy_vec = vec::Vectorized<T>::loadu(dY_data + d);
    auto x_vec = vec::Vectorized<T>::loadu(X_data + d);
    auto dx_vec = c1 * dy_vec +
      vec::Vectorized<T>(c2) * x_vec + vec::Vectorized<T>(c3);
    dx_vec.store(dX_data + d);
  }
  if (D - d > 0) {
    auto c1 = vec::Vectorized<T>(*rstd) *
      (gamma_null ? vec::Vectorized<T>(1) : vec::Vectorized<T>::loadu(gamma + d, D - d));
    auto dy_vec = vec::Vectorized<T>::loadu(dY_data + d, D - d);
    auto x_vec = vec::Vectorized<T>::loadu(X_data + d, D - d);
    auto dx_vec = c1 * dy_vec +
      vec::Vectorized<T>(c2) * x_vec + vec::Vectorized<T>(c3);
    dx_vec.store(dX_data + d, D - d);
  }
}

template <typename T, typename PT, typename opmath_t>
inline typename std::enable_if<!std::is_same<T, opmath_t>::value, void>::type
ApplyInputGradientsChannelsLastRowMov(
    const T* dY_data,
    const T* X_data,
    T* dX_data,
    const PT* rstd,
    const PT* gamma,
    opmath_t c2,
    opmath_t c3,
    int64_t HxW,
    int64_t C,
    int64_t D) {
  using Vec = vec::Vectorized<T>;
  using fVec = vec::Vectorized<opmath_t>;
  const bool gamma_null = (gamma == nullptr);
  auto K = Vec::size();
  int64_t d = 0;
  for (; d < D / K * K; d += K) {
    auto [c1_0, c1_1] = gamma_null ? std::tuple<fVec, fVec>(fVec(1), fVec(1))
                                      : load_util(gamma + d, K);
    c1_0 = c1_0 * fVec(opmath_t(*rstd));
    c1_1 = c1_1 * fVec(opmath_t(*rstd));
    Vec dy_vec = Vec::loadu(dY_data + d);
    Vec x_vec = Vec::loadu(X_data + d);
    auto [x_vec0, x_vec1] = convert_to_float<T>(x_vec);
    auto [dy_vec0, dy_vec1] = convert_to_float<T>(dy_vec);
    fVec dx_vec0 = c1_0 * dy_vec0 + fVec(c2) * x_vec0 + fVec(c3);
    fVec dx_vec1 = c1_1 * dy_vec1 + fVec(c2) * x_vec1 + fVec(c3);
    convert_from_float<T>(dx_vec0, dx_vec1).store(dX_data + d);
  }
  if (D - d > 0) {
    auto [c1_0, c1_1] = gamma_null ? std::tuple<fVec, fVec>(fVec(1), fVec(1))
                                      : load_util(gamma + d, D - d);
    c1_0 = c1_0 * fVec(opmath_t(*rstd));
    c1_1 = c1_1 * fVec(opmath_t(*rstd));
    Vec dy_vec = Vec::loadu(dY_data + d, D - d);
    Vec x_vec = Vec::loadu(X_data + d, D - d);
    auto [x_vec0, x_vec1] = convert_to_float<T>(x_vec);
    auto [dy_vec0, dy_vec1] = convert_to_float<T>(dy_vec);
    fVec dx_vec0 = c1_0 * dy_vec0 + fVec(c2) * x_vec0 + fVec(c3);
    fVec dx_vec1 = c1_1 * dy_vec1 + fVec(c2) * x_vec1 + fVec(c3);
    convert_from_float<T>(dx_vec0, dx_vec1).store(dX_data + d, D - d);
  }
}

template <typename T, typename PT, typename opmath_t>
inline typename std::
    enable_if<std::is_same<T, opmath_t>::value, std::tuple<opmath_t, opmath_t>>::type
    CalcInternalGradientsChannelsLast(
    const T* X_data,
    const T* dY_data,
    const PT* gamma_ptr,
    opmath_t* ds_ptr,
    opmath_t* db_ptr,
    int64_t HxW,
    int64_t C,
    int64_t D) {
  using Vec = vec::Vectorized<T>;
  const bool gamma_null = (gamma_ptr == nullptr);
  constexpr int64_t K = Vec::size();
  const int64_t inner_size = D / K * K;
  int64_t d = 0;
  opmath_t ds_gamma{0}, db_gamma{0};
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
      acc0_vec * (gamma_null ? Vec(1) : Vec::loadu(gamma_ptr + d)));
    db_gamma += vec::vec_reduce_all([](Vec& x, Vec& y) { return x + y; },
      acc1_vec * (gamma_null ? Vec(1) : Vec::loadu(gamma_ptr + d)));
  }
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
      acc0_vec * (gamma_null ? Vec(1) : Vec::loadu(gamma_ptr + d, D - d)));
    db_gamma += vec::vec_reduce_all([](Vec& x, Vec& y) { return x + y; },
      acc1_vec * (gamma_null ? Vec(1) : Vec::loadu(gamma_ptr + d, D - d)));
  }
  return std::tuple<opmath_t, opmath_t>(ds_gamma, db_gamma);
}

template <typename T, typename PT, typename opmath_t>
inline typename std::
    enable_if<!std::is_same<T, opmath_t>::value, std::tuple<opmath_t, opmath_t>>::type
    CalcInternalGradientsChannelsLast(
        const T* X_data,
        const T* dY_data,
        const PT* gamma_ptr,
        opmath_t* ds_ptr,
        opmath_t* db_ptr,
        int64_t HxW,
        int64_t C,
        int64_t D) {
  using Vec = vec::Vectorized<T>;
  using fVec = vec::Vectorized<opmath_t>;
  const bool gamma_null = (gamma_ptr == nullptr);
  constexpr int64_t K = Vec::size();
  const int64_t inner_size = D / K * K;
  float ds_gamma{0}, db_gamma{0};
  int64_t d = 0;
  for (; d < inner_size; d += K) {
    fVec acc0_vec0{0}, acc0_vec1{0}, acc1_vec0{0}, acc1_vec1{0};
    for (const auto m : c10::irange(HxW)) {
      const T* X_ptr = X_data + m * C;
      const T* dY_ptr = dY_data + m * C;
      Vec x_vec = Vec::loadu(X_ptr + d);
      Vec dy_vec = Vec::loadu(dY_ptr + d);
      auto [x_vec0, x_vec1] = convert_to_float<T>(x_vec);
      auto [dy_vec0, dy_vec1] = convert_to_float<T>(dy_vec);
      acc0_vec0 += x_vec0 * dy_vec0;
      acc0_vec1 += x_vec1 * dy_vec1;
      acc1_vec0 += dy_vec0;
      acc1_vec1 += dy_vec1;
    }
    acc0_vec0.store(ds_ptr + d);
    acc0_vec1.store(ds_ptr + d + fVec::size());
    acc1_vec0.store(db_ptr + d);
    acc1_vec1.store(db_ptr + d + fVec::size());
    auto [gamma_vec0, gamma_vec1] = gamma_null ?
      std::tuple<fVec, fVec>(fVec(1), fVec(1)) : load_util(gamma_ptr + d, K);
    ds_gamma += vec::vec_reduce_all(
        [](fVec& x, fVec& y) { return x + y; }, acc0_vec0 * gamma_vec0);
    ds_gamma += vec::vec_reduce_all(
        [](fVec& x, fVec& y) { return x + y; }, acc0_vec1 * gamma_vec1);
    db_gamma += vec::vec_reduce_all(
        [](fVec& x, fVec& y) { return x + y; }, acc1_vec0 * gamma_vec0);
    db_gamma += vec::vec_reduce_all(
        [](fVec& x, fVec& y) { return x + y; }, acc1_vec1 * gamma_vec1);
  }
  for (; d < D; d++) {
    opmath_t acc0{0}, acc1{0};
    for (const auto m : c10::irange(HxW)) {
      const T* X_ptr = X_data + m * C;
      const T* dY_ptr = dY_data + m * C;
      acc0 += opmath_t(X_ptr[d]) * opmath_t(dY_ptr[d]);
      acc1 += opmath_t(dY_ptr[d]);
    }
    ds_ptr[d] = acc0;
    db_ptr[d] = acc1;
    opmath_t gamma_val = gamma_null ? opmath_t(1) : opmath_t(gamma_ptr[d]);
    ds_gamma += acc0 * gamma_val;
    db_gamma += acc1 * gamma_val;
  }

  return std::tuple<opmath_t, opmath_t>(ds_gamma, db_gamma);
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
  const T* dY_data = dY.data_ptr<T>();
  const T* X_data = X.data_ptr<T>();
  const PT* mean_data = mean.data_ptr<PT>();
  const PT* rstd_data = rstd.data_ptr<PT>();
  const PT* gamma_data = gamma.defined() ? gamma.data_ptr<PT>() : nullptr;
  T* dX_data = dX.defined() ? dX.data_ptr<T>() : nullptr;
  PT* dgamma_data = dgamma.defined() ? dgamma.data_ptr<PT>() : nullptr;
  PT* dbeta_data = dbeta.defined() ? dbeta.data_ptr<PT>() : nullptr;
  const bool gamma_null = (gamma_data == nullptr);
  using opmath_t = at::opmath_type<T>;
  Tensor ds = at::empty({N, C}, X.options().dtype(c10::CppTypeToScalarType<opmath_t>::value));
  Tensor db = at::empty({N, C}, X.options().dtype(c10::CppTypeToScalarType<opmath_t>::value));
  opmath_t* ds_data = ds.data_ptr<opmath_t>();
  opmath_t* db_data = db.data_ptr<opmath_t>();
  const opmath_t s = opmath_t(1) / static_cast<opmath_t>(D * HxW);

  // Similar to channels last forward, channels last backward has also 2 impls.
  // impl-1: parallel on N * G. Only need one omp session for input gradients
  //   but memory access per thread is non-contiguous.
  //
  // impl-2: parallel on N * HxW. Memory access per thread is contiguous,
  //   but requires help of extra temp buffer of size {T, N, 2C}.

  // Generally impl-2 has better performance when HxW is large enough, so that
  //   data per thread {NHWC / T} is much larger then temp buffer per thread {2NC}
  constexpr int64_t feature_map_threshold = 2048;
  if (HxW < feature_map_threshold) {
    // impl-1: parallel on N * G.
    at::parallel_for(0, N * G, 1, [=](int64_t begin, int64_t end) {
      int64_t n{0}, g{0};
      data_index_init(begin, n, N, g, G);
      for (const auto i : c10::irange(begin, end)) {
        // Step 1. Compute internal gradients.
        opmath_t* ds_ptr = ds_data + i * D;
        opmath_t* db_ptr = db_data + i * D;
        const T* X_ptr = X_data + n * HxW * C + g * D;
        const T* dY_ptr = dY_data + n * HxW * C + g * D;
        const PT* gamma_ptr = gamma_null ? gamma_data : (gamma_data + g * D);
        auto [ds_gamma, db_gamma] = CalcInternalGradientsChannelsLast<T, PT, opmath_t>(
          X_ptr,
          dY_ptr,
          gamma_ptr,
          ds_ptr,
          db_ptr,
          HxW,
          C,
          D);

        // Step 2. Compute dX.
        T* dX_ptr = dX_data + n * HxW * C + g * D;
        const PT* rstd_ptr = rstd_data + i;
        const opmath_t c2 = (db_gamma * opmath_t(mean_data[i]) - ds_gamma) *
            opmath_t(rstd_data[i]) * opmath_t(rstd_data[i]) * opmath_t(rstd_data[i]) * s;
        const opmath_t c3 = -c2 * opmath_t(mean_data[i]) - db_gamma * opmath_t(rstd_data[i]) * s;
        ApplyInputGradientsChannelsLastColMov<T, PT, opmath_t>(dY_ptr, X_ptr, dX_ptr, rstd_ptr, gamma_ptr, c2, c3, HxW, C, D);
        data_index_step(n, N, g, G);
      }
    });

  } else {
    // impl-2: parallel on N * HxW.
    int num_threads = at::get_num_threads();
    Tensor buffer = at::empty({num_threads, N, 2 * C},
      X.options().dtype(c10::CppTypeToScalarType<opmath_t>::value)).zero_();
    opmath_t* buffer_data = buffer.data_ptr<opmath_t>();

    Tensor tmp_buffer = at::empty({N, 2 * G},
      X.options().dtype(c10::CppTypeToScalarType<opmath_t>::value));
    opmath_t* tmp_buffer_data = tmp_buffer.data_ptr<opmath_t>();

    // Step 1. Each thread compute their own internal gradients to the buffer.
    at::parallel_for(0, N * HxW, 1, [&](int64_t begin, int64_t end) {
      int tid = at::get_thread_num();
      opmath_t* buffer_ptr = buffer_data + tid * N * 2 * C;
      int64_t n{0}, m{0};
      data_index_init(begin, n, N, m, HxW);
      for (const auto i : c10::irange(begin, end)) {
        opmath_t* ds_ptr = buffer_ptr + n * 2 * C;
        opmath_t* db_ptr = ds_ptr + C;
        const T* X_ptr = X_data + i * C;
        const T* dY_ptr = dY_data + i * C;

        DsDbRowwiseMomentsChannelsLast<T, opmath_t>(dY_ptr, X_ptr, ds_ptr, db_ptr, C);
        data_index_step(n, N, m, HxW);
      }
    });

    // Step 2. Collect internal gradients from each thread and
    // get the final internal gradients to ds, db, and tmp_buffer.
    for (const auto n : c10::irange(N)) {
      for (const auto g : c10::irange(G)) {
        opmath_t ds_gamma{0}, db_gamma{0};
        for (const auto d : c10::irange(D)) {
          opmath_t ds_val{0}, db_val{0};
          for (const auto t : c10::irange(num_threads)) {
            opmath_t* buffer_ptr = buffer_data + t * N * 2 * C + n * 2 * C;
            opmath_t gamma_val = gamma_null ? opmath_t(1) : opmath_t(gamma_data[g * D + d]);
            ds_gamma += buffer_ptr[g * D + d] * gamma_val;
            db_gamma += buffer_ptr[g * D + d + C] * gamma_val;
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

    // Step 3. Compute dx.
    if (dX_data != nullptr) {
      at::parallel_for(0, N * HxW, 1, [&](int64_t begin, int64_t end) {
        int64_t n{0}, m{0};
        data_index_init(begin, n, N, m, HxW);
        for (const auto i : c10::irange(begin, end)) {
          for (const auto g : c10::irange(G)) {
            const T* X_ptr = X_data + i * C + g * D;
            const T* dY_ptr = dY_data + i * C + g * D;
            T* dX_ptr = dX_data + i * C + g * D;
            const PT* mean_ptr = mean_data + n * G + g;
            const PT* rstd_ptr = rstd_data + n * G + g;
            const PT* gamma_ptr = gamma_null ? gamma_data : (gamma_data + g * D);
            opmath_t ds_val = tmp_buffer_data[n * 2 * G + 2 * g];
            opmath_t db_val = tmp_buffer_data[n * 2 * G + 2 * g + 1];

            const opmath_t c2 = (db_val * opmath_t(*mean_ptr) - ds_val) *
                opmath_t(*rstd_ptr) * opmath_t(*rstd_ptr)* opmath_t(*rstd_ptr) * s;
            const opmath_t c3 = -c2 * opmath_t(*mean_ptr) - db_val * opmath_t(*rstd_ptr) * s;
            ApplyInputGradientsChannelsLastRowMov<T, PT, opmath_t>(dY_ptr, X_ptr, dX_ptr, rstd_ptr, gamma_ptr, c2, c3, HxW, C, D);
          }

          data_index_step(n, N, m, HxW);
        }
      });
    }

  }

  // Finally compute dgamma and dbeta.
  if (dgamma_data != nullptr) {
    GammaBackward(
        N, C, group, mean_data, rstd_data, ds_data, db_data, dgamma_data);
  }
  if (dbeta_data != nullptr) {
    BetaBackward(N, C, db_data, dbeta_data);
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
  // In training, using Amp to enable lower precision data type,
  // i.e., BFloat16 or Half, is recommended.
  // It will keep module parameters in opmath dtype i.e. float
  // while input/output will be in lower precision data type.
  // Using parameters in BFloat16 or Half may cause high precision loss.
  const bool mixed_type = is_mixed_type(dY, mean);
  switch (X.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES_AND2(
        ScalarType::BFloat16, ScalarType::Half, X.scalar_type(), "GroupNormBackwardKernelImpl", [&]() {
        using param_t = at::opmath_type<scalar_t>;
        if(mixed_type) {
          GroupNormBackwardKernelImplInternal<scalar_t, param_t>(
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
      AT_DISPATCH_FLOATING_TYPES_AND2(
        ScalarType::BFloat16, ScalarType::Half, X.scalar_type(), "GroupNormBackwardKernelImpl", [&]() {
        using param_t = at::opmath_type<scalar_t>;
        if(mixed_type) {
          GroupNormBackwardKernelImplChannelsLastInternal<scalar_t, param_t>(
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

} // namespace at::native
