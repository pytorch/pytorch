#pragma once

#include <array>
#include <cstring>
#include <utility>
#include <iostream>

#include <ATen/Parallel.h>
#include <ATen/OpMathType.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/SmallVector.h>
#include <c10/util/irange.h>
#include <c10/util/llvmMathExtras.h>

namespace at::native {
inline namespace CPU_CAPABILITY {

template<typename T> using opmath_t = at::opmath_type<T>;

#ifdef __clang__
    #define UNROLL_LOOP_FULL       _Pragma("clang loop unroll(full)")
#elif defined(__GNUC__)
    #define UNROLL_LOOP_FULL       _Pragma("GCC unroll 128")
#elif defined(_MSC_VER)
    #define UNROLL_LOOP_FULL       __pragma(loop(hint_unroll(128)))
#elif defined(__INTEL_COMPILER) || defined(__ICC)
    #define UNROLL_LOOP_FULL       _Pragma("unroll")
#else
    #define UNROLL_LOOP_FULL
#endif

constexpr int64_t kChunkSize = 16;

template <typename T>
using InVec = vec::Vectorized<T>;

template <typename T>
using OpVec = vec::Vectorized<opmath_t<T>>;

template <typename T, int64_t kUnroll>
C10_ALWAYS_INLINE void welfordUpdateInner(
  const std::array<OpVec<T>, kUnroll> &x,
  const OpVec<T> &rcp,
  std::array<OpVec<T>, kUnroll> &m1,
  std::array<OpVec<T>, kUnroll> &m2) {
  std::array<OpVec<T>, kUnroll> delta;
  std::array<OpVec<T>, kUnroll> delta2;
  // If you write this function the normal way, with a single loop with many operations in it,
  // then you will find that LLVM on aarch64 is unwilling to rearrange your instructions.
  // It will just emit a sequence of 4 sequentially dependent instructions, followed by
  // another independent sequence of 4 sequentially dependent instructions, and so on.
  // This is suboptimal. So we have tried to provide the instructions in the order we would
  // actually like them.
  // This trouble accomplishes nothing on x64, because LLVM behaves more sensibly on x64.
  UNROLL_LOOP_FULL
  for (int i=0; i<kUnroll; i++) {
    delta[i] = x[i] - m1[i];
  }
  UNROLL_LOOP_FULL
  for (int i=0; i<kUnroll; i++) {
    m1[i] = vec::fmadd(delta[i], rcp, m1[i]);
  }
  UNROLL_LOOP_FULL
  for (int i=0; i<kUnroll; i++) {
    delta2[i] = x[i] - m1[i];
  }
  UNROLL_LOOP_FULL
  for (int i=0; i<kUnroll; i++) {
    m2[i] = vec::fmadd(delta[i], delta2[i], m2[i]);
  }
}

template <typename T, int64_t kUnroll>
C10_ALWAYS_INLINE std::enable_if_t<std::is_same_v<T, opmath_t<T>>, void>
welfordUpdate(
  const T* &X_ptr,
  const OpVec<T> &rcp,
  std::array<OpVec<T>, kUnroll> &m1,
  std::array<OpVec<T>, kUnroll> &m2) {
  std::array<OpVec<T>, kUnroll> x;
  UNROLL_LOOP_FULL
  for (int i=0; i<kUnroll; i++) {
    x[i] = InVec<T>::loadu(X_ptr);
    X_ptr += InVec<T>::size();
  }
  welfordUpdateInner<T, kUnroll>(x, rcp, m1, m2);
}

template <typename T, int64_t kUnroll>
C10_ALWAYS_INLINE std::enable_if_t<!std::is_same_v<T, at::opmath_type<T>>, void>
welfordUpdate(
  const T* &X_ptr,
  const OpVec<T> &rcp,
  std::array<OpVec<T>, kUnroll> &m1,
  std::array<OpVec<T>, kUnroll> &m2) {
  std::array<OpVec<T>, kUnroll> x;
  static_assert(kUnroll % 2 == 0);
  UNROLL_LOOP_FULL
  for (int i=0; i<kUnroll; i+=2) {
    InVec<T> x_bvec = InVec<T>::loadu(X_ptr);
    auto [x_fvec0, x_fvec1] = convert_to_float<T>(x_bvec);
    x[i] = x_fvec0;
    x[i+1] = x_fvec1;
    X_ptr += InVec<T>::size();
  }
  welfordUpdateInner<T, kUnroll>(x, rcp, m1, m2);
}

template<typename T, int64_t kUnroll>
C10_ALWAYS_INLINE void welfordMergeAccumulators(
  OpVec<T> &half_count,
  OpVec<T>* m1a,
  OpVec<T>* m2a,
  const std::array<OpVec<T>, kUnroll> &m1b,
  const std::array<OpVec<T>, kUnroll> &m2b) {
  std::array<OpVec<T>, kUnroll> delta;
  std::array<OpVec<T>, kUnroll> delta2;
  UNROLL_LOOP_FULL
  for (int i=0; i<kUnroll; i++) {
    delta[i] = m1b[i] - m1a[i];
  }
  UNROLL_LOOP_FULL
  for (int i=0; i<kUnroll; i++) {
    m2a[i] += m2b[i];
  }
  UNROLL_LOOP_FULL
  for (int i=0; i<kUnroll; i++) {
    delta2[i] = delta[i] * delta[i];
  }
  UNROLL_LOOP_FULL
  for (int i=0; i<kUnroll; i++) {
    m1a[i] = vec::fmadd(delta[i], OpVec<T>(0.5), m1a[i]);
  }
  UNROLL_LOOP_FULL
  for (int i=0; i<kUnroll; i++) {
    m2a[i] = vec::fmadd(delta2[i], half_count, m2a[i]);
  }
  half_count += half_count;
}

template <typename T, int64_t kUnroll, int64_t step>
C10_ALWAYS_INLINE void welfordMergeLayerVec(
  OpVec<T> &half_count,
  std::array<OpVec<T>, kUnroll> &m1,
  std::array<OpVec<T>, kUnroll> &m2) {
  constexpr int64_t final_elements = kUnroll / (1<<step);
  if constexpr (final_elements == 0) {
    return;
  }
  std::array<OpVec<T>, final_elements> delta;
  std::array<OpVec<T>, final_elements> delta2;
  UNROLL_LOOP_FULL
  for (int64_t i=0; i<final_elements; i++) {
    delta[i] = m1[i+final_elements] - m1[i];
  }
  UNROLL_LOOP_FULL
  for (int64_t i=0; i<final_elements; i++) {
    m2[i] += m2[i+final_elements];
  }
  UNROLL_LOOP_FULL
  for (int64_t i=0; i<final_elements; i++) {
    delta2[i] = delta[i] * delta[i];
  }
  UNROLL_LOOP_FULL
  for (int64_t i=0; i<final_elements; i++) {
    m1[i] = vec::fmadd(delta[i], OpVec<T>(0.5), m1[i]);
  }
  UNROLL_LOOP_FULL
  for (int64_t i=0; i<final_elements; i++) {
    m2[i] = vec::fmadd(delta2[i], half_count, m2[i]);
  }
  half_count += half_count;
}

template <typename T, int64_t kOpVecSize, int64_t step>
C10_ALWAYS_INLINE void welfordMergeLayer(
  opmath_t<T> &half_count,
  std::array<opmath_t<T>, kOpVecSize> &m1,
  std::array<opmath_t<T>, kOpVecSize> &m2) {
  constexpr int64_t final_elements = kOpVecSize / (1<<step);
  if constexpr (final_elements == 0) {
    return;
  }
  using math_t = opmath_t<T>;
  std::array<math_t, final_elements> delta;
  std::array<math_t, final_elements> delta2;
  UNROLL_LOOP_FULL
  for (int64_t i=0; i<final_elements; i++) {
    delta[i] = m1[i+final_elements] - m1[i];
  }
  UNROLL_LOOP_FULL
  for (int64_t i=0; i<final_elements; i++) {
    m2[i] += m2[i+final_elements];
  }
  UNROLL_LOOP_FULL
  for (int64_t i=0; i<final_elements; i++) {
    delta2[i] = delta[i] * delta[i];
  }
  UNROLL_LOOP_FULL
  for (int64_t i=0; i<final_elements; i++) {
    m1[i] = std::fma(delta[i], 0.5, m1[i]);
  }
  UNROLL_LOOP_FULL
  for (int64_t i=0; i<final_elements; i++) {
    m2[i] = std::fma(delta2[i], half_count, m2[i]);
  }
  half_count += half_count;
}

template <typename T>
C10_ALWAYS_INLINE void welfordMergeRemoveFakeZeros(
  const opmath_t<T> na,
  opmath_t<T> &m1,
  opmath_t<T> &m2,
  const opmath_t<T> nb,
  const opmath_t<T> n) {
  using math_t = opmath_t<T>;
  const math_t c = n == 0 ? static_cast<math_t>(0) : static_cast<math_t>(na) / static_cast<math_t>(n);
  const math_t delta2 = m1 * m1;
  const math_t delta2_nb = delta2 * nb;
  m1 *= c;
  m2 = std::fma(delta2_nb, c, m2);
}

// Compute rowwise moments by Welford algorithm and cascade sum to improve
// numerical stability.
// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
// https://en.wikipedia.org/wiki/Pairwise_summation
template <typename T>
std::pair<opmath_t<T>, opmath_t<T>> RowwiseMoments(const T* X, const int64_t N, int64_t ddof = 0) {
  using InVec = vec::Vectorized<T>;
  using OpVec = vec::Vectorized<opmath_t<T>>;
  using math_t = opmath_t<T>;

  // please run benchmarks, revisit unrolling choices VECTOR_WIDTH is bigger than 64.
  static_assert(VECTOR_WIDTH <= 64);

  // the inner loop of incremental welford updates benefits from some
  // unrolling because the update is most cheaply expressed as
  // sub, fma, sub, fma, where each op depends on the previous op's result
  // these instructions each have latency 3 or 4 and cpi 0.5, so let's unroll!
  //
  // aside: on some CPUs we are less likely to use, it's cheaper to mix in some
  // updates that are expressed as sub, fma, mul, fma, with fewer dependencies
  //
  // empirically determined unroll factor
  constexpr int64_t kInVecSize = InVec::size();
  constexpr int64_t kOpVecSize = OpVec::size();
  constexpr int64_t kUnroll = 128/VECTOR_WIDTH;
  constexpr int64_t kMaxDepth = 32;
  constexpr int64_t kLogChunkSize = llvm::countTrailingZeros(kChunkSize);
  constexpr int64_t kNAccumulators = kUnroll * kOpVecSize;
  constexpr int64_t kLogNAccumulators = llvm::countTrailingZeros(kNAccumulators);
  static_assert(1 << kLogChunkSize == kChunkSize);
  static_assert(1 << kLogNAccumulators == kNAccumulators);
  //constexpr int64_t kRoundUpAddend = (1 << (kLogChunkSize + kLogNAccumulators)) - 1;
  //constexpr int64_t kRoundUpMask = ~kRoundUpAddend;
  const int64_t n_whole_chunks = N / (kChunkSize * kNAccumulators);
  const int64_t leftover_n = N % (kChunkSize * kNAccumulators);
  const int64_t very_leftover_n = N % (kNAccumulators);

  std::array<std::array<OpVec, kUnroll>, kMaxDepth> m1_stk;
  std::array<std::array<OpVec, kUnroll>, kMaxDepth> m2_stk;
  std::array<OpVec, kUnroll> m1;
  std::array<OpVec, kUnroll> m2;
  const T* start_ptr = X;

  static std::array<OpVec, kChunkSize> c_vecs = ([]() {
    std::array<OpVec, kChunkSize> result;
    for (const auto i : c10::irange(kChunkSize)) {
      result[i] = OpVec(math_t(1) / static_cast<math_t>(i + 1));
    }
    return result;
  })();

  for (int64_t i = 0; i < n_whole_chunks; i++) {
    if (i > 0) {
      OpVec half_count{kChunkSize / 2};
      const int64_t dest_slot = llvm::countTrailingZeros(i);
      for (int64_t j=0; j < dest_slot; j++) {
        welfordMergeAccumulators<T, kUnroll>(half_count, &m1[0], &m2[0], m1_stk[j], m2_stk[j]);
      }
      m1_stk[dest_slot] = m1;
      m2_stk[dest_slot] = m2;
    }
    m1 = std::array<OpVec, kUnroll>{};
    m2 = std::array<OpVec, kUnroll>{};
    for (int64_t j=0; j < kChunkSize; j++) {
      welfordUpdate<T, kUnroll>(start_ptr, c_vecs[j], m1, m2);
    }
  }
  if (leftover_n > 0) {
    if (n_whole_chunks > 0) {
      const int64_t dest_slot = llvm::countTrailingZeros(n_whole_chunks);
      OpVec half_count{kChunkSize / 2};
      for (int64_t j=0; j < dest_slot; j++) {
        welfordMergeAccumulators<T, kUnroll>(half_count, &m1[0], &m2[0], m1_stk[j], m2_stk[j]);
      }
      m1_stk[dest_slot] = m1;
      m2_stk[dest_slot] = m2;
    }
    m1 = std::array<OpVec, kUnroll>{};
    m2 = std::array<OpVec, kUnroll>{};
    std::array<T, kInVecSize * kUnroll> fake_xs;
    const int64_t n_leftover_iters = divup(leftover_n, kNAccumulators);
    for (int64_t i = kChunkSize - n_leftover_iters; i < kChunkSize; i++) {
      if ((i == kChunkSize - 1) && (very_leftover_n > 0)) {
        std::memcpy(&fake_xs[0], start_ptr, very_leftover_n * sizeof(T));
        start_ptr = &fake_xs[0];
        std::memset(&fake_xs[very_leftover_n], 0, (kInVecSize * kUnroll - very_leftover_n) * sizeof(T));
      }
      welfordUpdate<T, kUnroll>(start_ptr, c_vecs[i], m1, m2);
    }
  }

  OpVec half_count{kChunkSize / 2};
  int64_t to_merge_mask = n_whole_chunks + (leftover_n > 0) - 1;
  if (to_merge_mask > 0) {
    int64_t depth = 64 - __builtin_clzll(to_merge_mask);
    for (int64_t i=0; i < depth; i++) {
      if (!(to_merge_mask & (1 << i))) {
        m1_stk[i] = std::array<OpVec, kUnroll>{};
        m2_stk[i] = std::array<OpVec, kUnroll>{};
      }
      welfordMergeAccumulators<T, kUnroll>(half_count, &m1[0], &m2[0], m1_stk[i], m2_stk[i]);
    }
  }

  welfordMergeLayerVec<T, kUnroll, 1>(half_count, m1, m2);
  welfordMergeLayerVec<T, kUnroll, 2>(half_count, m1, m2);
  welfordMergeLayerVec<T, kUnroll, 3>(half_count, m1, m2);
  welfordMergeLayerVec<T, kUnroll, 4>(half_count, m1, m2);
  // please add ^ vector reduction step(s) if kUnroll is larger than 16.
  static_assert(kUnroll <= 16);
  std::array<math_t, kOpVecSize> m1_arr;
  std::array<math_t, kOpVecSize> m2_arr;
  std::array<math_t, kOpVecSize> half_count_arr;
  half_count.store(&half_count_arr[0], OpVec::size());
  m1[0].store(&m1_arr[0], OpVec::size());
  m2[0].store(&m2_arr[0], OpVec::size());
  math_t half_count_scalar = half_count_arr[0];
  welfordMergeLayer<T, kOpVecSize, 1>(half_count_scalar, m1_arr, m2_arr);
  welfordMergeLayer<T, kOpVecSize, 2>(half_count_scalar, m1_arr, m2_arr);
  welfordMergeLayer<T, kOpVecSize, 3>(half_count_scalar, m1_arr, m2_arr);
  welfordMergeLayer<T, kOpVecSize, 4>(half_count_scalar, m1_arr, m2_arr);
  // please add ^ scalar reduction step(s) if vectors hold more than 16 math_t.
  static_assert(kOpVecSize <= 16);
  const math_t count = half_count_scalar + half_count_scalar;
  const int64_t int_count = static_cast<int64_t>(count);
  if (int_count != N) {
    // subtract out fake zeros
    const math_t other_count = static_cast<math_t>(N-int_count);
    welfordMergeRemoveFakeZeros<T>(count, m1_arr[0], m2_arr[0], other_count, static_cast<math_t>(N));
  }

  return std::make_pair(m1_arr[0], m2_arr[0] / static_cast<math_t>(N - ddof));
}

#undef UNROLL_LOOP_FULL

} // namespace CPU_CAPABILITY
} // namespace at::native
