#pragma once

#include <array>
#include <cstring>
#include <tuple>
#include <type_traits>
#include <utility>

#include <ATen/Parallel.h>
#include <ATen/OpMathType.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>

namespace at::native {
inline namespace CPU_CAPABILITY {

template<typename T> using opmath_t = at::opmath_type<T>;

constexpr int64_t kChunkSize = 16;

// Each Welford accumulator spans 256 bits (a register pair on 128-bit ISAs) so
// that every lane sees the same elements on every ISA and the moments match AVX2.
template <typename T>
constexpr int kMomentsVecRegs =
    vec::Vectorized<opmath_t<T>>::size() * sizeof(opmath_t<T>) < 32 ? 2 : 1;

template <typename T>
using MomentsVec = std::conditional_t<
    kMomentsVecRegs<T> == 1,
    vec::Vectorized<opmath_t<T>>,
    vec::VectorizedN<opmath_t<T>, kMomentsVecRegs<T>>>;

template <typename T>
constexpr int64_t kMomentsVecSize = vec::Vectorized<T>::size() * kMomentsVecRegs<T>;

template <typename T>
void AddMoments(
    int64_t m0_add,
    const T& m1_add,
    const T& m2_add,
    int64_t& m0,
    T& m1,
    T& m2) {
  const int64_t n = m0 + m0_add;
  const T c = n == 0 ? static_cast<T>(0) : static_cast<T>(m0_add) / static_cast<T>(n);
  const T delta = m1_add - m1;
  m1 += c * delta;
  m2 += m2_add + delta * delta * c * static_cast<T>(m0);
  m0 = n;
}

template <typename Vec>
C10_ALWAYS_INLINE void AddMomentsVec(
    int64_t m0_add,
    const Vec& m1_add,
    const Vec& m2_add,
    int64_t& m0,
    Vec& m1,
    Vec& m2) {
  using T = typename Vec::value_type;
  const int64_t n = m0 + m0_add;
  const T c = n == 0 ? static_cast<T>(0) : static_cast<T>(m0_add) / static_cast<T>(n);
  const Vec c_vec(c);
  const Vec delta = m1_add - m1;
  const Vec m2_tmp = m2 + m2_add;
  const Vec c_vec_delta = c_vec * delta;
  const Vec m0_delta = delta * Vec(static_cast<T>(m0));
  m1 = m1 + c_vec_delta;
  m2 = fmadd(m0_delta, c_vec_delta, m2_tmp);
  m0 = n;
}

template <typename T>
inline std::enable_if_t<std::is_same_v<T, opmath_t<T>>, void>
UpdateMomentsVec(
    int64_t m0,
    const T* X_ptr,
    const std::array<MomentsVec<T>, kChunkSize>& c_vecs,
    int64_t& m0_stk0,
    MomentsVec<T>& m1_stk0,
    MomentsVec<T>& m2_stk0) {
  using Vec = MomentsVec<T>;
  Vec m1_vec(0);
  Vec m2_vec(0);
  for (const auto j : c10::irange(m0)) {
    const Vec x_vec = Vec::loadu(X_ptr + j * Vec::size());
    const Vec tmpVec = c_vecs[j];
    const Vec delta_vec = x_vec - m1_vec;
    m1_vec = fmadd(tmpVec, delta_vec, m1_vec);
    const Vec tmpVec2 = x_vec - m1_vec;
    m2_vec = fmadd(delta_vec, tmpVec2, m2_vec);
  }
  AddMomentsVec(m0, m1_vec, m2_vec, m0_stk0, m1_stk0, m2_stk0);
}

// Loads kMomentsVecSize<T> reduced-precision scalars as two float vectors:
// the first holds scalars [0, kMomentsVecSize<T> / 2), the second the rest.
template <typename T>
inline std::tuple<MomentsVec<T>, MomentsVec<T>> LoadMomentsVec(const T* ptr) {
  using Vec = vec::Vectorized<T>;
  if constexpr (kMomentsVecRegs<T> == 1) {
    return convert_to_float<T>(Vec::loadu(ptr));
  } else {
    auto [x0, x1] = convert_to_float<T>(Vec::loadu(ptr));
    auto [x2, x3] = convert_to_float<T>(Vec::loadu(ptr + Vec::size()));
    return {MomentsVec<T>(x0, x1), MomentsVec<T>(x2, x3)};
  }
}

// each bfloat16/half vector will be converted to two float vectors,
// and accumulated successively on m1_stk0/m2_stk0.
template <typename T>
inline std::enable_if_t<!std::is_same_v<T, at::opmath_type<T>>, void>
UpdateMomentsVec(
    int64_t m0,
    const T* X_ptr,
    const std::array<MomentsVec<T>, kChunkSize>& c_vecs,
    int64_t& m0_stk0,
    MomentsVec<T>& m1_stk0,
    MomentsVec<T>& m2_stk0) {
  using fVec = MomentsVec<T>;
  fVec m1_fvec0(0), m1_fvec1(0);
  fVec m2_fvec0(0), m2_fvec1(0);
  for (const auto j : c10::irange(m0)) {
    const fVec tmpVec = c_vecs[j];
    auto [x_fvec0, x_fvec1] = LoadMomentsVec(X_ptr + j * kMomentsVecSize<T>);
    const fVec delta_fvec0 = x_fvec0 - m1_fvec0;
    const fVec delta_fvec1 = x_fvec1 - m1_fvec1;
    m1_fvec0 = fmadd(delta_fvec0, tmpVec, m1_fvec0);
    m1_fvec1 = fmadd(delta_fvec1, tmpVec, m1_fvec1);
    const fVec delta_fvec2 = x_fvec0 - m1_fvec0;
    const fVec delta_fvec3 = x_fvec1 - m1_fvec1;
    m2_fvec0 = fmadd(delta_fvec0, delta_fvec2, m2_fvec0);
    m2_fvec1 = fmadd(delta_fvec1, delta_fvec3, m2_fvec1);
  }
  AddMomentsVec(m0, m1_fvec0, m2_fvec0, m0_stk0, m1_stk0, m2_stk0);
  AddMomentsVec(m0, m1_fvec1, m2_fvec1, m0_stk0, m1_stk0, m2_stk0);
}

// Compute rowwise moments by Welford algorithm and cascade sum to improve
// numerical stability.
// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
// https://en.wikipedia.org/wiki/Pairwise_summation
template <typename T, int64_t kMaxDepth>
std::pair<opmath_t<T>, opmath_t<T>> RowwiseMomentsImpl(const T* X, int64_t N, int64_t ddof = 0) {
  using math_t = opmath_t<T>;

  using Vec = MomentsVec<T>;
  constexpr int64_t kVecSize = kMomentsVecSize<T>;
  constexpr int64_t kAccVecSize = Vec::size();
  const int64_t n = N / kVecSize;
  const int64_t m = divup(n, kChunkSize);
  const int64_t depth = utils::CeilLog2(m);

  const Vec kZeroVec(math_t(0));
  std::array<int64_t, kMaxDepth> m0_stk = {{0}};
  std::array<Vec, kMaxDepth> m1_stk;
  m1_stk.fill(kZeroVec);
  std::array<Vec, kMaxDepth> m2_stk;
  m2_stk.fill(kZeroVec);

  for (const auto i : c10::irange(m)) {
    const T* X_ptr = X + i * kChunkSize * kVecSize;
    const int64_t m0 = std::min(kChunkSize, n - i * kChunkSize);
    static std::array<Vec, kChunkSize> c_vecs = ([]() {
      std::array<Vec, kChunkSize> result;
      for (const auto i : c10::irange(kChunkSize)) {
        result[i] = Vec(math_t(1) / static_cast<math_t>(i + 1));
      }
      return result;
    })();
    UpdateMomentsVec(m0, X_ptr, c_vecs, m0_stk[0], m1_stk[0], m2_stk[0]);

    int64_t mask = i + 1;
    for (int64_t j = 1; j < depth && (mask & 1) == 0; ++j) {
      AddMomentsVec(
          m0_stk[j - 1],
          m1_stk[j - 1],
          m2_stk[j - 1],
          m0_stk[j],
          m1_stk[j],
          m2_stk[j]);
      m0_stk[j - 1] = 0;
      m1_stk[j - 1] = kZeroVec;
      m2_stk[j - 1] = kZeroVec;
      mask >>= 1;
    }
  }
  for (const auto i : c10::irange(1, depth)) {
    AddMomentsVec(
        m0_stk[i], m1_stk[i], m2_stk[i], m0_stk[0], m1_stk[0], m2_stk[0]);
  }

  std::array<math_t, kAccVecSize> m1_arr{};
  std::array<math_t, kAccVecSize> m2_arr{};
  m1_stk[0].store(m1_arr.data());
  m2_stk[0].store(m2_arr.data());

  int64_t m0 = 0;
  math_t m1 = 0;
  math_t m2 = 0;
  for (int64_t i = n * kVecSize; i < N; ++i) {
    math_t x = static_cast<math_t>(X[i]);
    const math_t delta = x - m1;
    ++m0;
    m1 += delta / static_cast<math_t>(m0);
    m2 += delta * (x - m1);
  }
  // for BFloat16, each vector in m1_arr/m2_arr holds 2*n accumulated result
  int64_t m0_add = n * kVecSize / kAccVecSize;
  for (const auto i : c10::irange(kAccVecSize)) {
    AddMoments(m0_add, m1_arr[i], m2_arr[i], m0, m1, m2);
  }

  return std::make_pair(m1, m2 / static_cast<math_t>(N - ddof));
}

template <typename T>
std::pair<opmath_t<T>, opmath_t<T>> RowwiseMoments(const T* X, int64_t N, int64_t ddof = 0) {
  constexpr int64_t kVecSize = kMomentsVecSize<T>;
  const int64_t n = N / kVecSize;
  const int64_t m = divup(n, kChunkSize);
  const int64_t depth = utils::CeilLog2(m);
  if (depth <= 4) {
    return RowwiseMomentsImpl<T, 4>(X, N, ddof);
  } else if (depth <= 8) {
    return RowwiseMomentsImpl<T, 8>(X, N, ddof);
  } else if (depth <= 16) {
    return RowwiseMomentsImpl<T, 16>(X, N, ddof);
  } else if (depth <= 32) {
    return RowwiseMomentsImpl<T, 32>(X, N, ddof);
  } else {
    return RowwiseMomentsImpl<T, 64>(X, N, ddof);
  }
}

} // namespace CPU_CAPABILITY
} // namespace at::native
