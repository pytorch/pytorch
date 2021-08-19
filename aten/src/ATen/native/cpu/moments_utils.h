#pragma once

#include <array>
#include <cstring>
#include <numeric>
#include <utility>
#include <vector>

#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/SmallVector.h>

namespace at {
namespace native {
namespace utils {

constexpr int64_t kChunkSize = 16;

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

template <typename T>
void AddMomentsVec(
    int64_t m0_add,
    const vec::Vectorized<T>& m1_add,
    const vec::Vectorized<T>& m2_add,
    int64_t& m0,
    vec::Vectorized<T>& m1,
    vec::Vectorized<T>& m2) {
  using Vec = vec::Vectorized<T>;
  const int64_t n = m0 + m0_add;
  const T c = n == 0 ? static_cast<T>(0) : static_cast<T>(m0_add) / static_cast<T>(n);
  const Vec c_vec(c);
  const Vec delta = m1_add - m1;
  m1 += c_vec * delta;
  m2 += m2_add + delta * delta * c_vec * Vec(static_cast<T>(m0));
  m0 = n;
}

// Compute rowwise moments by Welford algorithm and cascade sum to improve
// numerical stability.
// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
// https://en.wikipedia.org/wiki/Pairwise_summation
template <typename T, int64_t kMaxDepth>
std::pair<T, T> RowwiseMomentsImpl(const T* X, int64_t N, int64_t ddof = 0) {
  using Vec = vec::Vectorized<T>;

  constexpr int64_t kVecSize = Vec::size();
  const int64_t n = N / kVecSize;
  const int64_t m = divup(n, kChunkSize);
  const int64_t depth = CeilLog2(m);

  const Vec kZeroVec(T(0));
  c10::SmallVector<int64_t, kMaxDepth> m0_stk(depth, 0);
  c10::SmallVector<Vec, kMaxDepth> m1_stk(depth, kZeroVec);
  c10::SmallVector<Vec, kMaxDepth> m2_stk(depth, kZeroVec);

  for (int64_t i = 0; i < m; ++i) {
    const T* X_ptr = X + i * kChunkSize * kVecSize;
    const int64_t m0 = std::min(kChunkSize, n - i * kChunkSize);
    Vec m1_vec(0);
    Vec m2_vec(0);
    for (int64_t j = 0; j < m0; ++j) {
      const Vec x_vec = Vec::loadu(X_ptr + j * kVecSize);
      const Vec delta_vec = x_vec - m1_vec;
      const Vec c_vec = Vec(T(1) / static_cast<T>(j + 1));
      m1_vec += delta_vec * c_vec;
      m2_vec += delta_vec * (x_vec - m1_vec);
    }
    AddMomentsVec(m0, m1_vec, m2_vec, m0_stk[0], m1_stk[0], m2_stk[0]);
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
  for (int64_t i = 1; i < depth; ++i) {
    AddMomentsVec(
        m0_stk[i], m1_stk[i], m2_stk[i], m0_stk[0], m1_stk[0], m2_stk[0]);
  }

  std::array<T, kVecSize> m1_arr{};
  std::array<T, kVecSize> m2_arr{};
  m1_stk[0].store(m1_arr.data());
  m2_stk[0].store(m2_arr.data());

  int64_t m0 = 0;
  T m1 = 0;
  T m2 = 0;
  for (int64_t i = n * kVecSize; i < N; ++i) {
    const T delta = X[i] - m1;
    ++m0;
    m1 += delta / static_cast<T>(m0);
    m2 += delta * (X[i] - m1);
  }
  for (int64_t i = 0; i < kVecSize; ++i) {
    AddMoments(n, m1_arr[i], m2_arr[i], m0, m1, m2);
  }

  return std::make_pair(m1, m2 / static_cast<T>(N - ddof));
}

template <typename T>
std::pair<T, T> RowwiseMoments(const T* X, int64_t N, int64_t ddof = 0) {
  using Vec = vec::Vectorized<T>;
  constexpr int64_t kVecSize = Vec::size();
  const int64_t n = N / kVecSize;
  const int64_t m = divup(n, kChunkSize);
  const int64_t depth = CeilLog2(m);
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

} // namespace utils
} // namespace native
} // namespace at
