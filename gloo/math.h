/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <algorithm>
#include <cassert>

#ifdef __AVX2__
#include <immintrin.h>
#endif
#ifdef GLOO_USE_EIGEN
#include <Eigen/Core>
#endif

#include "gloo/types.h"

namespace gloo {

#ifdef GLOO_USE_EIGEN

template <typename T>
using EigenVectorArrayMap =
  Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> >;
template <typename T>
using ConstEigenVectorArrayMap =
  Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1> >;

template <typename T>
void sum(T* x, const T* y, size_t n) {
  EigenVectorArrayMap<T>(x, n) =
    ConstEigenVectorArrayMap<T>(x, n) + ConstEigenVectorArrayMap<T>(y, n);
};

template <typename T>
void product(T* x, const T* y, size_t n) {
  EigenVectorArrayMap<T>(x, n) =
    ConstEigenVectorArrayMap<T>(x, n) * ConstEigenVectorArrayMap<T>(y, n);
};

template <typename T>
void min(T* x, const T* y, size_t n) {
  EigenVectorArrayMap<T>(x, n) =
    ConstEigenVectorArrayMap<T>(x, n).min(ConstEigenVectorArrayMap<T>(y, n));
};

template <typename T>
void max(T* x, const T* y, size_t n) {
  EigenVectorArrayMap<T>(x, n) =
    ConstEigenVectorArrayMap<T>(x, n).max(ConstEigenVectorArrayMap<T>(y, n));
};

#else

template <typename T>
void sum(T* x, const T* y, size_t n) {
  for (auto i = 0; i < n; i++) {
    x[i] = x[i] + y[i];
  }
}

template <typename T>
void product(T* x, const T* y, size_t n) {
  for (auto i = 0; i < n; i++) {
    x[i] = x[i] * y[i];
  }
}

template <typename T>
void max(T* x, const T* y, size_t n) {
  for (auto i = 0; i < n; i++) {
    x[i] = std::max(x[i], y[i]);
  }
}

template <typename T>
void min(T* x, const T* y, size_t n) {
  for (auto i = 0; i < n; i++) {
    x[i] = std::min(x[i], y[i]);
  }
}

#endif

#ifdef __AVX2__
#define is_aligned(POINTER, BYTE_COUNT) \
  (((uintptr_t)(const void *)(POINTER)) % (BYTE_COUNT) == 0)

// Assumes x and y are either both aligned to 32 bytes or unaligned by the same
// offset, as would happen when reducing at an offset within an aligned buffer
template <>
inline void sum<float16>(float16* x, const float16* y, size_t n) {
  // Handle unaligned data at the beginning of the buffer
  while (!is_aligned(x, 32)) {
    *x += *y;
    x++;
    y++;
    n--;
  }
  assert(is_aligned(y, 32));
  size_t i;
  for (i = 0; i < (n / 8) * 8; i += 8) {
    __m256 va32 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(&x[i])));
    __m256 vb32 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(&y[i])));
    __m128i vc16 = _mm256_cvtps_ph(_mm256_add_ps(va32, vb32), 0);
    _mm_store_si128((__m128i*)(&x[i]), vc16);
  }
  // Leftovers
  for (; i < n; i++) {
    x[i] += y[i];
  }
}

// Assumes x and y are either both aligned to 32 bytes or unaligned by the same
// offset, as would happen when reducing at an offset within an aligned buffer
template <>
inline void product<float16>(float16* x, const float16* y, size_t n) {
  // Handle unaligned data at the beginning of the buffer
  while (!is_aligned(x, 32)) {
    *x *= *y;
    x++;
    y++;
    n--;
  }
  assert(is_aligned(y, 32));
  size_t i;
  for (i = 0; i < (n / 8) * 8; i += 8) {
    __m256 va32 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(&x[i])));
    __m256 vb32 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(&y[i])));
    __m128i vc16 = _mm256_cvtps_ph(_mm256_mul_ps(va32, vb32), 0);
    _mm_store_si128((__m128i*)(&x[i]), vc16);
  }
  // Leftovers
  for (; i < n; i++) {
    x[i] *= y[i];
  }
}

// Assumes x and y are either both aligned to 32 bytes or unaligned by the same
// offset, as would happen when reducing at an offset within an aligned buffer
template <>
inline void max<float16>(float16* x, const float16* y, size_t n) {
  // Handle unaligned data at the beginning of the buffer
  while (!is_aligned(x, 32)) {
    *x = std::max(*x, *y);
    x++;
    y++;
    n--;
  }
  assert(is_aligned(y, 32));
  size_t i;
  for (i = 0; i < (n / 8) * 8; i += 8) {
    __m256 va32 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(&x[i])));
    __m256 vb32 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(&y[i])));
    __m128i vc16 = _mm256_cvtps_ph(_mm256_max_ps(va32, vb32), 0);
    _mm_store_si128((__m128i*)(&x[i]), vc16);
  }
  // Leftovers
  for (; i < n; i++) {
    x[i] = std::max(x[i], y[i]);
  }
}

// Assumes x and y are either both aligned to 32 bytes or unaligned by the same
// offset, as would happen when reducing at an offset within an aligned buffer
template <>
inline void min<float16>(float16* x, const float16* y, size_t n) {
  // Handle unaligned data at the beginning of the buffer
  while (!is_aligned(x, 32)) {
    *x = std::min(*x, *y);
    x++;
    y++;
    n--;
  }
  assert(is_aligned(y, 32));
  size_t i;
  for (i = 0; i < (n / 8) * 8; i += 8) {
    __m256 va32 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(&x[i])));
    __m256 vb32 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(&y[i])));
    __m128i vc16 = _mm256_cvtps_ph(_mm256_min_ps(va32, vb32), 0);
    _mm_store_si128((__m128i*)(&x[i]), vc16);
  }
  // Leftovers
  for (; i < n; i++) {
    x[i] = std::min(x[i], y[i]);
  }
}
#endif

} // namespace gloo
