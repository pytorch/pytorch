#pragma once

#if defined(_MSC_VER)
/* Microsoft C/C++-compatible compiler */
#include <intrin.h>
#if _MSC_VER <= 1900
#define _mm256_extract_epi64(X, Y) (((uint64_t*)&X)[Y])
#endif
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
/* GCC-compatible compiler, targeting x86/x86-64 */
#include <x86intrin.h>
#elif defined(__GNUC__) && defined(__ARM_NEON__)
/* GCC-compatible compiler, targeting ARM with NEON */
#include <arm_neon.h>
#elif defined(__GNUC__) && defined(__IWMMXT__)
/* GCC-compatible compiler, targeting ARM with WMMX */
#include <mmintrin.h>
#elif (defined(__GNUC__) || defined(__xlC__)) && \
    (defined(__VEC__) || defined(__ALTIVEC__))
/* XLC or GCC-compatible compiler, targeting PowerPC with VMX/VSX */
#include <altivec.h>
#elif defined(__GNUC__) && defined(__SPE__)
/* GCC-compatible compiler, targeting PowerPC with SPE */
#include <spe.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>

// NOTE:
// If you specialize on a type, you must define all operations!
// C arrays and intrinsic types don't mix
namespace at { namespace native { namespace vec256 {

template <class T>
class Vec256 {
 public:
  T values[32 / sizeof(T)]; // Mimics AVX behavior
  inline void load(const T* ptr) {
    std::memcpy(values, ptr, 32);
  };
  inline void store(T* ptr) const {
    std::memcpy(ptr, values, 32);
  }
  inline void load(const T* ptr, size_t count) {
    std::memcpy(values, ptr, 32 / sizeof(T) * count);
  };
  inline void store(T* ptr, size_t count) const {
    std::memcpy(ptr, values, 32 / sizeof(T) * count);
  }
  size_t size = 32 / sizeof(T);
  inline void operator=(const Vec256<T>& b) {
    std::memcpy(values, b.values, 32);
  }
  inline Vec256<T> map(T (*f)(T)) {
    Vec256<T> ret;
    for (size_t i = 0; i < size; i++)
      ret.values[i] = f(values[i]);
    return ret;
  }
  inline Vec256<T> ceil() {
    return map(std::ceil);
  }
  inline Vec256<T> floor() {
    return map(std::floor);
  }
  inline Vec256<T> round() {
    return map(std::round);
  }
  inline Vec256<T> trunc() {
    return map(std::trunc);
  }
  inline Vec256<T> sqrt() {
    return map(std::sqrt);
  }
};

template <class T>
Vec256<T> operator+(const Vec256<T>& a, const Vec256<T>& b) {
  Vec256<T> c = Vec256<T>();
  for (size_t i = 0; i < a.size; i++)
    c.values[i] = a.values[i] + b.values[i];
  return c;
}

template <class T>
Vec256<T> operator*(const Vec256<T>& a, const Vec256<T>& b) {
  Vec256<T> c = Vec256<T>();
  for (size_t i = 0; i < a.size; i++)
    c.values[i] = a.values[i] * b.values[i];
  return c;
}

#ifdef __AVX__
template <>
class Vec256<float> {
 public:
  __m256 values;
  Vec256<float>() {}
  inline void load(const float* ptr) {
    values = _mm256_loadu_ps(ptr);
  }
  inline void store(float* ptr) const {
    _mm256_storeu_ps(ptr, values);
  }
  inline void load(const float* ptr, size_t count) {
    float tmp_values[8];
    std::memcpy(tmp_values, ptr, count * sizeof(float));
    load(tmp_values);
  }
  inline void store(float* ptr, size_t count) const {
    float tmp_values[8];
    store(tmp_values);
    std::memcpy(ptr, tmp_values, count * sizeof(float));
  }
  size_t size = 8;
  inline void operator=(const Vec256<float>& b) {
    values = b.values;
  }
  inline Vec256<float> ceil() {
    Vec256<float> ret;
    ret.values = _mm256_ceil_ps(values);
    return ret;
  }
  inline Vec256<float> floor() {
    Vec256<float> ret;
    ret.values = _mm256_floor_ps(values);
    return ret;
  }
  inline Vec256<float> round() {
    Vec256<float> ret;
    ret.values = _mm256_round_ps(
        values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    return ret;
  }
  inline Vec256<float> trunc() {
    Vec256<float> ret;
    ret.values =
        _mm256_round_ps(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
    return ret;
  }
  inline Vec256<float> sqrt() {
    Vec256<float> ret;
    ret.values = _mm256_sqrt_ps(values);
    return ret;
  }
};

template <>
class Vec256<double> {
 public:
  __m256d values;
  Vec256<double>() {}
  inline void load(const double* ptr) {
    values = _mm256_loadu_pd(ptr);
  }
  inline void store(double* ptr) const {
    _mm256_storeu_pd(ptr, values);
  }
  inline void load(const double* ptr, size_t count) {
    double tmp_values[4];
    std::memcpy(tmp_values, ptr, count * sizeof(double));
    load(tmp_values);
  }
  inline void store(double* ptr, size_t count) const {
    double tmp_values[4];
    store(tmp_values);
    std::memcpy(ptr, tmp_values, count * sizeof(double));
  }
  size_t size = 4;
  inline void operator=(const Vec256<double>& b) {
    values = b.values;
  }
  inline Vec256<double> ceil() {
    Vec256<double> ret;
    ret.values = _mm256_ceil_pd(values);
    return ret;
  }
  inline Vec256<double> floor() {
    Vec256<double> ret;
    ret.values = _mm256_floor_pd(values);
    return ret;
  }
  inline Vec256<double> round() {
    Vec256<double> ret;
    ret.values = _mm256_round_pd(
        values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    return ret;
  }
  inline Vec256<double> trunc() {
    Vec256<double> ret;
    ret.values =
        _mm256_round_pd(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
    return ret;
  }
  inline Vec256<double> sqrt() {
    Vec256<double> ret;
    ret.values = _mm256_sqrt_pd(values);
    return ret;
  }
};

template <>
Vec256<float> inline operator+(const Vec256<float>& a, const Vec256<float>& b) {
  Vec256<float> c = Vec256<float>();
  c.values = _mm256_add_ps(a.values, b.values);
  return c;
}

template <>
Vec256<float> inline operator*(const Vec256<float>& a, const Vec256<float>& b) {
  Vec256<float> c = Vec256<float>();
  c.values = _mm256_mul_ps(a.values, b.values);
  return c;
}

template <>
Vec256<double> inline operator+(
    const Vec256<double>& a,
    const Vec256<double>& b) {
  Vec256<double> c = Vec256<double>();
  c.values = _mm256_add_pd(a.values, b.values);
  return c;
}

template <>
Vec256<double> inline operator*(
    const Vec256<double>& a,
    const Vec256<double>& b) {
  Vec256<double> c = Vec256<double>();
  c.values = _mm256_mul_pd(a.values, b.values);
  return c;
}
#endif

#ifdef __AVX2__
template <>
class Vec256<int64_t> {
 public:
  __m256i values;
  Vec256<int64_t>() {}
  inline void load(const int64_t* ptr) {
    values = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
  }
  inline void store(int64_t* ptr) const {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
  }
  inline void load(const int64_t* ptr, size_t count) {
    int64_t tmp_values[4];
    std::memcpy(tmp_values, ptr, count * sizeof(int64_t));
    load(tmp_values);
  }
  inline void store(int64_t* ptr, size_t count) const {
    int64_t tmp_values[4];
    store(tmp_values);
    std::memcpy(ptr, tmp_values, count * sizeof(int64_t));
  }
  size_t size = 4;
  inline void operator=(const Vec256<int64_t>& b) {
    values = b.values;
  }
};

template <>
class Vec256<int32_t> {
 public:
  __m256i values;
  Vec256<int32_t>() {}
  inline void load(const int32_t* ptr) {
    values = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
  }
  inline void store(int32_t* ptr) const {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
  }
  inline void load(const int32_t* ptr, size_t count) {
    int32_t tmp_values[8];
    std::memcpy(tmp_values, ptr, count * sizeof(int32_t));
    load(tmp_values);
  }
  inline void store(int32_t* ptr, size_t count) const {
    int32_t tmp_values[8];
    store(tmp_values);
    std::memcpy(ptr, tmp_values, count * sizeof(int32_t));
  }
  size_t size = 8;
  inline void operator=(const Vec256<int32_t>& b) {
    values = b.values;
  }
};

template <>
class Vec256<int16_t> {
 public:
  __m256i values;
  Vec256<int16_t>() {}
  inline void load(const int16_t* ptr) {
    values = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
  }
  inline void store(int16_t* ptr) const {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
  }
  inline void load(const int16_t* ptr, size_t count) {
    int16_t tmp_values[16];
    std::memcpy(tmp_values, ptr, count * sizeof(int16_t));
    load(tmp_values);
  }
  inline void store(int16_t* ptr, size_t count) const {
    int16_t tmp_values[16];
    store(tmp_values);
    std::memcpy(ptr, tmp_values, count * sizeof(int16_t));
  }
  size_t size = 16;
  inline void operator=(const Vec256<int16_t>& b) {
    values = b.values;
  }
};

template <>
Vec256<int64_t> inline operator+(
    const Vec256<int64_t>& a,
    const Vec256<int64_t>& b) {
  Vec256<int64_t> c = Vec256<int64_t>();
  c.values = _mm256_add_epi64(a.values, b.values);
  return c;
}

template <>
Vec256<int32_t> inline operator+(
    const Vec256<int32_t>& a,
    const Vec256<int32_t>& b) {
  Vec256<int32_t> c = Vec256<int32_t>();
  c.values = _mm256_add_epi32(a.values, b.values);
  return c;
}

template <>
Vec256<int16_t> inline operator+(
    const Vec256<int16_t>& a,
    const Vec256<int16_t>& b) {
  Vec256<int16_t> c = Vec256<int16_t>();
  c.values = _mm256_add_epi16(a.values, b.values);
  return c;
}

// AVX2 has no intrinsic for int64_t multiply so it needs to be emulated
// This could be implemented more efficiently using epi32 instructions
// This is also technically avx compatible, but then we'll need AVX
// code for add as well.
template <>
Vec256<int64_t> inline operator*(
    const Vec256<int64_t>& a,
    const Vec256<int64_t>& b) {
  Vec256<int64_t> c = Vec256<int64_t>();

  int64_t a0 = _mm256_extract_epi64(a.values, 0);
  int64_t a1 = _mm256_extract_epi64(a.values, 1);
  int64_t a2 = _mm256_extract_epi64(a.values, 2);
  int64_t a3 = _mm256_extract_epi64(a.values, 3);

  int64_t b0 = _mm256_extract_epi64(b.values, 0);
  int64_t b1 = _mm256_extract_epi64(b.values, 1);
  int64_t b2 = _mm256_extract_epi64(b.values, 2);
  int64_t b3 = _mm256_extract_epi64(b.values, 3);

  int64_t c0 = a0 * b0;
  int64_t c1 = a1 * b1;
  int64_t c2 = a2 * b2;
  int64_t c3 = a3 * b3;

  c.values = _mm256_set_epi64x(c3, c2, c1, c0);
  return c;
}

template <>
Vec256<int32_t> inline operator*(
    const Vec256<int32_t>& a,
    const Vec256<int32_t>& b) {
  Vec256<int32_t> c = Vec256<int32_t>();
  c.values = _mm256_mullo_epi32(a.values, b.values);
  return c;
}

template <>
Vec256<int16_t> inline operator*(
    const Vec256<int16_t>& a,
    const Vec256<int16_t>& b) {
  Vec256<int16_t> c = Vec256<int16_t>();
  c.values = _mm256_mullo_epi16(a.values, b.values);
  return c;
}
#endif
}}} // namespace at::native::vec256
