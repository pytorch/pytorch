#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include "Intrinsics.h"
#ifdef __AVX2__
#include "avx_mathfun.h"
#endif

#if defined(__GNUC__)
#define __at_align32__ __attribute__((aligned(32)))
#elif defined(_WIN32)
#define __at_align32__ __declspec(align(32))
#else
#define __at_align32__
#endif

// NOTE:
// If you specialize on a type, you must define all operations!
// C arrays and intrinsic types don't mix
//
// NOTE:
// When testing make sure to test all capabilities (AVX, AVX2, DEFAULT, etc.)

namespace at { namespace native { namespace vec256 {

template <class T>
class Vec256 {
 public:
  __at_align32__ T values[32 / sizeof(T)]; // Mimics AVX behavior
  inline void load(const T* ptr) {
    std::memcpy(values, ptr, 32);
  };
  inline void store(T* ptr) const {
    std::memcpy(ptr, values, 32);
  }
  inline void load(const T* ptr, size_t count) {
    size_t section = count * sizeof(T);
    std::memcpy(values, ptr, section);
  };
  inline void store(T* ptr, size_t count) const {
    size_t section = count * sizeof(T);
    std::memcpy(ptr, values, section);
  }
  size_t size = 32 / sizeof(T);
  inline void operator=(const Vec256<T>& b) {
    std::memcpy(values, b.values, 32);
  }
  inline Vec256<T> map(T (*f)(T)) {
    Vec256<T> ret;
    for (size_t i = 0; i < size; i++) {
      ret.values[i] = f(values[i]);
    }
    return ret;
  }
  inline Vec256<T> abs() {
    Vec256<T> ret;
    for (size_t i = 0; i < size; i++)
      ret.values[i] = values[i] < 0 ? -values[i] : values[i];
    return ret;
  }
  inline Vec256<T> exp() {
    return map(std::exp);
  }
  inline Vec256<T> log() {
    return map(std::log);
  }
  inline Vec256<T> ceil() {
    return map(std::ceil);
  }
  inline Vec256<T> cos() {
    return map(std::cos);
  }
  inline Vec256<T> floor() {
    return map(std::floor);
  }
  inline Vec256<T> round() {
    return map(std::round);
  }
  inline Vec256<T> sin() {
    return map(std::sin);
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
  inline Vec256<float> map(float (*f)(float)) {
    __at_align32__ float tmp[8];
    store(tmp);
    for (size_t i = 0; i < 8; i++)
      tmp[i] = f(tmp[i]);
    Vec256<float> ret;
    ret.load(tmp);
    return ret;
  }
  inline Vec256<float> abs() {
    Vec256<float> ret;
    __m256 mask = _mm256_set1_ps(-0.f);
    ret.values = _mm256_andnot_ps(mask, values);
    return ret;
  }

#ifdef __AVX2__
  inline Vec256<float> exp() {
    Vec256<float> ret;
    ret.values = exp256_ps(values);
    return ret;
  }
#else
  inline Vec256<float> exp() {
    return map(std::exp);
  }
#endif

#ifdef __AVX2__
  inline Vec256<float> log() {
    Vec256<float> ret;
    ret.values = log256_ps(values);
    return ret;
  }
#else
  inline Vec256<float> log() {
    return map(std::log);
  }
#endif

#ifdef __AVX2__
  inline Vec256<float> sin() {
    Vec256<float> ret;
    ret.values = sin256_ps(values);
    return ret;
  }
#else
  inline Vec256<float> sin() {
    return map(std::sin);
  }
#endif

#ifdef __AVX2__
  inline Vec256<float> cos() {
    Vec256<float> ret;
    ret.values = cos256_ps(values);
    return ret;
  }
#else
  inline Vec256<float> cos() {
    return map(std::cos);
  }
#endif

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
  inline Vec256<double> map(double (*f)(double)) {
    __at_align32__ double tmp[4];
    store(tmp);
    for (size_t i = 0; i < 4; i++)
      tmp[i] = f(tmp[i]);
    Vec256<double> ret;
    ret.load(tmp);
    return ret;
  }
  inline Vec256<double> abs() {
    Vec256<double> ret;
    __m256d mask = _mm256_set1_pd(-0.);
    ret.values = _mm256_andnot_pd(mask, values);
    return ret;
  }
  inline Vec256<double> exp() {
    return map(std::exp);
  }
  inline Vec256<double> log() {
    return map(std::log);
  }
  inline Vec256<double> cos() {
    return map(std::cos);
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
  inline Vec256<double> sin() {
    return map(std::sin);
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
  inline Vec256<int64_t> abs() {
    __m256i zero = _mm256_set1_epi64x(0);
    __m256i is_larger = _mm256_cmpgt_epi64(zero, values);
    __m256i inverse = _mm256_xor_si256(values, is_larger);
    Vec256<int64_t> ret;
    ret.values = _mm256_sub_epi64(inverse, is_larger);
    return ret;
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
  inline Vec256<int32_t> abs() {
    Vec256<int32_t> ret;
    ret.values = _mm256_abs_epi32(values);
    return ret;
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
  inline Vec256<int16_t> abs() {
    Vec256<int16_t> ret;
    ret.values = _mm256_abs_epi16(values);
    return ret;
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
