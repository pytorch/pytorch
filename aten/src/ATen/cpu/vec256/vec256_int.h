#pragma once

#include "intrinsics.h"
#include "vec256_base.h"

namespace at {
namespace vec256 {

#ifdef __AVX2__

struct Vec256i {
  __m256i values;
  Vec256i() {}
  Vec256i(__m256i v) : values(v) {}
  operator __m256i() const {
    return values;
  }
  void load(const void *ptr) {
    values = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
  }
  void store(void *ptr) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
  }
};

template <>
struct Vec256<int64_t> : public Vec256i {
  using Vec256i::Vec256i;
  Vec256() {}
  Vec256(int64_t v) { values = _mm256_set1_epi64x(v); }
  static Vec256<int64_t> s_load(const void* ptr) {
    Vec256<int64_t> vec;
    vec.load(ptr);
    return vec;
  }
  static int size() { return 4; }
};

template <>
struct Vec256<int32_t> : public Vec256i {
  using Vec256i::Vec256i;
  Vec256() {}
  Vec256(int32_t v) { values = _mm256_set1_epi32(v); }
  static Vec256<int32_t> s_load(const void* ptr) {
    Vec256<int32_t> vec;
    vec.load(ptr);
    return vec;
  }
  static int size() { return 8; }
};

template <>
struct Vec256<int16_t> : public Vec256i {
  using Vec256i::Vec256i;
  Vec256() {}
  Vec256(int16_t v) { values = _mm256_set1_epi16(v); }
  static Vec256<int16_t> s_load(const void* ptr) {
    Vec256<int16_t> vec;
    vec.load(ptr);
    return vec;
  }
  static int size() { return 16; }
};

template <>
Vec256<int64_t> inline operator+(const Vec256<int64_t> &a,
                                 const Vec256<int64_t> &b) {
  return _mm256_add_epi64(a, b);
}

template <>
Vec256<int32_t> inline operator+(const Vec256<int32_t> &a,
                                 const Vec256<int32_t> &b) {
  return _mm256_add_epi32(a, b);
}

template <>
Vec256<int16_t> inline operator+(const Vec256<int16_t> &a,
                                 const Vec256<int16_t> &b) {
  return _mm256_add_epi16(a, b);
}

// AVX2 has no intrinsic for int64_t multiply so it needs to be emulated
// This could be implemented more efficiently using epi32 instructions
// This is also technically avx compatible, but then we'll need AVX
// code for add as well.
template <>
Vec256<int64_t> inline operator*(const Vec256<int64_t> &a,
                                 const Vec256<int64_t> &b) {
  int64_t a0 = _mm256_extract_epi64(a, 0);
  int64_t a1 = _mm256_extract_epi64(a, 1);
  int64_t a2 = _mm256_extract_epi64(a, 2);
  int64_t a3 = _mm256_extract_epi64(a, 3);

  int64_t b0 = _mm256_extract_epi64(b, 0);
  int64_t b1 = _mm256_extract_epi64(b, 1);
  int64_t b2 = _mm256_extract_epi64(b, 2);
  int64_t b3 = _mm256_extract_epi64(b, 3);

  int64_t c0 = a0 * b0;
  int64_t c1 = a1 * b1;
  int64_t c2 = a2 * b2;
  int64_t c3 = a3 * b3;

  return _mm256_set_epi64x(c3, c2, c1, c0);
}

template <>
Vec256<int32_t> inline operator*(const Vec256<int32_t> &a,
                                 const Vec256<int32_t> &b) {
  return _mm256_mullo_epi32(a, b);
}

template <>
Vec256<int16_t> inline operator*(const Vec256<int16_t> &a,
                                 const Vec256<int16_t> &b) {
  return _mm256_mullo_epi16(a, b);
}
#endif

}}
