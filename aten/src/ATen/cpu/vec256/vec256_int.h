#pragma once

#include "intrinsics.h"
#include "vec256_base.h"

namespace at {
namespace vec256 {
namespace {

#ifdef __AVX2__

struct Vec256i {
  __m256i values;
  Vec256i() {}
  Vec256i(__m256i v) : values(v) {}
  operator __m256i() const {
    return values;
  }
};

template <>
struct Vec256<int64_t> : public Vec256i {
  static constexpr int size = 4;
  using Vec256i::Vec256i;
  Vec256() {}
  Vec256(int64_t v) { values = _mm256_set1_epi64x(v); }
  template <int64_t mask>
  static Vec256<int64_t> blend(Vec256<int64_t> a, Vec256<int64_t> b) {
    __at_align32__ int64_t tmp_values[size];
    a.store(tmp_values);
    if (mask & 0x01)
      tmp_values[0] = _mm256_extract_epi16(b.values, 0);
    if (mask & 0x02)
      tmp_values[1] = _mm256_extract_epi16(b.values, 1);
    if (mask & 0x04)
      tmp_values[2] = _mm256_extract_epi16(b.values, 2);
    if (mask & 0x08)
      tmp_values[3] = _mm256_extract_epi16(b.values, 3);
    return loadu(tmp_values);
  }
  static Vec256<int64_t>
  set(Vec256<int64_t> a, Vec256<int64_t> b, int64_t count = size) {
    switch (count) {
      case 0:
        return a;
      case 1:
        return blend<1>(a, b);
      case 2:
        return blend<3>(a, b);
      case 3:
        return blend<7>(a, b);
    }
    return b;
  }
  static Vec256<int64_t> loadu(const void* ptr) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
  }
  static Vec256<int64_t> loadu(const void* ptr, int64_t count) {
    __at_align32__ int64_t tmp_values[size];
    std::memcpy(tmp_values, ptr, count * sizeof(int64_t));
    return loadu(tmp_values);
  }
  void store(void* ptr, int count = size) const {
    if (count == size) {
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
    } else {
      __at_align32__ int64_t tmp_values[size];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(int64_t));
    }
  }
  Vec256<int64_t> abs() const {
    auto zero = _mm256_set1_epi64x(0);
    auto is_larger = _mm256_cmpgt_epi64(zero, values);
    auto inverse = _mm256_xor_si256(values, is_larger);
    return _mm256_sub_epi64(inverse, is_larger);
  }
};

template <>
struct Vec256<int32_t> : public Vec256i {
  static constexpr int size = 8;
  using Vec256i::Vec256i;
  Vec256() {}
  Vec256(int32_t v) { values = _mm256_set1_epi32(v); }
  template <int64_t mask>
  static Vec256<int32_t> blend(Vec256<int32_t> a, Vec256<int32_t> b) {
    return _mm256_blend_epi32(a, b, mask);
  }
  static Vec256<int32_t>
  set(Vec256<int32_t> a, Vec256<int32_t> b, int32_t count = size) {
    switch (count) {
      case 0:
        return a;
      case 1:
        return blend<1>(a, b);
      case 2:
        return blend<3>(a, b);
      case 3:
        return blend<7>(a, b);
      case 4:
        return blend<15>(a, b);
      case 5:
        return blend<31>(a, b);
      case 6:
        return blend<63>(a, b);
      case 7:
        return blend<127>(a, b);
    }
    return b;
  }
  static Vec256<int32_t> loadu(const void* ptr) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
  }
  static Vec256<int32_t> loadu(const void* ptr, int32_t count) {
    __at_align32__ int32_t tmp_values[size];
    std::memcpy(tmp_values, ptr, count * sizeof(int32_t));
    return loadu(tmp_values);
  }
  void store(void* ptr, int count = size) const {
    if (count == size) {
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
    } else {
      __at_align32__ int32_t tmp_values[size];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(int32_t));
    }
  }
  Vec256<int32_t> abs() const {
    return _mm256_abs_epi32(values);
  }
};

template <>
struct Vec256<int16_t> : public Vec256i {
  static constexpr int size = 16;
  using Vec256i::Vec256i;
  Vec256() {}
  Vec256(int16_t v) { values = _mm256_set1_epi16(v); }
  template <int64_t mask>
  static Vec256<int16_t> blend(Vec256<int16_t> a, Vec256<int16_t> b) {
    __at_align32__ int16_t tmp_values[size];
    a.store(tmp_values);
    if (mask & 0x01)
      tmp_values[0] = _mm256_extract_epi16(b.values, 0);
    if (mask & 0x02)
      tmp_values[1] = _mm256_extract_epi16(b.values, 1);
    if (mask & 0x04)
      tmp_values[2] = _mm256_extract_epi16(b.values, 2);
    if (mask & 0x08)
      tmp_values[3] = _mm256_extract_epi16(b.values, 3);
    if (mask & 0x10)
      tmp_values[4] = _mm256_extract_epi16(b.values, 4);
    if (mask & 0x20)
      tmp_values[5] = _mm256_extract_epi16(b.values, 5);
    if (mask & 0x40)
      tmp_values[6] = _mm256_extract_epi16(b.values, 6);
    if (mask & 0x80)
      tmp_values[7] = _mm256_extract_epi16(b.values, 7);
    if (mask & 0x100)
      tmp_values[8] = _mm256_extract_epi16(b.values, 8);
    if (mask & 0x200)
      tmp_values[9] = _mm256_extract_epi16(b.values, 9);
    if (mask & 0x400)
      tmp_values[10] = _mm256_extract_epi16(b.values, 10);
    if (mask & 0x800)
      tmp_values[11] = _mm256_extract_epi16(b.values, 11);
    if (mask & 0x1000)
      tmp_values[12] = _mm256_extract_epi16(b.values, 12);
    if (mask & 0x2000)
      tmp_values[13] = _mm256_extract_epi16(b.values, 13);
    if (mask & 0x4000)
      tmp_values[14] = _mm256_extract_epi16(b.values, 14);
    if (mask & 0x8000)
      tmp_values[15] = _mm256_extract_epi16(b.values, 15);
    return loadu(tmp_values);
  }
  static Vec256<int16_t>
  set(Vec256<int16_t> a, Vec256<int16_t> b, int16_t count = size) {
    switch (count) {
      case 0:
        return a;
      case 1:
        return blend<1>(a, b);
      case 2:
        return blend<3>(a, b);
      case 3:
        return blend<7>(a, b);
      case 4:
        return blend<15>(a, b);
      case 5:
        return blend<31>(a, b);
      case 6:
        return blend<63>(a, b);
      case 7:
        return blend<127>(a, b);
      case 8:
        return blend<255>(a, b);
      case 9:
        return blend<511>(a, b);
      case 10:
        return blend<1023>(a, b);
      case 11:
        return blend<2047>(a, b);
      case 12:
        return blend<4095>(a, b);
      case 13:
        return blend<8191>(a, b);
      case 14:
        return blend<16383>(a, b);
      case 15:
        return blend<32767>(a, b);
    }
    return b;
  }
  static Vec256<int16_t> loadu(const void* ptr) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
  }
  static Vec256<int16_t> loadu(const void* ptr, int16_t count) {
    __at_align32__ int16_t tmp_values[size];
    std::memcpy(tmp_values, ptr, count * sizeof(int16_t));
    return loadu(tmp_values);
  }
  void store(void* ptr, int count = size) const {
    if (count == size) {
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
    } else {
      __at_align32__ int16_t tmp_values[size];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(int16_t));
    }
  }
  Vec256<int16_t> abs() const {
    return _mm256_abs_epi16(values);
  }
};

template <>
Vec256<int64_t> inline operator+(const Vec256<int64_t>& a, const Vec256<int64_t>& b) {
  return _mm256_add_epi64(a, b);
}

template <>
Vec256<int32_t> inline operator+(const Vec256<int32_t>& a, const Vec256<int32_t>& b) {
  return _mm256_add_epi32(a, b);
}

template <>
Vec256<int16_t> inline operator+(const Vec256<int16_t>& a, const Vec256<int16_t>& b) {
  return _mm256_add_epi16(a, b);
}

// AVX2 has no intrinsic for int64_t multiply so it needs to be emulated
// This could be implemented more efficiently using epi32 instructions
// This is also technically avx compatible, but then we'll need AVX
// code for add as well.
template <>
Vec256<int64_t> inline operator*(const Vec256<int64_t>& a, const Vec256<int64_t>& b) {
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
Vec256<int32_t> inline operator*(const Vec256<int32_t>& a, const Vec256<int32_t>& b) {
  return _mm256_mullo_epi32(a, b);
}

template <>
Vec256<int16_t> inline operator*(const Vec256<int16_t>& a, const Vec256<int16_t>& b) {
  return _mm256_mullo_epi16(a, b);
}
#endif

}}}
