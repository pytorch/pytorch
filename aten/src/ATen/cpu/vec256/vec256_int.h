#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec256/intrinsics.h>
#include <ATen/cpu/vec256/vec256_base.h>
#include <c10/macros/Macros.h>

namespace at {
namespace vec256 {
namespace {

#ifdef CPU_CAPABILITY_AVX2

struct Vec256i {
protected:
  __m256i values;

  static inline __m256i invert(const __m256i& v) {
    const auto ones = _mm256_set1_epi64x(-1);
    return _mm256_xor_si256(ones, v);
  }
public:
  Vec256i() {}
  Vec256i(__m256i v) : values(v) {}
  operator __m256i() const {
    return values;
  }
};

#else

struct Vec256i {};  // dummy definition to make Vec256i always defined

#endif // CPU_CAPABILITY_AVX2

#ifdef CPU_CAPABILITY_AVX2

template <>
class Vec256<int64_t> : public Vec256i {
private:
  static const Vec256<int64_t> ones;
public:
  using value_type = int64_t;
  static constexpr int size() {
    return 4;
  }
  using Vec256i::Vec256i;
  Vec256() {}
  Vec256(int64_t v) { values = _mm256_set1_epi64x(v); }
  Vec256(int64_t val1, int64_t val2, int64_t val3, int64_t val4) {
    values = _mm256_setr_epi64x(val1, val2, val3, val4);
  }
  template <int64_t mask>
  static Vec256<int64_t> blend(Vec256<int64_t> a, Vec256<int64_t> b) {
    __at_align32__ int64_t tmp_values[size()];
    a.store(tmp_values);
    if (mask & 0x01)
      tmp_values[0] = _mm256_extract_epi64(b.values, 0);
    if (mask & 0x02)
      tmp_values[1] = _mm256_extract_epi64(b.values, 1);
    if (mask & 0x04)
      tmp_values[2] = _mm256_extract_epi64(b.values, 2);
    if (mask & 0x08)
      tmp_values[3] = _mm256_extract_epi64(b.values, 3);
    return loadu(tmp_values);
  }
  static Vec256<int64_t> blendv(const Vec256<int64_t>& a, const Vec256<int64_t>& b,
                                const Vec256<int64_t>& mask) {
    return _mm256_blendv_epi8(a.values, b.values, mask.values);
  }
  template <typename step_t>
  static Vec256<int64_t> arange(int64_t base = 0, step_t step = static_cast<step_t>(1)) {
    return Vec256<int64_t>(base, base + step, base + 2 * step, base + 3 * step);
  }
  static Vec256<int64_t>
  set(Vec256<int64_t> a, Vec256<int64_t> b, int64_t count = size()) {
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
    __at_align32__ int64_t tmp_values[size()];
    // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
    // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
    // instructions while a loop would be compiled to one instruction.
    for (auto i = 0; i < size(); ++i) {
      tmp_values[i] = 0;
    }
    std::memcpy(tmp_values, ptr, count * sizeof(int64_t));
    return loadu(tmp_values);
  }
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      // ptr need not to be aligned here. See
      // https://software.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-intel-advanced-vector-extensions/intrinsics-for-load-and-store-operations-1/mm256-storeu-si256.html
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
    } else if (count > 0) {
      __at_align32__ int64_t tmp_values[size()];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(int64_t));
    }
  }
  const int64_t& operator[](int idx) const  = delete;
  int64_t& operator[](int idx)  = delete;
  Vec256<int64_t> abs() const {
    auto zero = _mm256_set1_epi64x(0);
    auto is_larger = _mm256_cmpgt_epi64(zero, values);
    auto inverse = _mm256_xor_si256(values, is_larger);
    return _mm256_sub_epi64(inverse, is_larger);
  }
  Vec256<int64_t> real() const {
    return *this;
  }
  Vec256<int64_t> imag() const {
    return _mm256_set1_epi64x(0);
  }
  Vec256<int64_t> conj() const {
    return *this;
  }
  Vec256<int64_t> frac() const;
  Vec256<int64_t> neg() const;
  Vec256<int64_t> operator==(const Vec256<int64_t>& other) const {
    return _mm256_cmpeq_epi64(values, other.values);
  }
  Vec256<int64_t> operator!=(const Vec256<int64_t>& other) const {
    return invert(_mm256_cmpeq_epi64(values, other.values));
  }
  Vec256<int64_t> operator<(const Vec256<int64_t>& other) const {
    return _mm256_cmpgt_epi64(other.values, values);
  }
  Vec256<int64_t> operator<=(const Vec256<int64_t>& other) const {
    return invert(_mm256_cmpgt_epi64(values, other.values));
  }
  Vec256<int64_t> operator>(const Vec256<int64_t>& other) const {
    return _mm256_cmpgt_epi64(values, other.values);
  }
  Vec256<int64_t> operator>=(const Vec256<int64_t>& other) const {
    return invert(_mm256_cmpgt_epi64(other.values, values));
  }

  Vec256<int64_t> eq(const Vec256<int64_t>& other) const;
  Vec256<int64_t> ne(const Vec256<int64_t>& other) const;
  Vec256<int64_t> gt(const Vec256<int64_t>& other) const;
  Vec256<int64_t> ge(const Vec256<int64_t>& other) const;
  Vec256<int64_t> lt(const Vec256<int64_t>& other) const;
  Vec256<int64_t> le(const Vec256<int64_t>& other) const;
};

template <>
class Vec256<int32_t> : public Vec256i {
private:
  static const Vec256<int32_t> ones;
public:
  using value_type = int32_t;
  static constexpr int size() {
    return 8;
  }
  using Vec256i::Vec256i;
  Vec256() {}
  Vec256(int32_t v) { values = _mm256_set1_epi32(v); }
  Vec256(int32_t val1, int32_t val2, int32_t val3, int32_t val4,
         int32_t val5, int32_t val6, int32_t val7, int32_t val8) {
    values = _mm256_setr_epi32(val1, val2, val3, val4, val5, val6, val7, val8);
  }
  template <int64_t mask>
  static Vec256<int32_t> blend(Vec256<int32_t> a, Vec256<int32_t> b) {
    return _mm256_blend_epi32(a, b, mask);
  }
  static Vec256<int32_t> blendv(const Vec256<int32_t>& a, const Vec256<int32_t>& b,
                                const Vec256<int32_t>& mask) {
    return _mm256_blendv_epi8(a.values, b.values, mask.values);
  }
  template <typename step_t>
  static Vec256<int32_t> arange(int32_t base = 0, step_t step = static_cast<step_t>(1)) {
    return Vec256<int32_t>(
      base,            base +     step, base + 2 * step, base + 3 * step,
      base + 4 * step, base + 5 * step, base + 6 * step, base + 7 * step);
  }
  static Vec256<int32_t>
  set(Vec256<int32_t> a, Vec256<int32_t> b, int32_t count = size()) {
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
    __at_align32__ int32_t tmp_values[size()];
    // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
    // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
    // instructions while a loop would be compiled to one instruction.
    for (auto i = 0; i < size(); ++i) {
      tmp_values[i] = 0;
    }
    std::memcpy(tmp_values, ptr, count * sizeof(int32_t));
    return loadu(tmp_values);
  }
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      // ptr need not to be aligned here. See
      // https://software.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-intel-advanced-vector-extensions/intrinsics-for-load-and-store-operations-1/mm256-storeu-si256.html
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
    } else if (count > 0) {
      __at_align32__ int32_t tmp_values[size()];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(int32_t));
    }
  }
  void dump() const {
      for (size_t i = 0; i < size(); ++i) {
          std::cout << (int)((value_type*)&values)[i] << " ";
      }
      std::cout << std::endl;
  }
  const int32_t& operator[](int idx) const  = delete;
  int32_t& operator[](int idx)  = delete;
  Vec256<int32_t> abs() const {
    return _mm256_abs_epi32(values);
  }
  Vec256<int32_t> real() const {
    return *this;
  }
  Vec256<int32_t> imag() const {
    return _mm256_set1_epi32(0);
  }
  Vec256<int32_t> conj() const {
    return *this;
  }
  Vec256<int32_t> frac() const;
  Vec256<int32_t> neg() const;
  Vec256<int32_t> operator==(const Vec256<int32_t>& other) const {
    return _mm256_cmpeq_epi32(values, other.values);
  }
  Vec256<int32_t> operator!=(const Vec256<int32_t>& other) const {
    return invert(_mm256_cmpeq_epi32(values, other.values));
  }
  Vec256<int32_t> operator<(const Vec256<int32_t>& other) const {
    return _mm256_cmpgt_epi32(other.values, values);
  }
  Vec256<int32_t> operator<=(const Vec256<int32_t>& other) const {
    return invert(_mm256_cmpgt_epi32(values, other.values));
  }
  Vec256<int32_t> operator>(const Vec256<int32_t>& other) const {
    return _mm256_cmpgt_epi32(values, other.values);
  }
  Vec256<int32_t> operator>=(const Vec256<int32_t>& other) const {
    return invert(_mm256_cmpgt_epi32(other.values, values));
  }
  Vec256<int32_t> eq(const Vec256<int32_t>& other) const;
  Vec256<int32_t> ne(const Vec256<int32_t>& other) const;
  Vec256<int32_t> gt(const Vec256<int32_t>& other) const;
  Vec256<int32_t> ge(const Vec256<int32_t>& other) const;
  Vec256<int32_t> lt(const Vec256<int32_t>& other) const;
  Vec256<int32_t> le(const Vec256<int32_t>& other) const;
};

template <>
inline void convert(const int32_t *src, float *dst, int64_t n) {
  int64_t i;
  // int32_t and float have same size
#ifndef _MSC_VER
# pragma unroll
#endif
  for (i = 0; i <= (n - Vec256<int32_t>::size()); i += Vec256<int32_t>::size()) {
    auto input_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
    auto output_vec = _mm256_cvtepi32_ps(input_vec);
    _mm256_storeu_ps(reinterpret_cast<float*>(dst + i), output_vec);
  }
#ifndef _MSC_VER
# pragma unroll
#endif
  for (; i < n; i++) {
    dst[i] = static_cast<float>(src[i]);
  }
}

template <>
inline void convert(const int32_t *src, double *dst, int64_t n) {
  int64_t i;
  // int32_t has half the size of double
#ifndef _MSC_VER
# pragma unroll
#endif
  for (i = 0; i <= (n - Vec256<double>::size()); i += Vec256<double>::size()) {
    auto input_128_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
    auto output_vec = _mm256_cvtepi32_pd(input_128_vec);
    _mm256_storeu_pd(reinterpret_cast<double*>(dst + i), output_vec);
  }
#ifndef _MSC_VER
# pragma unroll
#endif
  for (; i < n; i++) {
    dst[i] = static_cast<double>(src[i]);
  }
}

template <>
class Vec256<int16_t> : public Vec256i {
private:
  static const Vec256<int16_t> ones;
public:
  using value_type = int16_t;
  static constexpr int size() {
    return 16;
  }
  using Vec256i::Vec256i;
  Vec256() {}
  Vec256(int16_t v) { values = _mm256_set1_epi16(v); }
  Vec256(int16_t val1, int16_t val2, int16_t val3, int16_t val4,
         int16_t val5, int16_t val6, int16_t val7, int16_t val8,
         int16_t val9, int16_t val10, int16_t val11, int16_t val12,
         int16_t val13, int16_t val14, int16_t val15, int16_t val16) {
    values = _mm256_setr_epi16(val1, val2, val3, val4, val5, val6, val7, val8,
                               val9, val10, val11, val12, val13, val14, val15, val16);
  }
  template <int64_t mask>
  static Vec256<int16_t> blend(Vec256<int16_t> a, Vec256<int16_t> b) {
    __at_align32__ int16_t tmp_values[size()];
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
  static Vec256<int16_t> blendv(const Vec256<int16_t>& a, const Vec256<int16_t>& b,
                                const Vec256<int16_t>& mask) {
    return _mm256_blendv_epi8(a.values, b.values, mask.values);
  }
  template <typename step_t>
  static Vec256<int16_t> arange(int16_t base = 0, step_t step = static_cast<step_t>(1)) {
    return Vec256<int16_t>(
      base,             base +      step, base +  2 * step, base +  3 * step,
      base +  4 * step, base +  5 * step, base +  6 * step, base +  7 * step,
      base +  8 * step, base +  9 * step, base + 10 * step, base + 11 * step,
      base + 12 * step, base + 13 * step, base + 14 * step, base + 15 * step);
  }
  static Vec256<int16_t>
  set(Vec256<int16_t> a, Vec256<int16_t> b, int16_t count = size()) {
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
    __at_align32__ int16_t tmp_values[size()];
    // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
    // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
    // instructions while a loop would be compiled to one instruction.
    for (auto i = 0; i < size(); ++i) {
      tmp_values[i] = 0;
    }
    std::memcpy(tmp_values, ptr, count * sizeof(int16_t));
    return loadu(tmp_values);
  }
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      // ptr need not to be aligned here. See
      // https://software.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-intel-advanced-vector-extensions/intrinsics-for-load-and-store-operations-1/mm256-storeu-si256.html
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
    } else if (count > 0) {
      __at_align32__ int16_t tmp_values[size()];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(int16_t));
    }
  }
  const int16_t& operator[](int idx) const  = delete;
  int16_t& operator[](int idx)  = delete;
  Vec256<int16_t> abs() const {
    return _mm256_abs_epi16(values);
  }
  Vec256<int16_t> real() const {
    return *this;
  }
  Vec256<int16_t> imag() const {
    return _mm256_set1_epi16(0);
  }
  Vec256<int16_t> conj() const {
    return *this;
  }
  Vec256<int16_t> frac() const;
  Vec256<int16_t> neg() const;
  Vec256<int16_t> operator==(const Vec256<int16_t>& other) const {
    return _mm256_cmpeq_epi16(values, other.values);
  }
  Vec256<int16_t> operator!=(const Vec256<int16_t>& other) const {
    return invert(_mm256_cmpeq_epi16(values, other.values));
  }
  Vec256<int16_t> operator<(const Vec256<int16_t>& other) const {
    return _mm256_cmpgt_epi16(other.values, values);
  }
  Vec256<int16_t> operator<=(const Vec256<int16_t>& other) const {
    return invert(_mm256_cmpgt_epi16(values, other.values));
  }
  Vec256<int16_t> operator>(const Vec256<int16_t>& other) const {
    return _mm256_cmpgt_epi16(values, other.values);
  }
  Vec256<int16_t> operator>=(const Vec256<int16_t>& other) const {
    return invert(_mm256_cmpgt_epi16(other.values, values));
  }

  Vec256<int16_t> eq(const Vec256<int16_t>& other) const;
  Vec256<int16_t> ne(const Vec256<int16_t>& other) const;
  Vec256<int16_t> gt(const Vec256<int16_t>& other) const;
  Vec256<int16_t> ge(const Vec256<int16_t>& other) const;
  Vec256<int16_t> lt(const Vec256<int16_t>& other) const;
  Vec256<int16_t> le(const Vec256<int16_t>& other) const;
};

template <>
class Vec256<int8_t> : public Vec256i {
private:
  static const Vec256<int8_t> ones;
public:
  using value_type = int8_t;
  static constexpr int size() {
    return 32;
  }
  using Vec256i::Vec256i;
  Vec256() {}
  Vec256(int8_t v) { values = _mm256_set1_epi8(v); }
  Vec256(int8_t val1, int8_t val2, int8_t val3, int8_t val4,
         int8_t val5, int8_t val6, int8_t val7, int8_t val8,
         int8_t val9, int8_t val10, int8_t val11, int8_t val12,
         int8_t val13, int8_t val14, int8_t val15, int8_t val16,
         int8_t val17, int8_t val18, int8_t val19, int8_t val20,
         int8_t val21, int8_t val22, int8_t val23, int8_t val24,
         int8_t val25, int8_t val26, int8_t val27, int8_t val28,
         int8_t val29, int8_t val30, int8_t val31, int8_t val32) {
    values = _mm256_setr_epi8(val1, val2, val3, val4, val5, val6, val7, val8,
                              val9, val10, val11, val12, val13, val14, val15, val16,
                              val17, val18, val19, val20, val21, val22, val23, val24,
                              val25, val26, val27, val28, val29, val30, val31, val32);
  }
  template <int64_t mask>
  static Vec256<int8_t> blend(Vec256<int8_t> a, Vec256<int8_t> b) {
    __at_align32__ int8_t tmp_values[size()];
    a.store(tmp_values);
    if (mask & 0x01)
      tmp_values[0] = _mm256_extract_epi8(b.values, 0);
    if (mask & 0x02)
      tmp_values[1] = _mm256_extract_epi8(b.values, 1);
    if (mask & 0x04)
      tmp_values[2] = _mm256_extract_epi8(b.values, 2);
    if (mask & 0x08)
      tmp_values[3] = _mm256_extract_epi8(b.values, 3);
    if (mask & 0x10)
      tmp_values[4] = _mm256_extract_epi8(b.values, 4);
    if (mask & 0x20)
      tmp_values[5] = _mm256_extract_epi8(b.values, 5);
    if (mask & 0x40)
      tmp_values[6] = _mm256_extract_epi8(b.values, 6);
    if (mask & 0x80)
      tmp_values[7] = _mm256_extract_epi8(b.values, 7);
    if (mask & 0x100)
      tmp_values[8] = _mm256_extract_epi8(b.values, 8);
    if (mask & 0x200)
      tmp_values[9] = _mm256_extract_epi8(b.values, 9);
    if (mask & 0x400)
      tmp_values[10] = _mm256_extract_epi8(b.values, 10);
    if (mask & 0x800)
      tmp_values[11] = _mm256_extract_epi8(b.values, 11);
    if (mask & 0x1000)
      tmp_values[12] = _mm256_extract_epi8(b.values, 12);
    if (mask & 0x2000)
      tmp_values[13] = _mm256_extract_epi8(b.values, 13);
    if (mask & 0x4000)
      tmp_values[14] = _mm256_extract_epi8(b.values, 14);
    if (mask & 0x8000)
      tmp_values[15] = _mm256_extract_epi8(b.values, 15);
    if (mask & 0x010000)
      tmp_values[16] = _mm256_extract_epi8(b.values, 16);
    if (mask & 0x020000)
      tmp_values[17] = _mm256_extract_epi8(b.values, 17);
    if (mask & 0x040000)
      tmp_values[18] = _mm256_extract_epi8(b.values, 18);
    if (mask & 0x080000)
      tmp_values[19] = _mm256_extract_epi8(b.values, 19);
    if (mask & 0x100000)
      tmp_values[20] = _mm256_extract_epi8(b.values, 20);
    if (mask & 0x200000)
      tmp_values[21] = _mm256_extract_epi8(b.values, 21);
    if (mask & 0x400000)
      tmp_values[22] = _mm256_extract_epi8(b.values, 22);
    if (mask & 0x800000)
      tmp_values[23] = _mm256_extract_epi8(b.values, 23);
    if (mask & 0x1000000)
      tmp_values[24] = _mm256_extract_epi8(b.values, 24);
    if (mask & 0x2000000)
      tmp_values[25] = _mm256_extract_epi8(b.values, 25);
    if (mask & 0x4000000)
      tmp_values[26] = _mm256_extract_epi8(b.values, 26);
    if (mask & 0x8000000)
      tmp_values[27] = _mm256_extract_epi8(b.values, 27);
    if (mask & 0x10000000)
      tmp_values[28] = _mm256_extract_epi8(b.values, 28);
    if (mask & 0x20000000)
      tmp_values[29] = _mm256_extract_epi8(b.values, 29);
    if (mask & 0x40000000)
      tmp_values[30] = _mm256_extract_epi8(b.values, 30);
    if (mask & 0x80000000)
      tmp_values[31] = _mm256_extract_epi8(b.values, 31);
    return loadu(tmp_values);
  }
  static Vec256<int8_t> blendv(const Vec256<int8_t>& a, const Vec256<int8_t>& b,
                               const Vec256<int8_t>& mask) {
    return _mm256_blendv_epi8(a.values, b.values, mask.values);
  }
  template <typename step_t>
  static Vec256<int8_t> arange(int8_t base = 0, step_t step = static_cast<step_t>(1)) {
    return Vec256<int8_t>(
      base,             base +      step, base +  2 * step, base +  3 * step,
      base +  4 * step, base +  5 * step, base +  6 * step, base +  7 * step,
      base +  8 * step, base +  9 * step, base + 10 * step, base + 11 * step,
      base + 12 * step, base + 13 * step, base + 14 * step, base + 15 * step,
      base + 16 * step, base + 17 * step, base + 18 * step, base + 19 * step,
      base + 20 * step, base + 21 * step, base + 22 * step, base + 23 * step,
      base + 24 * step, base + 25 * step, base + 26 * step, base + 27 * step,
      base + 28 * step, base + 29 * step, base + 30 * step, base + 31 * step);
  }
  static Vec256<int8_t>
  set(Vec256<int8_t> a, Vec256<int8_t> b, int8_t count = size()) {
    switch (count) {
      case 0:
        return a;
      case 1:
        return blend<0x1>(a, b);
      case 2:
        return blend<0x3>(a, b);
      case 3:
        return blend<0x7>(a, b);
      case 4:
        return blend<0xF>(a, b);
      case 5:
        return blend<0x1F>(a, b);
      case 6:
        return blend<0x3F>(a, b);
      case 7:
        return blend<0x7F>(a, b);
      case 8:
        return blend<0xFF>(a, b);
      case 9:
        return blend<0x1FF>(a, b);
      case 10:
        return blend<0x3FF>(a, b);
      case 11:
        return blend<0x7FF>(a, b);
      case 12:
        return blend<0xFFF>(a, b);
      case 13:
        return blend<0x1FFF>(a, b);
      case 14:
        return blend<0x3FFF>(a, b);
      case 15:
        return blend<0x7FFF>(a, b);
      case 16:
        return blend<0xFFFF>(a, b);
      case 17:
        return blend<0x1FFFF>(a, b);
      case 18:
        return blend<0x3FFFF>(a, b);
      case 19:
        return blend<0x7FFFF>(a, b);
      case 20:
        return blend<0xFFFFF>(a, b);
      case 21:
        return blend<0x1FFFFF>(a, b);
      case 22:
        return blend<0x3FFFFF>(a, b);
      case 23:
        return blend<0x7FFFFF>(a, b);
      case 24:
        return blend<0xFFFFFF>(a, b);
      case 25:
        return blend<0x1FFFFFF>(a, b);
      case 26:
        return blend<0x3FFFFFF>(a, b);
      case 27:
        return blend<0x7FFFFFF>(a, b);
      case 28:
        return blend<0xFFFFFFF>(a, b);
      case 29:
        return blend<0x1FFFFFFF>(a, b);
      case 30:
        return blend<0x3FFFFFFF>(a, b);
      case 31:
        return blend<0x7FFFFFFF>(a, b);
    }
    return b;
  }
  static Vec256<int8_t> loadu(const void* ptr) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
  }
  static Vec256<int8_t> loadu(const void* ptr, int8_t count) {
    __at_align32__ int8_t tmp_values[size()];
    // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
    // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
    // instructions while a loop would be compiled to one instruction.
    for (size_t i = 0; i < size(); ++i) {
      tmp_values[i] = 0;
    }
    std::memcpy(tmp_values, ptr, count * sizeof(int8_t));
    return loadu(tmp_values);
  }
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      // ptr need not to be aligned here. See
      // https://software.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-intel-advanced-vector-extensions/intrinsics-for-load-and-store-operations-1/mm256-storeu-si256.html
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
    } else if (count > 0) {
      __at_align32__ int8_t tmp_values[size()];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(int8_t));
    }
  }
  const int8_t& operator[](int idx) const  = delete;
  int8_t& operator[](int idx)  = delete;
  Vec256<int8_t> abs() const {
    return _mm256_abs_epi8(values);
  }
  Vec256<int8_t> real() const {
    return *this;
  }
  Vec256<int8_t> imag() const {
    return _mm256_set1_epi8(0);
  }
  Vec256<int8_t> conj() const {
    return *this;
  }
  Vec256<int8_t> frac() const;
  Vec256<int8_t> neg() const;
  Vec256<int8_t> operator==(const Vec256<int8_t>& other) const {
    return _mm256_cmpeq_epi8(values, other.values);
  }
  Vec256<int8_t> operator!=(const Vec256<int8_t>& other) const {
    return invert(_mm256_cmpeq_epi8(values, other.values));
  }
  Vec256<int8_t> operator<(const Vec256<int8_t>& other) const {
    return _mm256_cmpgt_epi8(other.values, values);
  }
  Vec256<int8_t> operator<=(const Vec256<int8_t>& other) const {
    return invert(_mm256_cmpgt_epi8(values, other.values));
  }
  Vec256<int8_t> operator>(const Vec256<int8_t>& other) const {
    return _mm256_cmpgt_epi8(values, other.values);
  }
  Vec256<int8_t> operator>=(const Vec256<int8_t>& other) const {
    return invert(_mm256_cmpgt_epi8(other.values, values));
  }

  Vec256<int8_t> eq(const Vec256<int8_t>& other) const;
  Vec256<int8_t> ne(const Vec256<int8_t>& other) const;
  Vec256<int8_t> gt(const Vec256<int8_t>& other) const;
  Vec256<int8_t> ge(const Vec256<int8_t>& other) const;
  Vec256<int8_t> lt(const Vec256<int8_t>& other) const;
  Vec256<int8_t> le(const Vec256<int8_t>& other) const;
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

template <>
Vec256<int8_t> inline operator+(const Vec256<int8_t>& a, const Vec256<int8_t>& b) {
  return _mm256_add_epi8(a, b);
}

template <>
Vec256<int64_t> inline operator-(const Vec256<int64_t>& a, const Vec256<int64_t>& b) {
  return _mm256_sub_epi64(a, b);
}

template <>
Vec256<int32_t> inline operator-(const Vec256<int32_t>& a, const Vec256<int32_t>& b) {
  return _mm256_sub_epi32(a, b);
}

template <>
Vec256<int16_t> inline operator-(const Vec256<int16_t>& a, const Vec256<int16_t>& b) {
  return _mm256_sub_epi16(a, b);
}

template <>
Vec256<int8_t> inline operator-(const Vec256<int8_t>& a, const Vec256<int8_t>& b) {
  return _mm256_sub_epi8(a, b);
}

// Negation. Defined here so we can utilize operator-
Vec256<int64_t> Vec256<int64_t>::neg() const {
  return Vec256<int64_t>(0) - *this;
}

Vec256<int32_t> Vec256<int32_t>::neg() const {
  return Vec256<int32_t>(0) - *this;
}

Vec256<int16_t> Vec256<int16_t>::neg() const {
  return Vec256<int16_t>(0) - *this;
}

Vec256<int8_t> Vec256<int8_t>::neg() const {
  return Vec256<int8_t>(0) - *this;
}

// Emulate operations with no native 64-bit support in avx,
// by extracting each element, performing the operation pointwise,
// then combining the results into a vector.
template <typename op_t>
Vec256<int64_t> inline emulate(const Vec256<int64_t>& a, const Vec256<int64_t>& b, const op_t& op) {
  int64_t a0 = _mm256_extract_epi64(a, 0);
  int64_t a1 = _mm256_extract_epi64(a, 1);
  int64_t a2 = _mm256_extract_epi64(a, 2);
  int64_t a3 = _mm256_extract_epi64(a, 3);

  int64_t b0 = _mm256_extract_epi64(b, 0);
  int64_t b1 = _mm256_extract_epi64(b, 1);
  int64_t b2 = _mm256_extract_epi64(b, 2);
  int64_t b3 = _mm256_extract_epi64(b, 3);

  int64_t c0 = op(a0, b0);
  int64_t c1 = op(a1, b1);
  int64_t c2 = op(a2, b2);
  int64_t c3 = op(a3, b3);

  return _mm256_set_epi64x(c3, c2, c1, c0);
}

template <typename op_t>
Vec256<int64_t> inline emulate(const Vec256<int64_t>& a, const Vec256<int64_t>& b, const Vec256<int64_t>& c, const op_t& op) {
  int64_t a0 = _mm256_extract_epi64(a, 0);
  int64_t a1 = _mm256_extract_epi64(a, 1);
  int64_t a2 = _mm256_extract_epi64(a, 2);
  int64_t a3 = _mm256_extract_epi64(a, 3);

  int64_t b0 = _mm256_extract_epi64(b, 0);
  int64_t b1 = _mm256_extract_epi64(b, 1);
  int64_t b2 = _mm256_extract_epi64(b, 2);
  int64_t b3 = _mm256_extract_epi64(b, 3);

  int64_t c0 = _mm256_extract_epi64(c, 0);
  int64_t c1 = _mm256_extract_epi64(c, 1);
  int64_t c2 = _mm256_extract_epi64(c, 2);
  int64_t c3 = _mm256_extract_epi64(c, 3);

  int64_t d0 = op(a0, b0, c0);
  int64_t d1 = op(a1, b1, c1);
  int64_t d2 = op(a2, b2, c2);
  int64_t d3 = op(a3, b3, c3);

  return _mm256_set_epi64x(d3, d2, d1, d0);
}

// AVX2 has no intrinsic for int64_t multiply so it needs to be emulated
// This could be implemented more efficiently using epi32 instructions
// This is also technically avx compatible, but then we'll need AVX
// code for add as well.
// Note: intentionally ignores undefined behavior like (-lowest * -1).
template <>
Vec256<int64_t> inline operator*(const Vec256<int64_t>& a, const Vec256<int64_t>& b) {
  return emulate(a, b, [](int64_t a_point, int64_t b_point) __ubsan_ignore_undefined__ {return a_point * b_point;});
}

template <>
Vec256<int32_t> inline operator*(const Vec256<int32_t>& a, const Vec256<int32_t>& b) {
  return _mm256_mullo_epi32(a, b);
}

template <>
Vec256<int16_t> inline operator*(const Vec256<int16_t>& a, const Vec256<int16_t>& b) {
  return _mm256_mullo_epi16(a, b);
}

template <typename T, typename Op>
Vec256<T> inline int_elementwise_binary_256(const Vec256<T>& a, const Vec256<T>& b, Op op) {
  T values_a[Vec256<T>::size()];
  T values_b[Vec256<T>::size()];
  a.store(values_a);
  b.store(values_b);
  for (int i = 0; i != Vec256<T>::size(); i++) {
    values_a[i] = op(values_a[i], values_b[i]);
  }
  return Vec256<T>::loadu(values_a);
}

template <>
Vec256<int8_t> inline operator*(const Vec256<int8_t>& a, const Vec256<int8_t>& b) {
  // We don't have an instruction for multiplying int8_t
  return int_elementwise_binary_256(a, b, std::multiplies<int8_t>());
}

template <>
Vec256<int64_t> inline minimum(const Vec256<int64_t>& a, const Vec256<int64_t>& b) {
  return emulate(a, b, [](int64_t a_point, int64_t b_point) {return std::min(a_point, b_point);});
}

template <>
Vec256<int32_t> inline minimum(const Vec256<int32_t>& a, const Vec256<int32_t>& b) {
  return _mm256_min_epi32(a, b);
}

template <>
Vec256<int16_t> inline minimum(const Vec256<int16_t>& a, const Vec256<int16_t>& b) {
  return _mm256_min_epi16(a, b);
}

template <>
Vec256<int8_t> inline minimum(const Vec256<int8_t>& a, const Vec256<int8_t>& b) {
  return _mm256_min_epi8(a, b);
}

template <>
Vec256<int64_t> inline maximum(const Vec256<int64_t>& a, const Vec256<int64_t>& b) {
  return emulate(a, b, [](int64_t a_point, int64_t b_point) {return std::max(a_point, b_point);});
}

template <>
Vec256<int32_t> inline maximum(const Vec256<int32_t>& a, const Vec256<int32_t>& b) {
  return _mm256_max_epi32(a, b);
}

template <>
Vec256<int16_t> inline maximum(const Vec256<int16_t>& a, const Vec256<int16_t>& b) {
  return _mm256_max_epi16(a, b);
}

template <>
Vec256<int8_t> inline maximum(const Vec256<int8_t>& a, const Vec256<int8_t>& b) {
  return _mm256_max_epi8(a, b);
}

template <>
Vec256<int64_t> inline clamp(const Vec256<int64_t>& a, const Vec256<int64_t>& min_val, const Vec256<int64_t>& max_val) {
  return emulate(a, min_val, max_val, [](int64_t a_point, int64_t min_point, int64_t max_point) {return std::min(max_point, std::max(a_point, min_point));});
}

template <>
Vec256<int32_t> inline clamp(const Vec256<int32_t>& a, const Vec256<int32_t>& min_val, const Vec256<int32_t>& max_val) {
  return _mm256_min_epi32(max_val, _mm256_max_epi32(a, min_val));
}

template <>
Vec256<int16_t> inline clamp(const Vec256<int16_t>& a, const Vec256<int16_t>& min_val, const Vec256<int16_t>& max_val) {
  return _mm256_min_epi16(max_val, _mm256_max_epi16(a, min_val));
}

template <>
Vec256<int8_t> inline clamp(const Vec256<int8_t>& a, const Vec256<int8_t>& min_val, const Vec256<int8_t>& max_val) {
  return _mm256_min_epi8(max_val, _mm256_max_epi8(a, min_val));
}

template <>
Vec256<int64_t> inline clamp_max(const Vec256<int64_t>& a, const Vec256<int64_t>& max_val) {
  return emulate(a, max_val, [](int64_t a_point, int64_t max_point) {return std::min(max_point, a_point);});
}

template <>
Vec256<int32_t> inline clamp_max(const Vec256<int32_t>& a, const Vec256<int32_t>& max_val) {
  return _mm256_min_epi32(max_val, a);
}

template <>
Vec256<int16_t> inline clamp_max(const Vec256<int16_t>& a, const Vec256<int16_t>& max_val) {
  return _mm256_min_epi16(max_val, a);
}

template <>
Vec256<int8_t> inline clamp_max(const Vec256<int8_t>& a, const Vec256<int8_t>& max_val) {
  return _mm256_min_epi8(max_val, a);
}

template <>
Vec256<int64_t> inline clamp_min(const Vec256<int64_t>& a, const Vec256<int64_t>& min_val) {
  return emulate(a, min_val, [](int64_t a_point, int64_t min_point) {return std::max(min_point, a_point);});
}

template <>
Vec256<int32_t> inline clamp_min(const Vec256<int32_t>& a, const Vec256<int32_t>& min_val) {
  return _mm256_max_epi32(min_val, a);
}

template <>
Vec256<int16_t> inline clamp_min(const Vec256<int16_t>& a, const Vec256<int16_t>& min_val) {
  return _mm256_max_epi16(min_val, a);
}

template <>
Vec256<int8_t> inline clamp_min(const Vec256<int8_t>& a, const Vec256<int8_t>& min_val) {
  return _mm256_max_epi8(min_val, a);
}

template<typename T>
Vec256<int32_t> inline convert_to_int32(const T* ptr) {
  return Vec256<int32_t>::loadu(ptr);
}

template<>
Vec256<int32_t> inline convert_to_int32<int8_t>(const int8_t* ptr) {
  return _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr)));
}

template<>
Vec256<int32_t> inline convert_to_int32<uint8_t>(const uint8_t* ptr) {
  return _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr)));
}

template <>
Vec256<int64_t> inline operator/(const Vec256<int64_t>& a, const Vec256<int64_t>& b) {
  return int_elementwise_binary_256(a, b, std::divides<int64_t>());
}
template <>
Vec256<int32_t> inline operator/(const Vec256<int32_t>& a, const Vec256<int32_t>& b) {
  return int_elementwise_binary_256(a, b, std::divides<int32_t>());
}
template <>
Vec256<int16_t> inline operator/(const Vec256<int16_t>& a, const Vec256<int16_t>& b) {
  return int_elementwise_binary_256(a, b, std::divides<int16_t>());
}
template <>
Vec256<int8_t> inline operator/(const Vec256<int8_t>& a, const Vec256<int8_t>& b) {
  return int_elementwise_binary_256(a, b, std::divides<int8_t>());
}

template<class T, typename std::enable_if_t<std::is_base_of<Vec256i, Vec256<T>>::value, int> = 0>
inline Vec256<T> operator&(const Vec256<T>& a, const Vec256<T>& b) {
  return _mm256_and_si256(a, b);
}
template<class T, typename std::enable_if_t<std::is_base_of<Vec256i, Vec256<T>>::value, int> = 0>
inline Vec256<T> operator|(const Vec256<T>& a, const Vec256<T>& b) {
  return _mm256_or_si256(a, b);
}
template<class T, typename std::enable_if_t<std::is_base_of<Vec256i, Vec256<T>>::value, int> = 0>
inline Vec256<T> operator^(const Vec256<T>& a, const Vec256<T>& b) {
  return _mm256_xor_si256(a, b);
}
template<class T, typename std::enable_if_t<std::is_base_of<Vec256i, Vec256<T>>::value, int> = 0>
inline Vec256<T> operator~(const Vec256<T>& a) {
  return _mm256_xor_si256(a, _mm256_set1_epi32(-1));
}

Vec256<int64_t> Vec256<int64_t>::eq(const Vec256<int64_t>& other) const {
  return (*this == other) & Vec256<int64_t>(1);
}

Vec256<int64_t> Vec256<int64_t>::ne(const Vec256<int64_t>& other) const {
  return (*this != other) & Vec256<int64_t>(1);
}

Vec256<int64_t> Vec256<int64_t>::gt(const Vec256<int64_t>& other) const {
  return (*this > other) & Vec256<int64_t>(1);
}

Vec256<int64_t> Vec256<int64_t>::ge(const Vec256<int64_t>& other) const {
  return (*this >= other) & Vec256<int64_t>(1);
}

Vec256<int64_t> Vec256<int64_t>::lt(const Vec256<int64_t>& other) const {
  return (*this < other) & Vec256<int64_t>(1);
}

Vec256<int64_t> Vec256<int64_t>::le(const Vec256<int64_t>& other) const {
  return (*this <= other) & Vec256<int64_t>(1);
}

Vec256<int32_t> Vec256<int32_t>::eq(const Vec256<int32_t>& other) const {
  return (*this == other) & Vec256<int32_t>(1);
}

Vec256<int32_t> Vec256<int32_t>::ne(const Vec256<int32_t>& other) const {
  return (*this != other) & Vec256<int32_t>(1);
}

Vec256<int32_t> Vec256<int32_t>::gt(const Vec256<int32_t>& other) const {
  return (*this > other) & Vec256<int32_t>(1);
}

Vec256<int32_t> Vec256<int32_t>::ge(const Vec256<int32_t>& other) const {
  return (*this >= other) & Vec256<int32_t>(1);
}

Vec256<int32_t> Vec256<int32_t>::lt(const Vec256<int32_t>& other) const {
  return (*this < other) & Vec256<int32_t>(1);
}

Vec256<int32_t> Vec256<int32_t>::le(const Vec256<int32_t>& other) const {
  return (*this <= other) & Vec256<int32_t>(1);
}

Vec256<int16_t> Vec256<int16_t>::eq(const Vec256<int16_t>& other) const {
  return (*this == other) & Vec256<int16_t>(1);
}

Vec256<int16_t> Vec256<int16_t>::ne(const Vec256<int16_t>& other) const {
  return (*this != other) & Vec256<int16_t>(1);
}

Vec256<int16_t> Vec256<int16_t>::gt(const Vec256<int16_t>& other) const {
  return (*this > other) & Vec256<int16_t>(1);
}

Vec256<int16_t> Vec256<int16_t>::ge(const Vec256<int16_t>& other) const {
  return (*this >= other) & Vec256<int16_t>(1);
}

Vec256<int16_t> Vec256<int16_t>::lt(const Vec256<int16_t>& other) const {
  return (*this < other) & Vec256<int16_t>(1);
}

Vec256<int16_t> Vec256<int16_t>::le(const Vec256<int16_t>& other) const {
  return (*this <= other) & Vec256<int16_t>(1);
}

Vec256<int8_t> Vec256<int8_t>::eq(const Vec256<int8_t>& other) const {
  return (*this == other) & Vec256<int8_t>(1);
}

Vec256<int8_t> Vec256<int8_t>::ne(const Vec256<int8_t>& other) const {
  return (*this != other) & Vec256<int8_t>(1);
}

Vec256<int8_t> Vec256<int8_t>::gt(const Vec256<int8_t>& other) const {
  return (*this > other) & Vec256<int8_t>(1);
}

Vec256<int8_t> Vec256<int8_t>::ge(const Vec256<int8_t>& other) const {
  return (*this >= other) & Vec256<int8_t>(1);
}

Vec256<int8_t> Vec256<int8_t>::lt(const Vec256<int8_t>& other) const {
  return (*this < other) & Vec256<int8_t>(1);
}

Vec256<int8_t> Vec256<int8_t>::le(const Vec256<int8_t>& other) const {
  return (*this <= other) & Vec256<int8_t>(1);
}

#endif

}}}
