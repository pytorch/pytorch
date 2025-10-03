#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/macros/Macros.h>
#include <c10/util/irange.h>

namespace at::vec {
inline namespace CPU_CAPABILITY {

#ifdef CPU_CAPABILITY_AVX512

struct Vectorizedi {
 protected:
  __m512i values;
  static constexpr __m512i zero_vector{0, 0, 0, 0, 0, 0, 0, 0};
  static inline __m512i invert(const __m512i& v) {
    const auto ones = _mm512_set1_epi64(-1);
    return _mm512_xor_si512(ones, v);
  }

 public:
  Vectorizedi() {}
  Vectorizedi(__m512i v) : values(v) {}
  operator __m512i() const {
    return values;
  }
};

#else

struct Vectorizedi {}; // dummy definition to make Vectorizedi always defined

#endif // CPU_CAPABILITY_AVX512

#ifdef CPU_CAPABILITY_AVX512

template <>
struct is_vec_specialized_for<int64_t> : std::bool_constant<true> {};

template <>
class Vectorized<int64_t> : public Vectorizedi {
 private:
  static const Vectorized<int64_t> ones;

 public:
  using value_type = int64_t;
  using size_type = int;
  static constexpr size_type size() {
    return 8;
  }
  using Vectorizedi::Vectorizedi;
  Vectorized() {
    values = _mm512_setzero_si512();
  }
  Vectorized(int64_t v) {
    values = _mm512_set1_epi64(v);
  }
  Vectorized(
      int64_t val1,
      int64_t val2,
      int64_t val3,
      int64_t val4,
      int64_t val5,
      int64_t val6,
      int64_t val7,
      int64_t val8) {
    values = _mm512_setr_epi64(val1, val2, val3, val4, val5, val6, val7, val8);
  }
  template <int64_t mask>
  static Vectorized<int64_t> blend(
      Vectorized<int64_t> a,
      Vectorized<int64_t> b) {
    return _mm512_mask_blend_epi64(mask, a.values, b.values);
  }
  static Vectorized<int64_t> blendv(
      const Vectorized<int64_t>& a,
      const Vectorized<int64_t>& b,
      const Vectorized<int64_t>& mask) {
    auto msb_one = _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF);
    auto mask_ = _mm512_cmp_epi64_mask(mask, msb_one, _MM_CMPINT_EQ);
    return _mm512_mask_blend_epi64(mask_, a.values, b.values);
  }
  template <typename step_t>
  static Vectorized<int64_t> arange(
      int64_t base = 0,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<int64_t>(
        base,
        base + step,
        base + 2 * step,
        base + 3 * step,
        base + 4 * step,
        base + 5 * step,
        base + 6 * step,
        base + 7 * step);
  }
  static Vectorized<int64_t> set(
      Vectorized<int64_t> a,
      Vectorized<int64_t> b,
      int64_t count = size()) {
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
  static Vectorized<int64_t> loadu(const void* ptr) {
    return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr));
  }
  static Vectorized<int64_t> loadu(const void* ptr, int64_t count) {
    if (count == size()) {
      return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr));
    } else {
      __mmask8 mask = (1ULL << count) - 1;
      return _mm512_maskz_loadu_epi64(mask, ptr);
    }
  }
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      // ptr need not to be aligned here. See
      // https://software.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-intel-advanced-vector-extensions/intrinsics-for-load-and-store-operations-1/mm512-storeu-si512.html
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), values);
    } else if (count > 0) {
      __mmask8 mask = (1ULL << count) - 1;
      _mm512_mask_storeu_epi64(ptr, mask, values);
    }
  }
  const int64_t& operator[](int idx) const = delete;
  int64_t& operator[](int idx) = delete;
  Vectorized<int64_t> abs() const {
    auto is_larger_mask = _mm512_cmpgt_epi64_mask(zero_vector, values);
    auto is_larger =
        _mm512_mask_set1_epi64(zero_vector, is_larger_mask, 0xFFFFFFFFFFFFFFFF);
    auto inverse = _mm512_xor_si512(values, is_larger);
    return _mm512_sub_epi64(inverse, is_larger);
  }
  Vectorized<int64_t> real() const {
    return *this;
  }
  Vectorized<int64_t> imag() const {
    return _mm512_set1_epi64(0);
  }
  Vectorized<int64_t> conj() const {
    return *this;
  }
  Vectorized<int64_t> neg() const;
  Vectorized<int64_t> operator==(const Vectorized<int64_t>& other) const {
    auto mask = _mm512_cmpeq_epi64_mask(values, other.values);
    return _mm512_mask_set1_epi64(zero_vector, mask, 0xFFFFFFFFFFFFFFFF);
  }
  Vectorized<int64_t> operator!=(const Vectorized<int64_t>& other) const {
    auto mask = _mm512_cmpneq_epi64_mask(values, other.values);
    return _mm512_mask_set1_epi64(zero_vector, mask, 0xFFFFFFFFFFFFFFFF);
  }
  Vectorized<int64_t> operator<(const Vectorized<int64_t>& other) const {
    auto mask = _mm512_cmplt_epi64_mask(values, other.values);
    return _mm512_mask_set1_epi64(zero_vector, mask, 0xFFFFFFFFFFFFFFFF);
  }
  Vectorized<int64_t> operator<=(const Vectorized<int64_t>& other) const {
    auto mask = _mm512_cmple_epi64_mask(values, other.values);
    return _mm512_mask_set1_epi64(zero_vector, mask, 0xFFFFFFFFFFFFFFFF);
  }
  Vectorized<int64_t> operator>(const Vectorized<int64_t>& other) const {
    auto mask = _mm512_cmpgt_epi64_mask(values, other.values);
    return _mm512_mask_set1_epi64(zero_vector, mask, 0xFFFFFFFFFFFFFFFF);
  }
  Vectorized<int64_t> operator>=(const Vectorized<int64_t>& other) const {
    auto mask = _mm512_cmpge_epi64_mask(values, other.values);
    return _mm512_mask_set1_epi64(zero_vector, mask, 0xFFFFFFFFFFFFFFFF);
  }

  Vectorized<int64_t> eq(const Vectorized<int64_t>& other) const;
  Vectorized<int64_t> ne(const Vectorized<int64_t>& other) const;
  Vectorized<int64_t> gt(const Vectorized<int64_t>& other) const;
  Vectorized<int64_t> ge(const Vectorized<int64_t>& other) const;
  Vectorized<int64_t> lt(const Vectorized<int64_t>& other) const;
  Vectorized<int64_t> le(const Vectorized<int64_t>& other) const;
};

template <>
struct is_vec_specialized_for<int32_t> : std::bool_constant<true> {};
template <>
class Vectorized<int32_t> : public Vectorizedi {
 private:
  static constexpr __m512i zero_vector{0, 0, 0, 0, 0, 0, 0, 0};
  static const Vectorized<int32_t> ones;

 public:
  using value_type = int32_t;
  static constexpr int size() {
    return 16;
  }
  using Vectorizedi::Vectorizedi;
  Vectorized() {}
  Vectorized(int32_t v) {
    values = _mm512_set1_epi32(v);
  }
  Vectorized(
      int32_t val1,
      int32_t val2,
      int32_t val3,
      int32_t val4,
      int32_t val5,
      int32_t val6,
      int32_t val7,
      int32_t val8,
      int32_t val9,
      int32_t val10,
      int32_t val11,
      int32_t val12,
      int32_t val13,
      int32_t val14,
      int32_t val15,
      int32_t val16) {
    values = _mm512_setr_epi32(
        val1,
        val2,
        val3,
        val4,
        val5,
        val6,
        val7,
        val8,
        val9,
        val10,
        val11,
        val12,
        val13,
        val14,
        val15,
        val16);
  }
  template <int64_t mask>
  static Vectorized<int32_t> blend(
      Vectorized<int32_t> a,
      Vectorized<int32_t> b) {
    return _mm512_mask_blend_epi32(mask, a.values, b.values);
  }
  static Vectorized<int32_t> blendv(
      const Vectorized<int32_t>& a,
      const Vectorized<int32_t>& b,
      const Vectorized<int32_t>& mask) {
    auto msb_one = _mm512_set1_epi32(0xFFFFFFFF);
    auto mask_ = _mm512_cmp_epi32_mask(mask, msb_one, _MM_CMPINT_EQ);
    return _mm512_mask_blend_epi32(mask_, a.values, b.values);
  }
  template <typename step_t>
  static Vectorized<int32_t> arange(
      int32_t base = 0,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<int32_t>(
        base,
        base + step,
        base + 2 * step,
        base + 3 * step,
        base + 4 * step,
        base + 5 * step,
        base + 6 * step,
        base + 7 * step,
        base + 8 * step,
        base + 9 * step,
        base + 10 * step,
        base + 11 * step,
        base + 12 * step,
        base + 13 * step,
        base + 14 * step,
        base + 15 * step);
  }
  static Vectorized<int32_t> set(
      Vectorized<int32_t> a,
      Vectorized<int32_t> b,
      int32_t count = size()) {
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
  static Vectorized<int32_t> loadu(const void* ptr) {
    return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr));
  }
  static Vectorized<int32_t> loadu(const void* ptr, int32_t count) {
    if (count == size()) {
      return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr));
    } else {
      __mmask16 mask = (1ULL << count) - 1;
      return _mm512_maskz_loadu_epi32(mask, ptr);
    }
  }
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      // ptr need not to be aligned here. See
      // https://software.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-intel-advanced-vector-extensions/intrinsics-for-load-and-store-operations-1/mm512-storeu-si512.html
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), values);
    } else if (count > 0) {
      __mmask16 mask = (1ULL << count) - 1;
      _mm512_mask_storeu_epi32(ptr, mask, values);
    }
  }
  const int32_t& operator[](int idx) const = delete;
  int32_t& operator[](int idx) = delete;
  Vectorized<int32_t> abs() const {
    return _mm512_abs_epi32(values);
  }
  Vectorized<int32_t> real() const {
    return *this;
  }
  Vectorized<int32_t> imag() const {
    return _mm512_set1_epi32(0);
  }
  Vectorized<int32_t> conj() const {
    return *this;
  }
  Vectorized<int32_t> neg() const;
  int32_t reduce_add() const {
    return _mm512_reduce_add_epi32(values);
  }
  int32_t reduce_max() const {
    return _mm512_reduce_max_epi32(values);
  }
  Vectorized<int32_t> operator==(const Vectorized<int32_t>& other) const {
    auto mask = _mm512_cmpeq_epi32_mask(values, other.values);
    return _mm512_mask_set1_epi32(zero_vector, mask, 0xFFFFFFFF);
  }
  Vectorized<int32_t> operator!=(const Vectorized<int32_t>& other) const {
    auto mask = _mm512_cmpneq_epi32_mask(values, other.values);
    return _mm512_mask_set1_epi32(zero_vector, mask, 0xFFFFFFFF);
  }
  Vectorized<int32_t> operator<(const Vectorized<int32_t>& other) const {
    auto mask = _mm512_cmplt_epi32_mask(values, other.values);
    return _mm512_mask_set1_epi32(zero_vector, mask, 0xFFFFFFFF);
  }
  Vectorized<int32_t> operator<=(const Vectorized<int32_t>& other) const {
    auto mask = _mm512_cmple_epi32_mask(values, other.values);
    return _mm512_mask_set1_epi32(zero_vector, mask, 0xFFFFFFFF);
  }
  Vectorized<int32_t> operator>(const Vectorized<int32_t>& other) const {
    auto mask = _mm512_cmpgt_epi32_mask(values, other.values);
    return _mm512_mask_set1_epi32(zero_vector, mask, 0xFFFFFFFF);
  }
  Vectorized<int32_t> operator>=(const Vectorized<int32_t>& other) const {
    auto mask = _mm512_cmpge_epi32_mask(values, other.values);
    return _mm512_mask_set1_epi32(zero_vector, mask, 0xFFFFFFFF);
  }
  Vectorized<int32_t> eq(const Vectorized<int32_t>& other) const;
  Vectorized<int32_t> ne(const Vectorized<int32_t>& other) const;
  Vectorized<int32_t> gt(const Vectorized<int32_t>& other) const;
  Vectorized<int32_t> ge(const Vectorized<int32_t>& other) const;
  Vectorized<int32_t> lt(const Vectorized<int32_t>& other) const;
  Vectorized<int32_t> le(const Vectorized<int32_t>& other) const;
};

template <>
inline void convert(const int32_t* src, float* dst, int64_t n) {
  int64_t i;
  // int32_t and float have same size
#ifndef _MSC_VER
#pragma unroll
#endif
  for (i = 0; i <= (n - Vectorized<int32_t>::size());
       i += Vectorized<int32_t>::size()) {
    auto input_vec =
        _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + i));
    auto output_vec = _mm512_cvtepi32_ps(input_vec);
    _mm512_storeu_ps(reinterpret_cast<float*>(dst + i), output_vec);
  }
#ifndef _MSC_VER
#pragma unroll
#endif
  for (; i < n; i++) {
    dst[i] = static_cast<float>(src[i]);
  }
}

template <>
inline void convert(const int32_t* src, double* dst, int64_t n) {
  int64_t i;
  // int32_t has half the size of double
#ifndef _MSC_VER
#pragma unroll
#endif
  for (i = 0; i <= (n - Vectorized<double>::size());
       i += Vectorized<double>::size()) {
    auto input_256_vec =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
    auto output_vec = _mm512_cvtepi32_pd(input_256_vec);
    _mm512_storeu_pd(reinterpret_cast<double*>(dst + i), output_vec);
  }
#ifndef _MSC_VER
#pragma unroll
#endif
  for (; i < n; i++) {
    dst[i] = static_cast<double>(src[i]);
  }
}

template <>
struct is_vec_specialized_for<int16_t> : std::bool_constant<true> {};

template <>
class Vectorized<int16_t> : public Vectorizedi {
 private:
  static const Vectorized<int16_t> ones;
  static constexpr __m512i zero_vector{0, 0, 0, 0, 0, 0, 0, 0};

 public:
  using value_type = int16_t;
  static constexpr int size() {
    return 32;
  }
  using Vectorizedi::Vectorizedi;
  Vectorized() {}
  Vectorized(int16_t v) {
    values = _mm512_set1_epi16(v);
  }
  Vectorized(
      int16_t val1,
      int16_t val2,
      int16_t val3,
      int16_t val4,
      int16_t val5,
      int16_t val6,
      int16_t val7,
      int16_t val8,
      int16_t val9,
      int16_t val10,
      int16_t val11,
      int16_t val12,
      int16_t val13,
      int16_t val14,
      int16_t val15,
      int16_t val16,
      int16_t val17,
      int16_t val18,
      int16_t val19,
      int16_t val20,
      int16_t val21,
      int16_t val22,
      int16_t val23,
      int16_t val24,
      int16_t val25,
      int16_t val26,
      int16_t val27,
      int16_t val28,
      int16_t val29,
      int16_t val30,
      int16_t val31,
      int16_t val32) {
    values = _mm512_set_epi16(
        val32,
        val31,
        val30,
        val29,
        val28,
        val27,
        val26,
        val25,
        val24,
        val23,
        val22,
        val21,
        val20,
        val19,
        val18,
        val17,
        val16,
        val15,
        val14,
        val13,
        val12,
        val11,
        val10,
        val9,
        val8,
        val7,
        val6,
        val5,
        val4,
        val3,
        val2,
        val1);
  }
  template <int64_t mask>
  static Vectorized<int16_t> blend(
      Vectorized<int16_t> a,
      Vectorized<int16_t> b) {
    return _mm512_mask_blend_epi16(mask, a.values, b.values);
  }
  static Vectorized<int16_t> blendv(
      const Vectorized<int16_t>& a,
      const Vectorized<int16_t>& b,
      const Vectorized<int16_t>& mask) {
    auto msb_one = _mm512_set1_epi16(0xFFFF);
    auto mask_ = _mm512_cmp_epi16_mask(mask, msb_one, _MM_CMPINT_EQ);
    return _mm512_mask_blend_epi16(mask_, a.values, b.values);
  }
  template <typename step_t>
  static Vectorized<int16_t> arange(
      int16_t base = 0,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<int16_t>(
        base,
        base + step,
        base + 2 * step,
        base + 3 * step,
        base + 4 * step,
        base + 5 * step,
        base + 6 * step,
        base + 7 * step,
        base + 8 * step,
        base + 9 * step,
        base + 10 * step,
        base + 11 * step,
        base + 12 * step,
        base + 13 * step,
        base + 14 * step,
        base + 15 * step,
        base + 16 * step,
        base + 17 * step,
        base + 18 * step,
        base + 19 * step,
        base + 20 * step,
        base + 21 * step,
        base + 22 * step,
        base + 23 * step,
        base + 24 * step,
        base + 25 * step,
        base + 26 * step,
        base + 27 * step,
        base + 28 * step,
        base + 29 * step,
        base + 30 * step,
        base + 31 * step);
  }
  static Vectorized<int16_t> set(
      Vectorized<int16_t> a,
      Vectorized<int16_t> b,
      int16_t count = size()) {
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
  static Vectorized<int16_t> loadu(const void* ptr) {
    return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr));
  }
  static Vectorized<int16_t> loadu(const void* ptr, int16_t count) {
    if (count == size()) {
      return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr));
    } else {
      __mmask32 mask = (1ULL << count) - 1;
      return _mm512_maskz_loadu_epi16(mask, ptr);
    }
  }
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      // ptr need not to be aligned here. See
      // https://software.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-intel-advanced-vector-extensions/intrinsics-for-load-and-store-operations-1/mm512-storeu-si512.html
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), values);
    } else if (count > 0) {
      __mmask32 mask = (1ULL << count) - 1;
      _mm512_mask_storeu_epi16(ptr, mask, values);
    }
  }
  const int16_t& operator[](int idx) const = delete;
  int16_t& operator[](int idx) = delete;
  Vectorized<int16_t> abs() const {
    return _mm512_abs_epi16(values);
  }
  Vectorized<int16_t> real() const {
    return *this;
  }
  Vectorized<int16_t> imag() const {
    return _mm512_set1_epi16(0);
  }
  Vectorized<int16_t> conj() const {
    return *this;
  }
  Vectorized<int16_t> neg() const;
  Vectorized<int16_t> operator==(const Vectorized<int16_t>& other) const {
    auto mask = _mm512_cmpeq_epi16_mask(values, other.values);
    return _mm512_mask_set1_epi16(zero_vector, mask, 0xFFFF);
  }
  Vectorized<int16_t> operator!=(const Vectorized<int16_t>& other) const {
    auto mask = _mm512_cmpneq_epi16_mask(values, other.values);
    return _mm512_mask_set1_epi16(zero_vector, mask, 0xFFFF);
  }
  Vectorized<int16_t> operator<(const Vectorized<int16_t>& other) const {
    auto mask = _mm512_cmplt_epi16_mask(values, other.values);
    return _mm512_mask_set1_epi16(zero_vector, mask, 0xFFFF);
  }
  Vectorized<int16_t> operator<=(const Vectorized<int16_t>& other) const {
    auto mask = _mm512_cmple_epi16_mask(values, other.values);
    return _mm512_mask_set1_epi16(zero_vector, mask, 0xFFFF);
  }
  Vectorized<int16_t> operator>(const Vectorized<int16_t>& other) const {
    auto mask = _mm512_cmpgt_epi16_mask(values, other.values);
    return _mm512_mask_set1_epi16(zero_vector, mask, 0xFFFF);
  }
  Vectorized<int16_t> operator>=(const Vectorized<int16_t>& other) const {
    auto mask = _mm512_cmpge_epi16_mask(values, other.values);
    return _mm512_mask_set1_epi16(zero_vector, mask, 0xFFFF);
  }

  Vectorized<int16_t> eq(const Vectorized<int16_t>& other) const;
  Vectorized<int16_t> ne(const Vectorized<int16_t>& other) const;
  Vectorized<int16_t> gt(const Vectorized<int16_t>& other) const;
  Vectorized<int16_t> ge(const Vectorized<int16_t>& other) const;
  Vectorized<int16_t> lt(const Vectorized<int16_t>& other) const;
  Vectorized<int16_t> le(const Vectorized<int16_t>& other) const;
};

template <typename T>
class Vectorized8 : public Vectorizedi {
  static_assert(
      std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>,
      "Only int8_t/uint8_t are supported");

 protected:
  static constexpr __m512i zero_vector{0, 0, 0, 0, 0, 0, 0, 0};
  static const Vectorized<T> ones;

 public:
  using value_type = T;
  static constexpr int size() {
    return 64;
  }
  using Vectorizedi::Vectorizedi;
  Vectorized8() {}
  Vectorized8(T v) {
    values = _mm512_set1_epi8(v);
  }
  Vectorized8(
      T val1,
      T val2,
      T val3,
      T val4,
      T val5,
      T val6,
      T val7,
      T val8,
      T val9,
      T val10,
      T val11,
      T val12,
      T val13,
      T val14,
      T val15,
      T val16,
      T val17,
      T val18,
      T val19,
      T val20,
      T val21,
      T val22,
      T val23,
      T val24,
      T val25,
      T val26,
      T val27,
      T val28,
      T val29,
      T val30,
      T val31,
      T val32,
      T val33,
      T val34,
      T val35,
      T val36,
      T val37,
      T val38,
      T val39,
      T val40,
      T val41,
      T val42,
      T val43,
      T val44,
      T val45,
      T val46,
      T val47,
      T val48,
      T val49,
      T val50,
      T val51,
      T val52,
      T val53,
      T val54,
      T val55,
      T val56,
      T val57,
      T val58,
      T val59,
      T val60,
      T val61,
      T val62,
      T val63,
      T val64) {
    values = _mm512_set_epi8(
        val64,
        val63,
        val62,
        val61,
        val60,
        val59,
        val58,
        val57,
        val56,
        val55,
        val54,
        val53,
        val52,
        val51,
        val50,
        val49,
        val48,
        val47,
        val46,
        val45,
        val44,
        val43,
        val42,
        val41,
        val40,
        val39,
        val38,
        val37,
        val36,
        val35,
        val34,
        val33,
        val32,
        val31,
        val30,
        val29,
        val28,
        val27,
        val26,
        val25,
        val24,
        val23,
        val22,
        val21,
        val20,
        val19,
        val18,
        val17,
        val16,
        val15,
        val14,
        val13,
        val12,
        val11,
        val10,
        val9,
        val8,
        val7,
        val6,
        val5,
        val4,
        val3,
        val2,
        val1);
  }
  template <int64_t mask>
  static Vectorized<T> blend(Vectorized<T> a, Vectorized<T> b) {
    return _mm512_mask_blend_epi8(mask, a.values, b.values);
  }
  template <typename step_t>
  static Vectorized<T> arange(
      T base = 0,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<T>(
        base,
        base + step,
        base + 2 * step,
        base + 3 * step,
        base + 4 * step,
        base + 5 * step,
        base + 6 * step,
        base + 7 * step,
        base + 8 * step,
        base + 9 * step,
        base + 10 * step,
        base + 11 * step,
        base + 12 * step,
        base + 13 * step,
        base + 14 * step,
        base + 15 * step,
        base + 16 * step,
        base + 17 * step,
        base + 18 * step,
        base + 19 * step,
        base + 20 * step,
        base + 21 * step,
        base + 22 * step,
        base + 23 * step,
        base + 24 * step,
        base + 25 * step,
        base + 26 * step,
        base + 27 * step,
        base + 28 * step,
        base + 29 * step,
        base + 30 * step,
        base + 31 * step,
        base + 32 * step,
        base + 33 * step,
        base + 34 * step,
        base + 35 * step,
        base + 36 * step,
        base + 37 * step,
        base + 38 * step,
        base + 39 * step,
        base + 40 * step,
        base + 41 * step,
        base + 42 * step,
        base + 43 * step,
        base + 44 * step,
        base + 45 * step,
        base + 46 * step,
        base + 47 * step,
        base + 48 * step,
        base + 49 * step,
        base + 50 * step,
        base + 51 * step,
        base + 52 * step,
        base + 53 * step,
        base + 54 * step,
        base + 55 * step,
        base + 56 * step,
        base + 57 * step,
        base + 58 * step,
        base + 59 * step,
        base + 60 * step,
        base + 61 * step,
        base + 62 * step,
        base + 63 * step);
  }
  static Vectorized<T> set(Vectorized<T> a, Vectorized<T> b, T count = size()) {
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
      case 32:
        return blend<0xFFFFFFFF>(a, b);
      case 33:
        return blend<0x1FFFFFFFF>(a, b);
      case 34:
        return blend<0x3FFFFFFFF>(a, b);
      case 35:
        return blend<0x7FFFFFFFF>(a, b);
      case 36:
        return blend<0xFFFFFFFFF>(a, b);
      case 37:
        return blend<0x1FFFFFFFFF>(a, b);
      case 38:
        return blend<0x3FFFFFFFFF>(a, b);
      case 39:
        return blend<0x7FFFFFFFFF>(a, b);
      case 40:
        return blend<0xFFFFFFFFFF>(a, b);
      case 41:
        return blend<0x1FFFFFFFFFF>(a, b);
      case 42:
        return blend<0x3FFFFFFFFFF>(a, b);
      case 43:
        return blend<0x7FFFFFFFFFF>(a, b);
      case 44:
        return blend<0xFFFFFFFFFFF>(a, b);
      case 45:
        return blend<0x1FFFFFFFFFFF>(a, b);
      case 46:
        return blend<0x3FFFFFFFFFFF>(a, b);
      case 47:
        return blend<0x7FFFFFFFFFFF>(a, b);
      case 48:
        return blend<0xFFFFFFFFFFFF>(a, b);
      case 49:
        return blend<0x1FFFFFFFFFFFF>(a, b);
      case 50:
        return blend<0x3FFFFFFFFFFFF>(a, b);
      case 51:
        return blend<0x7FFFFFFFFFFFF>(a, b);
      case 52:
        return blend<0xFFFFFFFFFFFFF>(a, b);
      case 53:
        return blend<0x1FFFFFFFFFFFFF>(a, b);
      case 54:
        return blend<0x3FFFFFFFFFFFFF>(a, b);
      case 55:
        return blend<0x7FFFFFFFFFFFFF>(a, b);
      case 56:
        return blend<0xFFFFFFFFFFFFFF>(a, b);
      case 57:
        return blend<0x1FFFFFFFFFFFFFF>(a, b);
      case 58:
        return blend<0x3FFFFFFFFFFFFFF>(a, b);
      case 59:
        return blend<0x7FFFFFFFFFFFFFF>(a, b);
      case 60:
        return blend<0xFFFFFFFFFFFFFFF>(a, b);
      case 61:
        return blend<0x1FFFFFFFFFFFFFFF>(a, b);
      case 62:
        return blend<0x3FFFFFFFFFFFFFFF>(a, b);
      case 63:
        return blend<0x7FFFFFFFFFFFFFFF>(a, b);
    }
    return b;
  }
  static Vectorized<T> loadu(const void* ptr) {
    return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr));
  }
  static Vectorized<T> loadu_one_fourth(const void* ptr) {
    // Fast path if only load element number of 16.
    // Note: We didn't merge it as fast path of loadu(const void* ptr, T count),
    // Because loadu(const void* ptr, T count) requires zero initialization for
    // upper 384 bits. However, by using _mm512_castsi128_si512, the upper 384
    // bits of the result are undefined.
    // TODO<leslie> We can use _mm512_zextsi128_si512 in the furture,
    // since gcc 9.3 doesn't support it now.
    __m128i input_128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
    return _mm512_castsi128_si512(input_128);
  }
  static Vectorized<T> loadu(const void* ptr, T count) {
    if (count == size()) {
      return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr));
    } else if (count == 16) {
      // Fast path if only load element number of 16
      return loadu_one_fourth(ptr);
    } else {
      __mmask64 mask = (1ULL << count) - 1;
      return _mm512_maskz_loadu_epi8(mask, ptr);
    }
  }
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      // ptr need not to be aligned here. See
      // https://software.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-intel-advanced-vector-extensions/intrinsics-for-load-and-store-operations-1/mm512-storeu-si512.html
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), values);
    } else if (count > 0) {
      if (count == 16) {
        // Fast path if only store element number of 16
        _mm_storeu_si128(
            reinterpret_cast<__m128i*>(ptr), _mm512_castsi512_si128(values));
      } else {
        __mmask64 mask = (1ULL << count) - 1;
        _mm512_mask_storeu_epi8(ptr, mask, values);
      }
    }
  }
  const T& operator[](int idx) const = delete;
  T& operator[](int idx) = delete;
  Vectorized<T> real() const {
    return *this;
  }
  Vectorized<T> imag() const {
    return _mm512_set1_epi8(0);
  }
  Vectorized<T> conj() const {
    return *this;
  }
};

template <>
struct is_vec_specialized_for<int8_t> : std::bool_constant<true> {};

template <>
class Vectorized<int8_t> : public Vectorized8<int8_t> {
 public:
  using Vectorized8::Vectorized8;

  static Vectorized<int8_t> blendv(
      const Vectorized<int8_t>& a,
      const Vectorized<int8_t>& b,
      const Vectorized<int8_t>& mask) {
    auto msb_one = _mm512_set1_epi8(0xFF);
    auto mask_ = _mm512_cmp_epi8_mask(mask, msb_one, _MM_CMPINT_EQ);
    return _mm512_mask_blend_epi8(mask_, a.values, b.values);
  }

  Vectorized<int8_t> neg() const;

  Vectorized<int8_t> abs() const {
    return _mm512_abs_epi8(values);
  }

  Vectorized<int8_t> operator==(const Vectorized<int8_t>& other) const {
    auto mask = _mm512_cmpeq_epi8_mask(values, other.values);
    return _mm512_mask_set1_epi8(zero_vector, mask, 0xFF);
  }
  Vectorized<int8_t> operator!=(const Vectorized<int8_t>& other) const {
    auto mask = _mm512_cmpneq_epi8_mask(values, other.values);
    return _mm512_mask_set1_epi8(zero_vector, mask, 0xFF);
  }
  Vectorized<int8_t> operator<(const Vectorized<int8_t>& other) const {
    auto mask = _mm512_cmplt_epi8_mask(values, other.values);
    return _mm512_mask_set1_epi8(zero_vector, mask, 0xFF);
  }
  Vectorized<int8_t> operator<=(const Vectorized<int8_t>& other) const {
    auto mask = _mm512_cmple_epi8_mask(values, other.values);
    return _mm512_mask_set1_epi8(zero_vector, mask, 0xFF);
  }
  Vectorized<int8_t> operator>(const Vectorized<int8_t>& other) const {
    return other < *this;
  }
  Vectorized<int8_t> operator>=(const Vectorized<int8_t>& other) const {
    return other <= *this;
  }

  Vectorized<int8_t> eq(const Vectorized<int8_t>& other) const;
  Vectorized<int8_t> ne(const Vectorized<int8_t>& other) const;
  Vectorized<int8_t> gt(const Vectorized<int8_t>& other) const;
  Vectorized<int8_t> ge(const Vectorized<int8_t>& other) const;
  Vectorized<int8_t> lt(const Vectorized<int8_t>& other) const;
  Vectorized<int8_t> le(const Vectorized<int8_t>& other) const;
};

template <>
struct is_vec_specialized_for<uint8_t> : std::bool_constant<true> {};

template <>
class Vectorized<uint8_t> : public Vectorized8<uint8_t> {
 public:
  using Vectorized8::Vectorized8;

  static Vectorized<uint8_t> blendv(
      const Vectorized<uint8_t>& a,
      const Vectorized<uint8_t>& b,
      const Vectorized<uint8_t>& mask) {
    auto msb_one = _mm512_set1_epi8(0xFF);
    auto mask_ = _mm512_cmp_epu8_mask(mask, msb_one, _MM_CMPINT_EQ);
    return _mm512_mask_blend_epi8(mask_, a.values, b.values);
  }

  Vectorized<uint8_t> neg() const;

  Vectorized<uint8_t> abs() const {
    return *this;
  }

  Vectorized<uint8_t> operator==(const Vectorized<uint8_t>& other) const {
    auto mask = _mm512_cmpeq_epu8_mask(values, other.values);
    return _mm512_mask_set1_epi8(zero_vector, mask, 0xFF);
  }
  Vectorized<uint8_t> operator!=(const Vectorized<uint8_t>& other) const {
    auto mask = _mm512_cmpneq_epu8_mask(values, other.values);
    return _mm512_mask_set1_epi8(zero_vector, mask, 0xFF);
  }
  Vectorized<uint8_t> operator<(const Vectorized<uint8_t>& other) const {
    auto mask = _mm512_cmplt_epu8_mask(values, other.values);
    return _mm512_mask_set1_epi8(zero_vector, mask, 0xFF);
  }
  Vectorized<uint8_t> operator<=(const Vectorized<uint8_t>& other) const {
    auto mask = _mm512_cmple_epu8_mask(values, other.values);
    return _mm512_mask_set1_epi8(zero_vector, mask, 0xFF);
  }
  Vectorized<uint8_t> operator>(const Vectorized<uint8_t>& other) const {
    return other < *this;
  }
  Vectorized<uint8_t> operator>=(const Vectorized<uint8_t>& other) const {
    return other <= *this;
  }

  Vectorized<uint8_t> eq(const Vectorized<uint8_t>& other) const;
  Vectorized<uint8_t> ne(const Vectorized<uint8_t>& other) const;
  Vectorized<uint8_t> gt(const Vectorized<uint8_t>& other) const;
  Vectorized<uint8_t> ge(const Vectorized<uint8_t>& other) const;
  Vectorized<uint8_t> lt(const Vectorized<uint8_t>& other) const;
  Vectorized<uint8_t> le(const Vectorized<uint8_t>& other) const;
};

template <>
Vectorized<int64_t> inline operator+(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return _mm512_add_epi64(a, b);
}

template <>
Vectorized<int32_t> inline operator+(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return _mm512_add_epi32(a, b);
}

template <>
Vectorized<int16_t> inline operator+(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& b) {
  return _mm512_add_epi16(a, b);
}

template <>
Vectorized<int8_t> inline operator+(
    const Vectorized<int8_t>& a,
    const Vectorized<int8_t>& b) {
  return _mm512_add_epi8(a, b);
}

template <>
Vectorized<uint8_t> inline operator+(
    const Vectorized<uint8_t>& a,
    const Vectorized<uint8_t>& b) {
  return _mm512_add_epi8(a, b);
}

template <>
Vectorized<int64_t> inline operator-(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return _mm512_sub_epi64(a, b);
}

template <>
Vectorized<int32_t> inline operator-(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return _mm512_sub_epi32(a, b);
}

template <>
Vectorized<int16_t> inline operator-(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& b) {
  return _mm512_sub_epi16(a, b);
}

template <>
Vectorized<int8_t> inline operator-(
    const Vectorized<int8_t>& a,
    const Vectorized<int8_t>& b) {
  return _mm512_sub_epi8(a, b);
}

template <>
Vectorized<uint8_t> inline operator-(
    const Vectorized<uint8_t>& a,
    const Vectorized<uint8_t>& b) {
  return _mm512_sub_epi8(a, b);
}

// Negation. Defined here so we can utilize operator-
inline Vectorized<int64_t> Vectorized<int64_t>::neg() const {
  return Vectorized<int64_t>(0) - *this;
}

inline Vectorized<int32_t> Vectorized<int32_t>::neg() const {
  return Vectorized<int32_t>(0) - *this;
}

inline Vectorized<int16_t> Vectorized<int16_t>::neg() const {
  return Vectorized<int16_t>(0) - *this;
}

inline Vectorized<int8_t> Vectorized<int8_t>::neg() const {
  return Vectorized<int8_t>(0) - *this;
}

inline Vectorized<uint8_t> Vectorized<uint8_t>::neg() const {
  return Vectorized<uint8_t>(0) - *this;
}

template <>
Vectorized<int64_t> inline operator*(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return _mm512_mullo_epi64(a, b);
}

template <>
Vectorized<int32_t> inline operator*(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return _mm512_mullo_epi32(a, b);
}

template <>
Vectorized<int16_t> inline operator*(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& b) {
  return _mm512_mullo_epi16(a, b);
}

template <typename T, typename Op>
Vectorized<T> inline int_elementwise_binary_512(
    const Vectorized<T>& a,
    const Vectorized<T>& b,
    Op op) {
  T values_a[Vectorized<T>::size()];
  T values_b[Vectorized<T>::size()];
  a.store(values_a);
  b.store(values_b);
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    values_a[i] = op(values_a[i], values_b[i]);
  }
  return Vectorized<T>::loadu(values_a);
}

template <>
Vectorized<int8_t> inline operator*(
    const Vectorized<int8_t>& a,
    const Vectorized<int8_t>& b) {
  // We don't have an instruction for multiplying int8_t
#ifndef CPU_CAPABILITY_AVX512
  return int_elementwise_binary_512(a, b, std::multiplies<int8_t>());
#else
  __m512i mask00FF = _mm512_set1_epi16(0x00FF);
  __m512i a_lo = _mm512_srai_epi16(_mm512_slli_epi16(a, 8), 8);
  __m512i b_lo = _mm512_srai_epi16(_mm512_slli_epi16(b, 8), 8);
  __m512i a_hi = _mm512_srai_epi16(a, 8);
  __m512i b_hi = _mm512_srai_epi16(b, 8);
  __m512i res_lo = _mm512_and_si512(_mm512_mullo_epi16(a_lo, b_lo), mask00FF);
  __m512i res_hi = _mm512_slli_epi16(_mm512_mullo_epi16(a_hi, b_hi), 8);
  __m512i res = _mm512_or_si512(res_hi, res_lo);
  return res;
#endif
}

template <>
Vectorized<uint8_t> inline operator*(
    const Vectorized<uint8_t>& a,
    const Vectorized<uint8_t>& b) {
  // We don't have an instruction for multiplying uint8_t
#ifndef CPU_CAPABILITY_AVX512
  return int_elementwise_binary_512(a, b, std::multiplies<uint8_t>());
#else
  __m512i mask00FF = _mm512_set1_epi16(0x00FF);
  __m512i a_lo = _mm512_and_si512(a, mask00FF);
  __m512i b_lo = _mm512_and_si512(b, mask00FF);
  __m512i a_hi = _mm512_srli_epi16(a, 8);
  __m512i b_hi = _mm512_srli_epi16(b, 8);
  __m512i res_lo = _mm512_and_si512(_mm512_mullo_epi16(a_lo, b_lo), mask00FF);
  __m512i res_hi = _mm512_slli_epi16(_mm512_mullo_epi16(a_hi, b_hi), 8);
  __m512i res = _mm512_or_si512(res_hi, res_lo);
  return res;
#endif
}

template <>
Vectorized<int64_t> inline minimum(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return _mm512_min_epi64(a, b);
}

template <>
Vectorized<int32_t> inline minimum(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return _mm512_min_epi32(a, b);
}

template <>
Vectorized<int16_t> inline minimum(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& b) {
  return _mm512_min_epi16(a, b);
}

template <>
Vectorized<int8_t> inline minimum(
    const Vectorized<int8_t>& a,
    const Vectorized<int8_t>& b) {
  return _mm512_min_epi8(a, b);
}

template <>
Vectorized<uint8_t> inline minimum(
    const Vectorized<uint8_t>& a,
    const Vectorized<uint8_t>& b) {
  return _mm512_min_epu8(a, b);
}

template <>
Vectorized<int64_t> inline maximum(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return _mm512_max_epi64(a, b);
}

template <>
Vectorized<int32_t> inline maximum(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return _mm512_max_epi32(a, b);
}

template <>
Vectorized<int16_t> inline maximum(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& b) {
  return _mm512_max_epi16(a, b);
}

template <>
Vectorized<int8_t> inline maximum(
    const Vectorized<int8_t>& a,
    const Vectorized<int8_t>& b) {
  return _mm512_max_epi8(a, b);
}

template <>
Vectorized<uint8_t> inline maximum(
    const Vectorized<uint8_t>& a,
    const Vectorized<uint8_t>& b) {
  return _mm512_max_epu8(a, b);
}

template <>
Vectorized<int64_t> inline clamp(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& min_val,
    const Vectorized<int64_t>& max_val) {
  return _mm512_min_epi64(max_val, _mm512_max_epi64(a, min_val));
}

template <>
Vectorized<int32_t> inline clamp(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& min_val,
    const Vectorized<int32_t>& max_val) {
  return _mm512_min_epi32(max_val, _mm512_max_epi32(a, min_val));
}

template <>
Vectorized<int16_t> inline clamp(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& min_val,
    const Vectorized<int16_t>& max_val) {
  return _mm512_min_epi16(max_val, _mm512_max_epi16(a, min_val));
}

template <>
Vectorized<int8_t> inline clamp(
    const Vectorized<int8_t>& a,
    const Vectorized<int8_t>& min_val,
    const Vectorized<int8_t>& max_val) {
  return _mm512_min_epi8(max_val, _mm512_max_epi8(a, min_val));
}

template <>
Vectorized<uint8_t> inline clamp(
    const Vectorized<uint8_t>& a,
    const Vectorized<uint8_t>& min_val,
    const Vectorized<uint8_t>& max_val) {
  return _mm512_min_epu8(max_val, _mm512_max_epu8(a, min_val));
}

template <>
Vectorized<int64_t> inline clamp_max(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& max_val) {
  return _mm512_min_epi64(max_val, a);
}

template <>
Vectorized<int32_t> inline clamp_max(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& max_val) {
  return _mm512_min_epi32(max_val, a);
}

template <>
Vectorized<int16_t> inline clamp_max(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& max_val) {
  return _mm512_min_epi16(max_val, a);
}

template <>
Vectorized<int8_t> inline clamp_max(
    const Vectorized<int8_t>& a,
    const Vectorized<int8_t>& max_val) {
  return _mm512_min_epi8(max_val, a);
}

template <>
Vectorized<uint8_t> inline clamp_max(
    const Vectorized<uint8_t>& a,
    const Vectorized<uint8_t>& max_val) {
  return _mm512_min_epu8(max_val, a);
}

template <>
Vectorized<int64_t> inline clamp_min(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& min_val) {
  return _mm512_max_epi64(min_val, a);
}

template <>
Vectorized<int32_t> inline clamp_min(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& min_val) {
  return _mm512_max_epi32(min_val, a);
}

template <>
Vectorized<int16_t> inline clamp_min(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& min_val) {
  return _mm512_max_epi16(min_val, a);
}

template <>
Vectorized<int8_t> inline clamp_min(
    const Vectorized<int8_t>& a,
    const Vectorized<int8_t>& min_val) {
  return _mm512_max_epi8(min_val, a);
}

template <>
Vectorized<uint8_t> inline clamp_min(
    const Vectorized<uint8_t>& a,
    const Vectorized<uint8_t>& min_val) {
  return _mm512_max_epu8(min_val, a);
}

template <typename T>
std::enable_if_t<
    !(std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>),
    Vectorized<
        int32_t>> inline convert_to_int32(const T* ptr, int count = Vectorized<int32_t>::size()) {
  return Vectorized<int32_t>::loadu(ptr, count);
}

template <typename T>
std::
    enable_if_t<std::is_same_v<T, int8_t>, Vectorized<int32_t>> inline convert_to_int32(
        const int8_t* ptr,
        int count = Vectorized<int32_t>::size()) {
  if (count == Vectorized<int32_t>::size()) {
    return _mm512_cvtepi8_epi32(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)));
  } else {
    auto a = Vectorized<int8_t>::loadu(ptr, count);
    return _mm512_cvtepi8_epi32(_mm512_castsi512_si128(a));
  }
}

template <typename T>
std::
    enable_if_t<std::is_same_v<T, uint8_t>, Vectorized<int32_t>> inline convert_to_int32(
        const uint8_t* ptr,
        int count = Vectorized<int32_t>::size()) {
  if (count == Vectorized<int32_t>::size()) {
    return _mm512_cvtepu8_epi32(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)));
  } else {
    auto a = Vectorized<uint8_t>::loadu(ptr, count);
    return _mm512_cvtepu8_epi32(_mm512_castsi512_si128(a));
  }
}

template <>
Vectorized<int64_t> inline operator/(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return int_elementwise_binary_512(a, b, std::divides<int64_t>());
}
template <>
Vectorized<int32_t> inline operator/(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return int_elementwise_binary_512(a, b, std::divides<int32_t>());
}
template <>
Vectorized<int16_t> inline operator/(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& b) {
  return int_elementwise_binary_512(a, b, std::divides<int16_t>());
}
template <>
Vectorized<int8_t> inline operator/(
    const Vectorized<int8_t>& a,
    const Vectorized<int8_t>& b) {
  return int_elementwise_binary_512(a, b, std::divides<int8_t>());
}
template <>
Vectorized<uint8_t> inline operator/(
    const Vectorized<uint8_t>& a,
    const Vectorized<uint8_t>& b) {
  return int_elementwise_binary_512(a, b, std::divides<uint8_t>());
}

template <
    class T,
    typename std::enable_if_t<
        std::is_base_of<Vectorizedi, Vectorized<T>>::value,
        int> = 0>
inline Vectorized<T> operator&(const Vectorized<T>& a, const Vectorized<T>& b) {
  return _mm512_and_si512(a, b);
}
template <
    class T,
    typename std::enable_if_t<
        std::is_base_of<Vectorizedi, Vectorized<T>>::value,
        int> = 0>
inline Vectorized<T> operator|(const Vectorized<T>& a, const Vectorized<T>& b) {
  return _mm512_or_si512(a, b);
}
template <
    class T,
    typename std::enable_if_t<
        std::is_base_of<Vectorizedi, Vectorized<T>>::value,
        int> = 0>
inline Vectorized<T> operator^(const Vectorized<T>& a, const Vectorized<T>& b) {
  return _mm512_xor_si512(a, b);
}
template <
    class T,
    typename std::enable_if_t<
        std::is_base_of<Vectorizedi, Vectorized<T>>::value,
        int> = 0>
inline Vectorized<T> operator~(const Vectorized<T>& a) {
  return _mm512_xor_si512(a, _mm512_set1_epi32(-1));
}

inline Vectorized<int64_t> Vectorized<int64_t>::eq(
    const Vectorized<int64_t>& other) const {
  return (*this == other) & Vectorized<int64_t>(1);
}

inline Vectorized<int64_t> Vectorized<int64_t>::ne(
    const Vectorized<int64_t>& other) const {
  return (*this != other) & Vectorized<int64_t>(1);
}

inline Vectorized<int64_t> Vectorized<int64_t>::gt(
    const Vectorized<int64_t>& other) const {
  return (*this > other) & Vectorized<int64_t>(1);
}

inline Vectorized<int64_t> Vectorized<int64_t>::ge(
    const Vectorized<int64_t>& other) const {
  return (*this >= other) & Vectorized<int64_t>(1);
}

inline Vectorized<int64_t> Vectorized<int64_t>::lt(
    const Vectorized<int64_t>& other) const {
  return (*this < other) & Vectorized<int64_t>(1);
}

inline Vectorized<int64_t> Vectorized<int64_t>::le(
    const Vectorized<int64_t>& other) const {
  return (*this <= other) & Vectorized<int64_t>(1);
}

inline Vectorized<int32_t> Vectorized<int32_t>::eq(
    const Vectorized<int32_t>& other) const {
  return (*this == other) & Vectorized<int32_t>(1);
}

inline Vectorized<int32_t> Vectorized<int32_t>::ne(
    const Vectorized<int32_t>& other) const {
  return (*this != other) & Vectorized<int32_t>(1);
}

inline Vectorized<int32_t> Vectorized<int32_t>::gt(
    const Vectorized<int32_t>& other) const {
  return (*this > other) & Vectorized<int32_t>(1);
}

inline Vectorized<int32_t> Vectorized<int32_t>::ge(
    const Vectorized<int32_t>& other) const {
  return (*this >= other) & Vectorized<int32_t>(1);
}

inline Vectorized<int32_t> Vectorized<int32_t>::lt(
    const Vectorized<int32_t>& other) const {
  return (*this < other) & Vectorized<int32_t>(1);
}

inline Vectorized<int32_t> Vectorized<int32_t>::le(
    const Vectorized<int32_t>& other) const {
  return (*this <= other) & Vectorized<int32_t>(1);
}

inline Vectorized<int16_t> Vectorized<int16_t>::eq(
    const Vectorized<int16_t>& other) const {
  return (*this == other) & Vectorized<int16_t>(1);
}

inline Vectorized<int16_t> Vectorized<int16_t>::ne(
    const Vectorized<int16_t>& other) const {
  return (*this != other) & Vectorized<int16_t>(1);
}

inline Vectorized<int16_t> Vectorized<int16_t>::gt(
    const Vectorized<int16_t>& other) const {
  return (*this > other) & Vectorized<int16_t>(1);
}

inline Vectorized<int16_t> Vectorized<int16_t>::ge(
    const Vectorized<int16_t>& other) const {
  return (*this >= other) & Vectorized<int16_t>(1);
}

inline Vectorized<int16_t> Vectorized<int16_t>::lt(
    const Vectorized<int16_t>& other) const {
  return (*this < other) & Vectorized<int16_t>(1);
}

inline Vectorized<int16_t> Vectorized<int16_t>::le(
    const Vectorized<int16_t>& other) const {
  return (*this <= other) & Vectorized<int16_t>(1);
}

inline Vectorized<int8_t> Vectorized<int8_t>::eq(
    const Vectorized<int8_t>& other) const {
  return (*this == other) & Vectorized<int8_t>(1);
}

inline Vectorized<int8_t> Vectorized<int8_t>::ne(
    const Vectorized<int8_t>& other) const {
  return (*this != other) & Vectorized<int8_t>(1);
}

inline Vectorized<int8_t> Vectorized<int8_t>::gt(
    const Vectorized<int8_t>& other) const {
  return (*this > other) & Vectorized<int8_t>(1);
}

inline Vectorized<int8_t> Vectorized<int8_t>::ge(
    const Vectorized<int8_t>& other) const {
  return (*this >= other) & Vectorized<int8_t>(1);
}

inline Vectorized<int8_t> Vectorized<int8_t>::lt(
    const Vectorized<int8_t>& other) const {
  return (*this < other) & Vectorized<int8_t>(1);
}

inline Vectorized<int8_t> Vectorized<int8_t>::le(
    const Vectorized<int8_t>& other) const {
  return (*this <= other) & Vectorized<int8_t>(1);
}

inline Vectorized<uint8_t> Vectorized<uint8_t>::eq(
    const Vectorized<uint8_t>& other) const {
  return (*this == other) & Vectorized<uint8_t>(1);
}

inline Vectorized<uint8_t> Vectorized<uint8_t>::ne(
    const Vectorized<uint8_t>& other) const {
  return (*this != other) & Vectorized<uint8_t>(1);
}

inline Vectorized<uint8_t> Vectorized<uint8_t>::gt(
    const Vectorized<uint8_t>& other) const {
  return (*this > other) & Vectorized<uint8_t>(1);
}

inline Vectorized<uint8_t> Vectorized<uint8_t>::ge(
    const Vectorized<uint8_t>& other) const {
  return (*this >= other) & Vectorized<uint8_t>(1);
}

inline Vectorized<uint8_t> Vectorized<uint8_t>::lt(
    const Vectorized<uint8_t>& other) const {
  return (*this < other) & Vectorized<uint8_t>(1);
}

inline Vectorized<uint8_t> Vectorized<uint8_t>::le(
    const Vectorized<uint8_t>& other) const {
  return (*this <= other) & Vectorized<uint8_t>(1);
}

template <
    bool left_shift,
    typename T,
    typename std::enable_if_t<
        std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>,
        int> = 0>
Vectorized<T> inline shift_512_8(
    const Vectorized<T>& a,
    const Vectorized<T>& b) {
  // No vector instruction for shifting int8_t/uint8_t, so emulating
  // it instead.

  // Control masks for shuffle operation, treating 512 bits as an
  // array of 8-bit elements, and considering pairs of neighboring
  // elements.  Specifially, a mask named "ctl_M_N" (M,N in [0,1], and
  // M!=N) is set so that shuffle will move element with index M from
  // input pair into element with index N in output pair, and element
  // with index M in output pair will be set to all 0s.
  __m512i ctl_0_1 = _mm512_set_epi8(
      62,
      0x80,
      60,
      0x80,
      58,
      0x80,
      56,
      0x80,
      54,
      0x80,
      52,
      0x80,
      50,
      0x80,
      48,
      0x80,
      46,
      0x80,
      44,
      0x80,
      42,
      0x80,
      40,
      0x80,
      38,
      0x80,
      36,
      0x80,
      34,
      0x80,
      32,
      0x80,
      30,
      0x80,
      28,
      0x80,
      26,
      0x80,
      24,
      0x80,
      22,
      0x80,
      20,
      0x80,
      18,
      0x80,
      16,
      0x80,
      14,
      0x80,
      12,
      0x80,
      10,
      0x80,
      8,
      0x80,
      6,
      0x80,
      4,
      0x80,
      2,
      0x80,
      0,
      0x80);
  __m512i ctl_1_0 = _mm512_set_epi8(
      0x80,
      63,
      0x80,
      61,
      0x80,
      59,
      0x80,
      57,
      0x80,
      55,
      0x80,
      53,
      0x80,
      51,
      0x80,
      49,
      0x80,
      47,
      0x80,
      45,
      0x80,
      43,
      0x80,
      41,
      0x80,
      39,
      0x80,
      37,
      0x80,
      35,
      0x80,
      33,
      0x80,
      31,
      0x80,
      29,
      0x80,
      27,
      0x80,
      25,
      0x80,
      23,
      0x80,
      21,
      0x80,
      19,
      0x80,
      17,
      0x80,
      15,
      0x80,
      13,
      0x80,
      11,
      0x80,
      9,
      0x80,
      7,
      0x80,
      5,
      0x80,
      3,
      0x80,
      1);

  // Masks for bitwise and operation, treating 512 bits as an array of
  // 8-bit elements, and considering them in pairs of neighboring
  // elements.  A mask named "keep_M" (M in [0,1]) is set so that
  // bitwise and will copy element with index M from input pair into
  // element with the same index in output pair, while the other
  // element in output pair will be set to all 0s.
  __m512i keep_0 = _mm512_set1_epi16(0xFF);
  __m512i keep_1 = _mm512_set1_epi16(0xFF00);

  // Take each 8-bit element with idx%2==0 from input array to be
  // shifted and extend it to 16 bits so that 0s are added to the
  // right.  Then, perform shifting on this 16-bit number.  Upper 8
  // bits will be proper result of shifting original 8-bit number, so
  // write them to result array, into the same position from which
  // corresponding input element is taken.  Also, make sure that
  // result array elements with idx%2!=0 are set to all 0s.
  //
  // Note that number of bits to shift for is extended to 16 bits by
  // adding 0s to the left.  That means this number is not properly
  // sign-extended for negative values.  However, number of bits to
  // shift is treated as an unsigned integer by respective shift
  // intrinsics anyway so if negative then either with or without
  // proper sign extension, it will be interpreted as a number greater
  // than 32, and the shifting result will be the same.
  __m512i a0 = _mm512_shuffle_epi8(a, ctl_0_1);
  __m512i b0 = _mm512_and_si512(b, keep_0);
  __m512i c0;
  if (left_shift)
    c0 = _mm512_sllv_epi16(a0, b0);
  else if constexpr (std::is_same_v<T, int8_t>)
    c0 = _mm512_srav_epi16(a0, b0);
  else
    c0 = _mm512_srlv_epi16(a0, b0);
  c0 = _mm512_shuffle_epi8(c0, ctl_1_0);

  // Peform shifting the same way for input array elements with
  // idx%2==1.
  __m512i a1 = _mm512_and_si512(a, keep_1);
  __m512i b1 = _mm512_shuffle_epi8(b, ctl_1_0);
  __m512i c1;
  if (left_shift)
    c1 = _mm512_sllv_epi16(a1, b1);
  else if constexpr (std::is_same_v<T, int8_t>)
    c1 = _mm512_srav_epi16(a1, b1);
  else
    c1 = _mm512_srlv_epi16(a1, b1);
  c1 = _mm512_and_si512(c1, keep_1);

  // Merge partial results into the final result.
  __m512i c = _mm512_or_si512(c0, c1);

  return c;
}

template <>
Vectorized<int64_t> inline operator<<(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return _mm512_sllv_epi64(a, b);
}

template <>
Vectorized<int32_t> inline operator<<(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return _mm512_sllv_epi32(a, b);
}

template <>
Vectorized<int16_t> inline operator<<(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& b) {
  return _mm512_sllv_epi16(a, b);
}

template <>
Vectorized<int8_t> inline operator<<(
    const Vectorized<int8_t>& a,
    const Vectorized<int8_t>& b) {
  return shift_512_8<true>(a, b);
}

template <>
Vectorized<uint8_t> inline operator<<(
    const Vectorized<uint8_t>& a,
    const Vectorized<uint8_t>& b) {
  return shift_512_8<true>(a, b);
}

template <>
Vectorized<int64_t> inline operator>>(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return _mm512_srav_epi64(a, b);
}

template <>
Vectorized<int32_t> inline operator>>(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return _mm512_srav_epi32(a, b);
}

template <>
Vectorized<int16_t> inline operator>>(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& b) {
  return _mm512_srav_epi16(a, b);
}

template <>
Vectorized<int8_t> inline operator>>(
    const Vectorized<int8_t>& a,
    const Vectorized<int8_t>& b) {
  return shift_512_8<false>(a, b);
}

template <>
Vectorized<uint8_t> inline operator>>(
    const Vectorized<uint8_t>& a,
    const Vectorized<uint8_t>& b) {
  return shift_512_8<false>(a, b);
}

#endif

} // namespace CPU_CAPABILITY
} // namespace at::vec
