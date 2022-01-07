#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/macros/Macros.h>

namespace at {
namespace vec {
namespace {

#ifdef CPU_CAPABILITY_AVX512

struct Vectorizedi {
protected:
  __m512i values;
  static constexpr __m512i zero_vector {0, 0, 0, 0, 0, 0, 0, 0};
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

struct Vectorizedi {};  // dummy definition to make Vectorizedi always defined

#endif // CPU_CAPABILITY_AVX512

#ifdef CPU_CAPABILITY_AVX512

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
  Vectorized() {}
  Vectorized(int64_t v) { values = _mm512_set1_epi64(v); }
  Vectorized(int64_t val1, int64_t val2, int64_t val3, int64_t val4,
         int64_t val5, int64_t val6, int64_t val7, int64_t val8) {
    values = _mm512_setr_epi64(val1, val2, val3, val4,
                                val5, val6, val7, val8);
  }
  template <int64_t mask>
  static Vectorized<int64_t> blend(Vectorized<int64_t> a, Vectorized<int64_t> b) {
    return _mm512_mask_blend_epi64(mask, a.values, b.values);
  }
  static Vectorized<int64_t> blendv(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b,
                                const Vectorized<int64_t>& mask) {
    auto msb_one = _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF);
    auto mask_ = _mm512_cmp_epi64_mask(mask, msb_one, _MM_CMPINT_EQ);
    return _mm512_mask_blend_epi64(mask_, a.values, b.values);
  }
  template <typename step_t>
  static Vectorized<int64_t> arange(int64_t base = 0, step_t step = static_cast<step_t>(1)) {
    return Vectorized<int64_t>(base,            base + step,     base + 2 * step, base + 3 * step,
                           base + 4 * step, base + 5 * step, base + 6 * step, base + 7 * step);
  }
  static Vectorized<int64_t>
  set(Vectorized<int64_t> a, Vectorized<int64_t> b, int64_t count = size()) {
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
    __at_align__ int64_t tmp_values[size()];
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
      // https://software.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-intel-advanced-vector-extensions/intrinsics-for-load-and-store-operations-1/mm512-storeu-si512.html
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), values);
    } else if (count > 0) {
      __at_align__ int64_t tmp_values[size()];
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(int64_t));
    }
  }
  const int64_t& operator[](int idx) const  = delete;
  int64_t& operator[](int idx)  = delete;
  Vectorized<int64_t> abs() const {
    auto is_larger_mask = _mm512_cmpgt_epi64_mask(zero_vector, values);
    auto is_larger = _mm512_mask_set1_epi64(zero_vector, is_larger_mask, 0xFFFFFFFFFFFFFFFF);
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
  Vectorized<int64_t> frac() const;
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
class Vectorized<int32_t> : public Vectorizedi {
private:
  static constexpr __m512i zero_vector {0, 0, 0, 0, 0, 0, 0, 0};
  static const Vectorized<int32_t> ones;
public:
  using value_type = int32_t;
  static constexpr int size() {
    return 16;
  }
  using Vectorizedi::Vectorizedi;
  Vectorized() {}
  Vectorized(int32_t v) { values = _mm512_set1_epi32(v); }
  Vectorized(int32_t val1, int32_t val2, int32_t val3, int32_t val4,
            int32_t val5, int32_t val6, int32_t val7, int32_t val8,
            int32_t val9, int32_t val10, int32_t val11, int32_t val12,
            int32_t val13, int32_t val14, int32_t val15, int32_t val16) {
    values = _mm512_setr_epi32(val1, val2, val3, val4, val5, val6, val7, val8,
                               val9, val10, val11, val12, val13, val14, val15, val16);
  }
  template <int64_t mask>
  static Vectorized<int32_t> blend(Vectorized<int32_t> a, Vectorized<int32_t> b) {
    return _mm512_mask_blend_epi32(mask, a.values, b.values);
  }
  static Vectorized<int32_t> blendv(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b,
                                const Vectorized<int32_t>& mask) {
    auto msb_one = _mm512_set1_epi32(0xFFFFFFFF);
    auto mask_ = _mm512_cmp_epi32_mask(mask, msb_one, _MM_CMPINT_EQ);
    return _mm512_mask_blend_epi32(mask_, a.values, b.values);
  }
  template <typename step_t>
  static Vectorized<int32_t> arange(int32_t base = 0, step_t step = static_cast<step_t>(1)) {
    return Vectorized<int32_t>(
      base,             base +      step, base +  2 * step, base +  3 * step,
      base +  4 * step, base +  5 * step, base +  6 * step, base +  7 * step,
      base +  8 * step, base +  9 * step, base + 10 * step, base + 11 * step,
      base + 12 * step, base + 13 * step, base + 14 * step, base + 15 * step);
  }
  static Vectorized<int32_t>
  set(Vectorized<int32_t> a, Vectorized<int32_t> b, int32_t count = size()) {
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
    __at_align__ int32_t tmp_values[size()];
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
      // https://software.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-intel-advanced-vector-extensions/intrinsics-for-load-and-store-operations-1/mm512-storeu-si512.html
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), values);
    } else if (count > 0) {
      __at_align__ int32_t tmp_values[size()];
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(int32_t));
    }
  }
  const int32_t& operator[](int idx) const  = delete;
  int32_t& operator[](int idx)  = delete;
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
  Vectorized<int32_t> frac() const;
  Vectorized<int32_t> neg() const;
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
inline void convert(const int32_t *src, float *dst, int64_t n) {
  int64_t i;
  // int32_t and float have same size
#ifndef _MSC_VER
# pragma unroll
#endif
  for (i = 0; i <= (n - Vectorized<int32_t>::size()); i += Vectorized<int32_t>::size()) {
    auto input_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + i));
    auto output_vec = _mm512_cvtepi32_ps(input_vec);
    _mm512_storeu_ps(reinterpret_cast<float*>(dst + i), output_vec);
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
  for (i = 0; i <= (n - Vectorized<double>::size()); i += Vectorized<double>::size()) {
    auto input_256_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
    auto output_vec = _mm512_cvtepi32_pd(input_256_vec);
    _mm512_storeu_pd(reinterpret_cast<double*>(dst + i), output_vec);
  }
#ifndef _MSC_VER
# pragma unroll
#endif
  for (; i < n; i++) {
    dst[i] = static_cast<double>(src[i]);
  }
}

template <>
class Vectorized<int16_t> : public Vectorizedi {
private:
  static const Vectorized<int16_t> ones;
  static constexpr __m512i zero_vector {0, 0, 0, 0, 0, 0, 0, 0};
public:
  using value_type = int16_t;
  static constexpr int size() {
    return 32;
  }
  using Vectorizedi::Vectorizedi;
  Vectorized() {}
  Vectorized(int16_t v) { values = _mm512_set1_epi16(v); }
  Vectorized(int16_t val1, int16_t val2, int16_t val3, int16_t val4,
         int16_t val5, int16_t val6, int16_t val7, int16_t val8,
         int16_t val9, int16_t val10, int16_t val11, int16_t val12,
         int16_t val13, int16_t val14, int16_t val15, int16_t val16,
         int16_t val17, int16_t val18, int16_t val19, int16_t val20,
         int16_t val21, int16_t val22, int16_t val23, int16_t val24,
         int16_t val25, int16_t val26, int16_t val27, int16_t val28,
         int16_t val29, int16_t val30, int16_t val31, int16_t val32) {
    values = _mm512_set_epi16(val32, val31, val30, val29, val28, val27, val26, val25,
                              val24, val23, val22, val21, val20, val19, val18, val17,
                              val16, val15, val14, val13, val12, val11, val10, val9,
                              val8, val7, val6, val5, val4, val3, val2, val1);
  }
  template <int64_t mask>
  static Vectorized<int16_t> blend(Vectorized<int16_t> a, Vectorized<int16_t> b) {
    return _mm512_mask_blend_epi16(mask, a.values, b.values);
  }
  static Vectorized<int16_t> blendv(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b,
                                const Vectorized<int16_t>& mask) {
    auto msb_one = _mm512_set1_epi16(0xFFFF);
    auto mask_ = _mm512_cmp_epi16_mask(mask, msb_one, _MM_CMPINT_EQ);
    return _mm512_mask_blend_epi16(mask_, a.values, b.values);
  }
  template <typename step_t>
  static Vectorized<int16_t> arange(int16_t base = 0, step_t step = static_cast<step_t>(1)) {
    return Vectorized<int16_t>(
      base,             base +      step, base +  2 * step, base +  3 * step,
      base +  4 * step, base +  5 * step, base +  6 * step, base +  7 * step,
      base +  8 * step, base +  9 * step, base + 10 * step, base + 11 * step,
      base + 12 * step, base + 13 * step, base + 14 * step, base + 15 * step,
      base + 16 * step, base + 17 * step, base + 18 * step, base + 19 * step,
      base + 20 * step, base + 21 * step, base + 22 * step, base + 23 * step,
      base + 24 * step, base + 25 * step, base + 26 * step, base + 27 * step,
      base + 28 * step, base + 29 * step, base + 30 * step, base + 31 * step
    );
  }
  static Vectorized<int16_t>
  set(Vectorized<int16_t> a, Vectorized<int16_t> b, int16_t count = size()) {
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
    __at_align__ int16_t tmp_values[size()];
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
      // https://software.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-intel-advanced-vector-extensions/intrinsics-for-load-and-store-operations-1/mm512-storeu-si512.html
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), values);
    } else if (count > 0) {
      __at_align__ int16_t tmp_values[size()];
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(int16_t));
    }
  }
  const int16_t& operator[](int idx) const  = delete;
  int16_t& operator[](int idx)  = delete;
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
  Vectorized<int16_t> frac() const;
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

template <>
class Vectorized<int8_t> : public Vectorizedi {
private:
  static constexpr __m512i zero_vector {0, 0, 0, 0, 0, 0, 0, 0};
  static const Vectorized<int8_t> ones;
public:
  using value_type = int8_t;
  static constexpr int size() {
    return 64;
  }
  using Vectorizedi::Vectorizedi;
  Vectorized() {}
  Vectorized(int8_t v) { values = _mm512_set1_epi8(v); }
  Vectorized(int8_t val1, int8_t val2, int8_t val3, int8_t val4,
         int8_t val5, int8_t val6, int8_t val7, int8_t val8,
         int8_t val9, int8_t val10, int8_t val11, int8_t val12,
         int8_t val13, int8_t val14, int8_t val15, int8_t val16,
         int8_t val17, int8_t val18, int8_t val19, int8_t val20,
         int8_t val21, int8_t val22, int8_t val23, int8_t val24,
         int8_t val25, int8_t val26, int8_t val27, int8_t val28,
         int8_t val29, int8_t val30, int8_t val31, int8_t val32,
         int8_t val33, int8_t val34, int8_t val35, int8_t val36,
         int8_t val37, int8_t val38, int8_t val39, int8_t val40,
         int8_t val41, int8_t val42, int8_t val43, int8_t val44,
         int8_t val45, int8_t val46, int8_t val47, int8_t val48,
         int8_t val49, int8_t val50, int8_t val51, int8_t val52,
         int8_t val53, int8_t val54, int8_t val55, int8_t val56,
         int8_t val57, int8_t val58, int8_t val59, int8_t val60,
         int8_t val61, int8_t val62, int8_t val63, int8_t val64){
    values = _mm512_set_epi8(val64, val63, val62, val61, val60, val59, val58, val57,
                              val56, val55, val54, val53,val52, val51, val50, val49,
                              val48, val47, val46, val45, val44, val43, val42, val41,
                              val40, val39, val38, val37, val36, val35, val34, val33,
                              val32, val31, val30, val29, val28, val27, val26, val25,
                              val24, val23, val22, val21, val20, val19, val18, val17,
                              val16, val15, val14, val13, val12, val11, val10, val9,
                              val8, val7, val6, val5, val4, val3, val2, val1);
  }
  template <int64_t mask>
  static Vectorized<int8_t> blend(Vectorized<int8_t> a, Vectorized<int8_t> b) {
    return _mm512_mask_blend_epi8(mask, a.values, b.values);
  }
  static Vectorized<int8_t> blendv(const Vectorized<int8_t>& a, const Vectorized<int8_t>& b,
                               const Vectorized<int8_t>& mask) {
    auto msb_one = _mm512_set1_epi8(0xFF);
    auto mask_ = _mm512_cmp_epi8_mask(mask, msb_one, _MM_CMPINT_EQ);
    return _mm512_mask_blend_epi8(mask_, a.values, b.values);
  }
  template <typename step_t>
  static Vectorized<int8_t> arange(int8_t base = 0, step_t step = static_cast<step_t>(1)) {
    return Vectorized<int8_t>(
      base,             base +      step, base +  2 * step, base +  3 * step,
      base +  4 * step, base +  5 * step, base +  6 * step, base +  7 * step,
      base +  8 * step, base +  9 * step, base + 10 * step, base + 11 * step,
      base + 12 * step, base + 13 * step, base + 14 * step, base + 15 * step,
      base + 16 * step, base + 17 * step, base + 18 * step, base + 19 * step,
      base + 20 * step, base + 21 * step, base + 22 * step, base + 23 * step,
      base + 24 * step, base + 25 * step, base + 26 * step, base + 27 * step,
      base + 28 * step, base + 29 * step, base + 30 * step, base + 31 * step,
      base + 32 * step, base + 33 * step, base + 34 * step, base + 35 * step,
      base + 36 * step, base + 37 * step, base + 38 * step, base + 39 * step,
      base + 40 * step, base + 41 * step, base + 42 * step, base + 43 * step,
      base + 44 * step, base + 45 * step, base + 46 * step, base + 47 * step,
      base + 48 * step, base + 49 * step, base + 50 * step, base + 51 * step,
      base + 52 * step, base + 53 * step, base + 54 * step, base + 55 * step,
      base + 56 * step, base + 57 * step, base + 58 * step, base + 59 * step,
      base + 60 * step, base + 61 * step, base + 62 * step, base + 63 * step);
  }
  static Vectorized<int8_t>
  set(Vectorized<int8_t> a, Vectorized<int8_t> b, int8_t count = size()) {
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
  static Vectorized<int8_t> loadu(const void* ptr) {
    return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr));
  }
  static Vectorized<int8_t> loadu(const void* ptr, int8_t count) {
    __at_align__ int8_t tmp_values[size()];
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
      // https://software.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-intel-advanced-vector-extensions/intrinsics-for-load-and-store-operations-1/mm512-storeu-si512.html
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), values);
    } else if (count > 0) {
      __at_align__ int8_t tmp_values[size()];
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(int8_t));
    }
  }
  const int8_t& operator[](int idx) const  = delete;
  int8_t& operator[](int idx)  = delete;
  Vectorized<int8_t> abs() const {
    return _mm512_abs_epi8(values);
  }
  Vectorized<int8_t> real() const {
    return *this;
  }
  Vectorized<int8_t> imag() const {
    return _mm512_set1_epi8(0);
  }
  Vectorized<int8_t> conj() const {
    return *this;
  }
  Vectorized<int8_t> frac() const;
  Vectorized<int8_t> neg() const;
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
    auto mask = _mm512_cmpgt_epi8_mask(values, other.values);
    return _mm512_mask_set1_epi8(zero_vector, mask, 0xFF);
  }
  Vectorized<int8_t> operator>=(const Vectorized<int8_t>& other) const {
    auto mask = _mm512_cmpge_epi8_mask(values, other.values);
    return _mm512_mask_set1_epi8(zero_vector, mask, 0xFF);
  }

  Vectorized<int8_t> eq(const Vectorized<int8_t>& other) const;
  Vectorized<int8_t> ne(const Vectorized<int8_t>& other) const;
  Vectorized<int8_t> gt(const Vectorized<int8_t>& other) const;
  Vectorized<int8_t> ge(const Vectorized<int8_t>& other) const;
  Vectorized<int8_t> lt(const Vectorized<int8_t>& other) const;
  Vectorized<int8_t> le(const Vectorized<int8_t>& other) const;
};

template <>
Vectorized<int64_t> inline operator+(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  return _mm512_add_epi64(a, b);
}

template <>
Vectorized<int32_t> inline operator+(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
  return _mm512_add_epi32(a, b);
}

template <>
Vectorized<int16_t> inline operator+(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
  return _mm512_add_epi16(a, b);
}

template <>
Vectorized<int8_t> inline operator+(const Vectorized<int8_t>& a, const Vectorized<int8_t>& b) {
  return _mm512_add_epi8(a, b);
}

template <>
Vectorized<int64_t> inline operator-(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  return _mm512_sub_epi64(a, b);
}

template <>
Vectorized<int32_t> inline operator-(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
  return _mm512_sub_epi32(a, b);
}

template <>
Vectorized<int16_t> inline operator-(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
  return _mm512_sub_epi16(a, b);
}

template <>
Vectorized<int8_t> inline operator-(const Vectorized<int8_t>& a, const Vectorized<int8_t>& b) {
  return _mm512_sub_epi8(a, b);
}

// Negation. Defined here so we can utilize operator-
Vectorized<int64_t> Vectorized<int64_t>::neg() const {
  return Vectorized<int64_t>(0) - *this;
}

Vectorized<int32_t> Vectorized<int32_t>::neg() const {
  return Vectorized<int32_t>(0) - *this;
}

Vectorized<int16_t> Vectorized<int16_t>::neg() const {
  return Vectorized<int16_t>(0) - *this;
}

Vectorized<int8_t> Vectorized<int8_t>::neg() const {
  return Vectorized<int8_t>(0) - *this;
}

template <>
Vectorized<int64_t> inline operator*(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  return _mm512_mullo_epi64(a, b);
}

template <>
Vectorized<int32_t> inline operator*(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
  return _mm512_mullo_epi32(a, b);
}

template <>
Vectorized<int16_t> inline operator*(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
  return _mm512_mullo_epi16(a, b);
}

template <typename T, typename Op>
Vectorized<T> inline int_elementwise_binary_512(const Vectorized<T>& a, const Vectorized<T>& b, Op op) {
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
Vectorized<int8_t> inline operator*(const Vectorized<int8_t>& a, const Vectorized<int8_t>& b) {
  // We don't have an instruction for multiplying int8_t
  return int_elementwise_binary_512(a, b, std::multiplies<int8_t>());
}

template <>
Vectorized<int64_t> inline minimum(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  return _mm512_min_epi64(a, b);
}

template <>
Vectorized<int32_t> inline minimum(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
  return _mm512_min_epi32(a, b);
}

template <>
Vectorized<int16_t> inline minimum(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
  return _mm512_min_epi16(a, b);
}

template <>
Vectorized<int8_t> inline minimum(const Vectorized<int8_t>& a, const Vectorized<int8_t>& b) {
  return _mm512_min_epi8(a, b);
}

template <>
Vectorized<int64_t> inline maximum(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  return _mm512_max_epi64(a, b);
}

template <>
Vectorized<int32_t> inline maximum(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
  return _mm512_max_epi32(a, b);
}

template <>
Vectorized<int16_t> inline maximum(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
  return _mm512_max_epi16(a, b);
}

template <>
Vectorized<int8_t> inline maximum(const Vectorized<int8_t>& a, const Vectorized<int8_t>& b) {
  return _mm512_max_epi8(a, b);
}

template <>
Vectorized<int64_t> inline clamp(const Vectorized<int64_t>& a, const Vectorized<int64_t>& min_val, const Vectorized<int64_t>& max_val) {
  return _mm512_min_epi64(max_val, _mm512_max_epi64(a, min_val));
}

template <>
Vectorized<int32_t> inline clamp(const Vectorized<int32_t>& a, const Vectorized<int32_t>& min_val, const Vectorized<int32_t>& max_val) {
  return _mm512_min_epi32(max_val, _mm512_max_epi32(a, min_val));
}

template <>
Vectorized<int16_t> inline clamp(const Vectorized<int16_t>& a, const Vectorized<int16_t>& min_val, const Vectorized<int16_t>& max_val) {
  return _mm512_min_epi16(max_val, _mm512_max_epi16(a, min_val));
}

template <>
Vectorized<int8_t> inline clamp(const Vectorized<int8_t>& a, const Vectorized<int8_t>& min_val, const Vectorized<int8_t>& max_val) {
  return _mm512_min_epi8(max_val, _mm512_max_epi8(a, min_val));
}

template <>
Vectorized<int64_t> inline clamp_max(const Vectorized<int64_t>& a, const Vectorized<int64_t>& max_val) {
  return _mm512_min_epi64(max_val, a);
}

template <>
Vectorized<int32_t> inline clamp_max(const Vectorized<int32_t>& a, const Vectorized<int32_t>& max_val) {
  return _mm512_min_epi32(max_val, a);
}

template <>
Vectorized<int16_t> inline clamp_max(const Vectorized<int16_t>& a, const Vectorized<int16_t>& max_val) {
  return _mm512_min_epi16(max_val, a);
}

template <>
Vectorized<int8_t> inline clamp_max(const Vectorized<int8_t>& a, const Vectorized<int8_t>& max_val) {
  return _mm512_min_epi8(max_val, a);
}

template <>
Vectorized<int64_t> inline clamp_min(const Vectorized<int64_t>& a, const Vectorized<int64_t>& min_val) {
  return _mm512_max_epi64(min_val, a);
}

template <>
Vectorized<int32_t> inline clamp_min(const Vectorized<int32_t>& a, const Vectorized<int32_t>& min_val) {
  return _mm512_max_epi32(min_val, a);
}

template <>
Vectorized<int16_t> inline clamp_min(const Vectorized<int16_t>& a, const Vectorized<int16_t>& min_val) {
  return _mm512_max_epi16(min_val, a);
}

template <>
Vectorized<int8_t> inline clamp_min(const Vectorized<int8_t>& a, const Vectorized<int8_t>& min_val) {
  return _mm512_max_epi8(min_val, a);
}

template<typename T>
Vectorized<int32_t> inline convert_to_int32(const T* ptr) {
  return Vectorized<int32_t>::loadu(ptr);
}

template<>
Vectorized<int32_t> inline convert_to_int32<int8_t>(const int8_t* ptr) {
  return _mm512_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)));
}

template<>
Vectorized<int32_t> inline convert_to_int32<uint8_t>(const uint8_t* ptr) {
  return _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)));
}

template <>
Vectorized<int64_t> inline operator/(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  return int_elementwise_binary_512(a, b, std::divides<int64_t>());
}
template <>
Vectorized<int32_t> inline operator/(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
  return int_elementwise_binary_512(a, b, std::divides<int32_t>());
}
template <>
Vectorized<int16_t> inline operator/(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
  return int_elementwise_binary_512(a, b, std::divides<int16_t>());
}
template <>
Vectorized<int8_t> inline operator/(const Vectorized<int8_t>& a, const Vectorized<int8_t>& b) {
  return int_elementwise_binary_512(a, b, std::divides<int8_t>());
}

template<class T, typename std::enable_if_t<std::is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
inline Vectorized<T> operator&(const Vectorized<T>& a, const Vectorized<T>& b) {
  return _mm512_and_si512(a, b);
}
template<class T, typename std::enable_if_t<std::is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
inline Vectorized<T> operator|(const Vectorized<T>& a, const Vectorized<T>& b) {
  return _mm512_or_si512(a, b);
}
template<class T, typename std::enable_if_t<std::is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
inline Vectorized<T> operator^(const Vectorized<T>& a, const Vectorized<T>& b) {
  return _mm512_xor_si512(a, b);
}
template<class T, typename std::enable_if_t<std::is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
inline Vectorized<T> operator~(const Vectorized<T>& a) {
  return _mm512_xor_si512(a, _mm512_set1_epi32(-1));
}

Vectorized<int64_t> Vectorized<int64_t>::eq(const Vectorized<int64_t>& other) const {
  return (*this == other) & Vectorized<int64_t>(1);
}

Vectorized<int64_t> Vectorized<int64_t>::ne(const Vectorized<int64_t>& other) const {
  return (*this != other) & Vectorized<int64_t>(1);
}

Vectorized<int64_t> Vectorized<int64_t>::gt(const Vectorized<int64_t>& other) const {
  return (*this > other) & Vectorized<int64_t>(1);
}

Vectorized<int64_t> Vectorized<int64_t>::ge(const Vectorized<int64_t>& other) const {
  return (*this >= other) & Vectorized<int64_t>(1);
}

Vectorized<int64_t> Vectorized<int64_t>::lt(const Vectorized<int64_t>& other) const {
  return (*this < other) & Vectorized<int64_t>(1);
}

Vectorized<int64_t> Vectorized<int64_t>::le(const Vectorized<int64_t>& other) const {
  return (*this <= other) & Vectorized<int64_t>(1);
}

Vectorized<int32_t> Vectorized<int32_t>::eq(const Vectorized<int32_t>& other) const {
  return (*this == other) & Vectorized<int32_t>(1);
}

Vectorized<int32_t> Vectorized<int32_t>::ne(const Vectorized<int32_t>& other) const {
  return (*this != other) & Vectorized<int32_t>(1);
}

Vectorized<int32_t> Vectorized<int32_t>::gt(const Vectorized<int32_t>& other) const {
  return (*this > other) & Vectorized<int32_t>(1);
}

Vectorized<int32_t> Vectorized<int32_t>::ge(const Vectorized<int32_t>& other) const {
  return (*this >= other) & Vectorized<int32_t>(1);
}

Vectorized<int32_t> Vectorized<int32_t>::lt(const Vectorized<int32_t>& other) const {
  return (*this < other) & Vectorized<int32_t>(1);
}

Vectorized<int32_t> Vectorized<int32_t>::le(const Vectorized<int32_t>& other) const {
  return (*this <= other) & Vectorized<int32_t>(1);
}

Vectorized<int16_t> Vectorized<int16_t>::eq(const Vectorized<int16_t>& other) const {
  return (*this == other) & Vectorized<int16_t>(1);
}

Vectorized<int16_t> Vectorized<int16_t>::ne(const Vectorized<int16_t>& other) const {
  return (*this != other) & Vectorized<int16_t>(1);
}

Vectorized<int16_t> Vectorized<int16_t>::gt(const Vectorized<int16_t>& other) const {
  return (*this > other) & Vectorized<int16_t>(1);
}

Vectorized<int16_t> Vectorized<int16_t>::ge(const Vectorized<int16_t>& other) const {
  return (*this >= other) & Vectorized<int16_t>(1);
}

Vectorized<int16_t> Vectorized<int16_t>::lt(const Vectorized<int16_t>& other) const {
  return (*this < other) & Vectorized<int16_t>(1);
}

Vectorized<int16_t> Vectorized<int16_t>::le(const Vectorized<int16_t>& other) const {
  return (*this <= other) & Vectorized<int16_t>(1);
}

Vectorized<int8_t> Vectorized<int8_t>::eq(const Vectorized<int8_t>& other) const {
  return (*this == other) & Vectorized<int8_t>(1);
}

Vectorized<int8_t> Vectorized<int8_t>::ne(const Vectorized<int8_t>& other) const {
  return (*this != other) & Vectorized<int8_t>(1);
}

Vectorized<int8_t> Vectorized<int8_t>::gt(const Vectorized<int8_t>& other) const {
  return (*this > other) & Vectorized<int8_t>(1);
}

Vectorized<int8_t> Vectorized<int8_t>::ge(const Vectorized<int8_t>& other) const {
  return (*this >= other) & Vectorized<int8_t>(1);
}

Vectorized<int8_t> Vectorized<int8_t>::lt(const Vectorized<int8_t>& other) const {
  return (*this < other) & Vectorized<int8_t>(1);
}

Vectorized<int8_t> Vectorized<int8_t>::le(const Vectorized<int8_t>& other) const {
  return (*this <= other) & Vectorized<int8_t>(1);
}

#endif

}}}
