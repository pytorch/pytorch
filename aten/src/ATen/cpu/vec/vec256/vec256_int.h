#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/macros/Macros.h>
#include <c10/util/irange.h>

namespace at::vec {
inline namespace CPU_CAPABILITY {

#ifdef CPU_CAPABILITY_AVX2

struct Vectorizedi {
protected:
  __m256i values;

  static inline __m256i invert(const __m256i& v) {
    const auto ones = _mm256_set1_epi64x(-1);
    return _mm256_xor_si256(ones, v);
  }
public:
  Vectorizedi() {}
  Vectorizedi(__m256i v) : values(v) {}
  operator __m256i() const {
    return values;
  }
};

#else

struct Vectorizedi {};  // dummy definition to make Vectorizedi always defined

#endif // CPU_CAPABILITY_AVX2

#ifdef CPU_CAPABILITY_AVX2

template <>
class Vectorized<int64_t> : public Vectorizedi {
private:
  static const Vectorized<int64_t> ones;
public:
  using value_type = int64_t;
  using size_type = int;
  static constexpr size_type size() {
    return 4;
  }
  using Vectorizedi::Vectorizedi;
  Vectorized() {}
  Vectorized(int64_t v) { values = _mm256_set1_epi64x(v); }
  Vectorized(int64_t val1, int64_t val2, int64_t val3, int64_t val4) {
    values = _mm256_setr_epi64x(val1, val2, val3, val4);
  }
  template <int64_t mask>
  static Vectorized<int64_t> blend(Vectorized<int64_t> a, Vectorized<int64_t> b) {
    __at_align__ int64_t tmp_values[size()];
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
  static Vectorized<int64_t> blendv(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b,
                                const Vectorized<int64_t>& mask) {
    return _mm256_blendv_epi8(a.values, b.values, mask.values);
  }
  template <typename step_t>
  static Vectorized<int64_t> arange(int64_t base = 0, step_t step = static_cast<step_t>(1)) {
    return Vectorized<int64_t>(base, base + step, base + 2 * step, base + 3 * step);
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
    }
    return b;
  }
  static Vectorized<int64_t> loadu(const void* ptr) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
  }
  static Vectorized<int64_t> loadu(const void* ptr, int64_t count) {
    __at_align__ int64_t tmp_values[size()];
    // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
    // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
    // instructions while a loop would be compiled to one instruction.
    for (const auto i : c10::irange(size())) {
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
      __at_align__ int64_t tmp_values[size()];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(int64_t));
    }
  }
  const int64_t& operator[](int idx) const  = delete;
  int64_t& operator[](int idx)  = delete;
  Vectorized<int64_t> abs() const {
    auto zero = _mm256_set1_epi64x(0);
    auto is_larger = _mm256_cmpgt_epi64(zero, values);
    auto inverse = _mm256_xor_si256(values, is_larger);
    return _mm256_sub_epi64(inverse, is_larger);
  }
  Vectorized<int64_t> real() const {
    return *this;
  }
  Vectorized<int64_t> imag() const {
    return _mm256_set1_epi64x(0);
  }
  Vectorized<int64_t> conj() const {
    return *this;
  }
  Vectorized<int64_t> neg() const;
  Vectorized<int64_t> operator==(const Vectorized<int64_t>& other) const {
    return _mm256_cmpeq_epi64(values, other.values);
  }
  Vectorized<int64_t> operator!=(const Vectorized<int64_t>& other) const {
    return invert(_mm256_cmpeq_epi64(values, other.values));
  }
  Vectorized<int64_t> operator<(const Vectorized<int64_t>& other) const {
    return _mm256_cmpgt_epi64(other.values, values);
  }
  Vectorized<int64_t> operator<=(const Vectorized<int64_t>& other) const {
    return invert(_mm256_cmpgt_epi64(values, other.values));
  }
  Vectorized<int64_t> operator>(const Vectorized<int64_t>& other) const {
    return _mm256_cmpgt_epi64(values, other.values);
  }
  Vectorized<int64_t> operator>=(const Vectorized<int64_t>& other) const {
    return invert(_mm256_cmpgt_epi64(other.values, values));
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
  static const Vectorized<int32_t> ones;
public:
  using value_type = int32_t;
  static constexpr int size() {
    return 8;
  }
  using Vectorizedi::Vectorizedi;
  Vectorized() {}
  Vectorized(int32_t v) { values = _mm256_set1_epi32(v); }
  Vectorized(int32_t val1, int32_t val2, int32_t val3, int32_t val4,
         int32_t val5, int32_t val6, int32_t val7, int32_t val8) {
    values = _mm256_setr_epi32(val1, val2, val3, val4, val5, val6, val7, val8);
  }
  template <int64_t mask>
  static Vectorized<int32_t> blend(Vectorized<int32_t> a, Vectorized<int32_t> b) {
    return _mm256_blend_epi32(a, b, mask);
  }
  static Vectorized<int32_t> blendv(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b,
                                const Vectorized<int32_t>& mask) {
    return _mm256_blendv_epi8(a.values, b.values, mask.values);
  }
  template <typename step_t>
  static Vectorized<int32_t> arange(int32_t base = 0, step_t step = static_cast<step_t>(1)) {
    return Vectorized<int32_t>(
      base,            base +     step, base + 2 * step, base + 3 * step,
      base + 4 * step, base + 5 * step, base + 6 * step, base + 7 * step);
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
    }
    return b;
  }
  static Vectorized<int32_t> loadu(const void* ptr) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
  }
  static Vectorized<int32_t> loadu(const void* ptr, int32_t count) {
    __at_align__ int32_t tmp_values[size()];
    // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
    // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
    // instructions while a loop would be compiled to one instruction.
    for (const auto i : c10::irange(size())) {
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
      __at_align__ int32_t tmp_values[size()];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(int32_t));
    }
  }
  const int32_t& operator[](int idx) const  = delete;
  int32_t& operator[](int idx)  = delete;
  Vectorized<int32_t> abs() const {
    return _mm256_abs_epi32(values);
  }
  Vectorized<int32_t> real() const {
    return *this;
  }
  Vectorized<int32_t> imag() const {
    return _mm256_set1_epi32(0);
  }
  Vectorized<int32_t> conj() const {
    return *this;
  }
  Vectorized<int32_t> neg() const;
  int32_t reduce_add() const {
    auto v = values;
    // 128-bit shuffle
    auto v1 = _mm256_permute2f128_si256(v, v, 0x1);
    v = _mm256_add_epi32(v, v1);
    // 64-bit shuffle
    v1 = _mm256_shuffle_epi32(v, 0x4E);
    v = _mm256_add_epi32(v, v1);
    // 32-bit shuffle
    v1 = _mm256_shuffle_epi32(v, 0xB1);
    v = _mm256_add_epi32(v, v1);
    __m128i lo = _mm256_castsi256_si128(v);
    return _mm_cvtsi128_si32(lo);
  }
  int32_t reduce_max() const {
    auto v = values;
    // 128-bit shuffle
    auto v1 = _mm256_permute2f128_si256(v, v, 0x1);
    v = _mm256_max_epi32(v, v1);
    // 64-bit shuffle
    v1 = _mm256_shuffle_epi32(v, 0x4E);
    v = _mm256_max_epi32(v, v1);
    // 32-bit shuffle
    v1 = _mm256_shuffle_epi32(v, 0xB1);
    v = _mm256_max_epi32(v, v1);
    __m128i lo = _mm256_castsi256_si128(v);
    return _mm_cvtsi128_si32(lo);
  }
  Vectorized<int32_t> operator==(const Vectorized<int32_t>& other) const {
    return _mm256_cmpeq_epi32(values, other.values);
  }
  Vectorized<int32_t> operator!=(const Vectorized<int32_t>& other) const {
    return invert(_mm256_cmpeq_epi32(values, other.values));
  }
  Vectorized<int32_t> operator<(const Vectorized<int32_t>& other) const {
    return _mm256_cmpgt_epi32(other.values, values);
  }
  Vectorized<int32_t> operator<=(const Vectorized<int32_t>& other) const {
    return invert(_mm256_cmpgt_epi32(values, other.values));
  }
  Vectorized<int32_t> operator>(const Vectorized<int32_t>& other) const {
    return _mm256_cmpgt_epi32(values, other.values);
  }
  Vectorized<int32_t> operator>=(const Vectorized<int32_t>& other) const {
    return invert(_mm256_cmpgt_epi32(other.values, values));
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
  for (i = 0; i <= (n - Vectorized<double>::size()); i += Vectorized<double>::size()) {
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
class Vectorized<int16_t> : public Vectorizedi {
private:
  static const Vectorized<int16_t> ones;
public:
  using value_type = int16_t;
  static constexpr int size() {
    return 16;
  }
  using Vectorizedi::Vectorizedi;
  Vectorized() {}
  Vectorized(int16_t v) { values = _mm256_set1_epi16(v); }
  Vectorized(int16_t val1, int16_t val2, int16_t val3, int16_t val4,
         int16_t val5, int16_t val6, int16_t val7, int16_t val8,
         int16_t val9, int16_t val10, int16_t val11, int16_t val12,
         int16_t val13, int16_t val14, int16_t val15, int16_t val16) {
    values = _mm256_setr_epi16(val1, val2, val3, val4, val5, val6, val7, val8,
                               val9, val10, val11, val12, val13, val14, val15, val16);
  }
  template <int64_t mask>
  static Vectorized<int16_t> blend(Vectorized<int16_t> a, Vectorized<int16_t> b) {
    __at_align__ int16_t tmp_values[size()];
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
  static Vectorized<int16_t> blendv(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b,
                                const Vectorized<int16_t>& mask) {
    return _mm256_blendv_epi8(a.values, b.values, mask.values);
  }
  template <typename step_t>
  static Vectorized<int16_t> arange(int16_t base = 0, step_t step = static_cast<step_t>(1)) {
    return Vectorized<int16_t>(
      base,             base +      step, base +  2 * step, base +  3 * step,
      base +  4 * step, base +  5 * step, base +  6 * step, base +  7 * step,
      base +  8 * step, base +  9 * step, base + 10 * step, base + 11 * step,
      base + 12 * step, base + 13 * step, base + 14 * step, base + 15 * step);
  }
  static Vectorized<int16_t>
  set(Vectorized<int16_t> a, Vectorized<int16_t> b, int16_t count = size()) {
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
  static Vectorized<int16_t> loadu(const void* ptr) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
  }
  static Vectorized<int16_t> loadu(const void* ptr, int16_t count) {
    __at_align__ int16_t tmp_values[size()];
    // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
    // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
    // instructions while a loop would be compiled to one instruction.
    for (const auto i : c10::irange(size())) {
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
      __at_align__ int16_t tmp_values[size()];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(int16_t));
    }
  }
  const int16_t& operator[](int idx) const  = delete;
  int16_t& operator[](int idx)  = delete;
  Vectorized<int16_t> abs() const {
    return _mm256_abs_epi16(values);
  }
  Vectorized<int16_t> real() const {
    return *this;
  }
  Vectorized<int16_t> imag() const {
    return _mm256_set1_epi16(0);
  }
  Vectorized<int16_t> conj() const {
    return *this;
  }
  Vectorized<int16_t> neg() const;
  Vectorized<int16_t> operator==(const Vectorized<int16_t>& other) const {
    return _mm256_cmpeq_epi16(values, other.values);
  }
  Vectorized<int16_t> operator!=(const Vectorized<int16_t>& other) const {
    return invert(_mm256_cmpeq_epi16(values, other.values));
  }
  Vectorized<int16_t> operator<(const Vectorized<int16_t>& other) const {
    return _mm256_cmpgt_epi16(other.values, values);
  }
  Vectorized<int16_t> operator<=(const Vectorized<int16_t>& other) const {
    return invert(_mm256_cmpgt_epi16(values, other.values));
  }
  Vectorized<int16_t> operator>(const Vectorized<int16_t>& other) const {
    return _mm256_cmpgt_epi16(values, other.values);
  }
  Vectorized<int16_t> operator>=(const Vectorized<int16_t>& other) const {
    return invert(_mm256_cmpgt_epi16(other.values, values));
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
  static const Vectorized<T> ones;
public:
  using value_type = T;
  static constexpr int size() {
    return 32;
  }
  using Vectorizedi::Vectorizedi;
  Vectorized8() {}
  Vectorized8(T v) { values = _mm256_set1_epi8(v); }
  Vectorized8(T val1, T val2, T val3, T val4,
         T val5, T val6, T val7, T val8,
         T val9, T val10, T val11, T val12,
         T val13, T val14, T val15, T val16,
         T val17, T val18, T val19, T val20,
         T val21, T val22, T val23, T val24,
         T val25, T val26, T val27, T val28,
         T val29, T val30, T val31, T val32) {
    values = _mm256_setr_epi8(val1, val2, val3, val4, val5, val6, val7, val8,
                              val9, val10, val11, val12, val13, val14, val15, val16,
                              val17, val18, val19, val20, val21, val22, val23, val24,
                              val25, val26, val27, val28, val29, val30, val31, val32);
  }
  template <int64_t mask>
  static Vectorized<T> blend(Vectorized<T> a, Vectorized<T> b) {
    __at_align__ T tmp_values[size()];
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
  static Vectorized<T> blendv(const Vectorized<T>& a, const Vectorized<T>& b,
                               const Vectorized<T>& mask) {
    return _mm256_blendv_epi8(a.values, b.values, mask.values);
  }
  template <typename step_t>
  static Vectorized<T> arange(T base = 0, step_t step = static_cast<step_t>(1)) {
    return Vectorized<T>(
      base,             base +      step, base +  2 * step, base +  3 * step,
      base +  4 * step, base +  5 * step, base +  6 * step, base +  7 * step,
      base +  8 * step, base +  9 * step, base + 10 * step, base + 11 * step,
      base + 12 * step, base + 13 * step, base + 14 * step, base + 15 * step,
      base + 16 * step, base + 17 * step, base + 18 * step, base + 19 * step,
      base + 20 * step, base + 21 * step, base + 22 * step, base + 23 * step,
      base + 24 * step, base + 25 * step, base + 26 * step, base + 27 * step,
      base + 28 * step, base + 29 * step, base + 30 * step, base + 31 * step);
  }
  static Vectorized<T>
  set(Vectorized<T> a, Vectorized<T> b, T count = size()) {
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
  static Vectorized<T> loadu(const void* ptr) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
  }
  static Vectorized<T> loadu_one_fourth(const void* ptr) {
      // Fast path if only load element number of 8.
      // Note: We didn't merge it as fast path of loadu(const void* ptr, T count),
      // Because loadu(const void* ptr, T count) requires zero initialization for upper 128 bits.
      // However, by using _mm256_castsi128_si256, the upper 128 bits of the result are undefined.
      // TODO<leslie> We can use _mm256_zextsi128_si256 in the furture,
      // since gcc 9.3 doesn't support it now.
      __m128i input_128 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr));
      return _mm256_castsi128_si256(input_128);
  }
  static Vectorized<T> loadu(const void* ptr, T count) {
    __at_align__ T tmp_values[size()];
    // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
    // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
    // instructions while a loop would be compiled to one instruction.
    for (const auto i : c10::irange(size())) {
      tmp_values[i] = 0;
    }
    std::memcpy(tmp_values, ptr, count * sizeof(T));
    return loadu(tmp_values);
  }
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      // ptr need not to be aligned here. See
      // https://software.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-intel-advanced-vector-extensions/intrinsics-for-load-and-store-operations-1/mm256-storeu-si256.html
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
    } else if (count > 0) {
      if (count == 8) {
        // Fast path if only store element number of 8
        _mm_storel_epi64(reinterpret_cast<__m128i*>(ptr), _mm256_castsi256_si128(values));
      } else {
        __at_align__ T tmp_values[size()];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp_values), values);
        std::memcpy(ptr, tmp_values, count * sizeof(T));
      }
    }
  }
  const T& operator[](int idx) const  = delete;
  T& operator[](int idx)  = delete;
  Vectorized<T> real() const {
    return *this;
  }
  Vectorized<T> imag() const {
    return _mm256_set1_epi8(0);
  }
  Vectorized<T> conj() const {
    return *this;
  }
};

template<>
class Vectorized<int8_t>: public Vectorized8<int8_t> {
public:
  using Vectorized8::Vectorized8;

  Vectorized<int8_t> neg() const;

  Vectorized<int8_t> abs() const {
   return _mm256_abs_epi8(values);
  }

  Vectorized<int8_t> operator==(const Vectorized<int8_t>& other) const {
    return _mm256_cmpeq_epi8(values, other.values);
  }
  Vectorized<int8_t> operator!=(const Vectorized<int8_t>& other) const {
    return invert(_mm256_cmpeq_epi8(values, other.values));
  }
  Vectorized<int8_t> operator<(const Vectorized<int8_t>& other) const {
    return _mm256_cmpgt_epi8(other.values, values);
  }
  Vectorized<int8_t> operator<=(const Vectorized<int8_t>& other) const {
    return invert(_mm256_cmpgt_epi8(values, other.values));
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

template<>
class Vectorized<uint8_t>: public Vectorized8<uint8_t> {
public:
  using Vectorized8::Vectorized8;

  Vectorized<uint8_t> neg() const;

  Vectorized<uint8_t> abs() const {
    return *this;
  }

  Vectorized<uint8_t> operator==(const Vectorized<uint8_t>& other) const {
    return _mm256_cmpeq_epi8(values, other.values);
  }
  Vectorized<uint8_t> operator!=(const Vectorized<uint8_t>& other) const {
    return invert(_mm256_cmpeq_epi8(values, other.values));
  }
  Vectorized<uint8_t> operator<(const Vectorized<uint8_t>& other) const {
    __m256i max = _mm256_max_epu8(values, other.values);
    return invert(_mm256_cmpeq_epi8(max, values));
  }
  Vectorized<uint8_t> operator<=(const Vectorized<uint8_t>& other) const {
    __m256i max = _mm256_max_epu8(values, other.values);
    return _mm256_cmpeq_epi8(max, other.values);
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
Vectorized<int64_t> inline operator+(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  return _mm256_add_epi64(a, b);
}

template <>
Vectorized<int32_t> inline operator+(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
  return _mm256_add_epi32(a, b);
}

template <>
Vectorized<int16_t> inline operator+(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
  return _mm256_add_epi16(a, b);
}

template <>
Vectorized<int8_t> inline operator+(const Vectorized<int8_t>& a, const Vectorized<int8_t>& b) {
  return _mm256_add_epi8(a, b);
}

template <>
Vectorized<uint8_t> inline operator+(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
  return _mm256_add_epi8(a, b);
}

template <>
Vectorized<int64_t> inline operator-(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  return _mm256_sub_epi64(a, b);
}

template <>
Vectorized<int32_t> inline operator-(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
  return _mm256_sub_epi32(a, b);
}

template <>
Vectorized<int16_t> inline operator-(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
  return _mm256_sub_epi16(a, b);
}

template <>
Vectorized<int8_t> inline operator-(const Vectorized<int8_t>& a, const Vectorized<int8_t>& b) {
  return _mm256_sub_epi8(a, b);
}

template <>
Vectorized<uint8_t> inline operator-(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
  return _mm256_sub_epi8(a, b);
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

// Emulate operations with no native 64-bit support in avx,
// by extracting each element, performing the operation pointwise,
// then combining the results into a vector.
template <typename op_t>
Vectorized<int64_t> inline emulate(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b, const op_t& op) {
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
Vectorized<int64_t> inline emulate(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b, const Vectorized<int64_t>& c, const op_t& op) {
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
Vectorized<int64_t> inline operator*(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  return emulate(a, b, [](int64_t a_point, int64_t b_point) __ubsan_ignore_undefined__ {return a_point * b_point;});
}

template <>
Vectorized<int32_t> inline operator*(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
  return _mm256_mullo_epi32(a, b);
}

template <>
Vectorized<int16_t> inline operator*(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
  return _mm256_mullo_epi16(a, b);
}

template <typename T, typename Op>
Vectorized<T> inline int_elementwise_binary_256(const Vectorized<T>& a, const Vectorized<T>& b, Op op) {
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
#ifndef CPU_CAPABILITY_AVX2
  return int_elementwise_binary_256(a, b, std::multiplies<int8_t>());
#else
  __m256i mask00FF = _mm256_set1_epi16(0x00FF);
  __m256i a_lo = _mm256_srai_epi16(_mm256_slli_epi16(a, 8), 8);
  __m256i b_lo = _mm256_srai_epi16(_mm256_slli_epi16(b, 8), 8);
  __m256i a_hi = _mm256_srai_epi16(a, 8);
  __m256i b_hi = _mm256_srai_epi16(b, 8);
  __m256i res_lo = _mm256_and_si256(_mm256_mullo_epi16(a_lo, b_lo), mask00FF);
  __m256i res_hi = _mm256_slli_epi16(_mm256_mullo_epi16(a_hi, b_hi), 8);
  __m256i res = _mm256_or_si256(res_hi, res_lo);
  return res;
#endif
}

template <>
Vectorized<uint8_t> inline operator*(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
  // We don't have an instruction for multiplying uint8_t
#ifndef CPU_CAPABILITY_AVX2
  return int_elementwise_binary_256(a, b, std::multiplies<uint8_t>());
#else
  __m256i mask00FF = _mm256_set1_epi16(0x00FF);
  __m256i a_lo = _mm256_and_si256 (a, mask00FF);
  __m256i b_lo = _mm256_and_si256 (b, mask00FF);
  __m256i a_hi = _mm256_srli_epi16(a, 8);
  __m256i b_hi = _mm256_srli_epi16(b, 8);
  __m256i res_lo = _mm256_and_si256(_mm256_mullo_epi16(a_lo, b_lo), mask00FF);
  __m256i res_hi = _mm256_slli_epi16(_mm256_mullo_epi16(a_hi, b_hi), 8);
  __m256i res = _mm256_or_si256(res_hi, res_lo);
  return res;
#endif
}

template <>
Vectorized<int64_t> inline minimum(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
#ifndef CPU_CAPABILITY_AVX2
  return emulate(a, b, [](int64_t a_point, int64_t b_point) {return std::min(a_point, b_point);});
#else
  __m256i cmp = _mm256_cmpgt_epi64(a, b);
  return _mm256_blendv_epi8(a, b, cmp);
#endif
}

template <>
Vectorized<int32_t> inline minimum(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
  return _mm256_min_epi32(a, b);
}

template <>
Vectorized<int16_t> inline minimum(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
  return _mm256_min_epi16(a, b);
}

template <>
Vectorized<int8_t> inline minimum(const Vectorized<int8_t>& a, const Vectorized<int8_t>& b) {
  return _mm256_min_epi8(a, b);
}

template <>
Vectorized<uint8_t> inline minimum(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
  return _mm256_min_epu8(a, b);
}

template <>
Vectorized<int64_t> inline maximum(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
#ifndef CPU_CAPABILITY_AVX2
  return emulate(a, b, [](int64_t a_point, int64_t b_point) {return std::max(a_point, b_point);});
#else
  __m256i cmp = _mm256_cmpgt_epi64(a, b);
  return _mm256_blendv_epi8(b, a, cmp);
#endif
}

template <>
Vectorized<int32_t> inline maximum(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
  return _mm256_max_epi32(a, b);
}

template <>
Vectorized<int16_t> inline maximum(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
  return _mm256_max_epi16(a, b);
}

template <>
Vectorized<int8_t> inline maximum(const Vectorized<int8_t>& a, const Vectorized<int8_t>& b) {
  return _mm256_max_epi8(a, b);
}

template <>
Vectorized<uint8_t> inline maximum(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
  return _mm256_max_epu8(a, b);
}

template <>
Vectorized<int64_t> inline clamp(const Vectorized<int64_t>& a, const Vectorized<int64_t>& min_val, const Vectorized<int64_t>& max_val) {
#ifndef CPU_CAPABILITY_AVX2
  return emulate(a, min_val, max_val, [](int64_t a_point, int64_t min_point, int64_t max_point) {return std::min(max_point, std::max(a_point, min_point));});
#else
  return minimum(maximum(a, min_val), max_val);
#endif
}

template <>
Vectorized<int32_t> inline clamp(const Vectorized<int32_t>& a, const Vectorized<int32_t>& min_val, const Vectorized<int32_t>& max_val) {
  return _mm256_min_epi32(max_val, _mm256_max_epi32(a, min_val));
}

template <>
Vectorized<int16_t> inline clamp(const Vectorized<int16_t>& a, const Vectorized<int16_t>& min_val, const Vectorized<int16_t>& max_val) {
  return _mm256_min_epi16(max_val, _mm256_max_epi16(a, min_val));
}

template <>
Vectorized<int8_t> inline clamp(const Vectorized<int8_t>& a, const Vectorized<int8_t>& min_val, const Vectorized<int8_t>& max_val) {
  return _mm256_min_epi8(max_val, _mm256_max_epi8(a, min_val));
}

template <>
Vectorized<uint8_t> inline clamp(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& min_val, const Vectorized<uint8_t>& max_val) {
  return _mm256_min_epu8(max_val, _mm256_max_epu8(a, min_val));
}

template <>
Vectorized<int64_t> inline clamp_max(const Vectorized<int64_t>& a, const Vectorized<int64_t>& max_val) {
#ifndef CPU_CAPABILITY_AVX2
  return emulate(a, max_val, [](int64_t a_point, int64_t max_point) {return std::min(max_point, a_point);});
#else
  return minimum(max_val, a);
#endif
}

template <>
Vectorized<int32_t> inline clamp_max(const Vectorized<int32_t>& a, const Vectorized<int32_t>& max_val) {
  return _mm256_min_epi32(max_val, a);
}

template <>
Vectorized<int16_t> inline clamp_max(const Vectorized<int16_t>& a, const Vectorized<int16_t>& max_val) {
  return _mm256_min_epi16(max_val, a);
}

template <>
Vectorized<int8_t> inline clamp_max(const Vectorized<int8_t>& a, const Vectorized<int8_t>& max_val) {
  return _mm256_min_epi8(max_val, a);
}

template <>
Vectorized<uint8_t> inline clamp_max(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& max_val) {
  return _mm256_min_epu8(max_val, a);
}

template <>
Vectorized<int64_t> inline clamp_min(const Vectorized<int64_t>& a, const Vectorized<int64_t>& min_val) {
#ifndef CPU_CAPABILITY_AVX2
  return emulate(a, min_val, [](int64_t a_point, int64_t min_point) {return std::max(min_point, a_point);});
#else
  return maximum(min_val, a);
#endif
}

template <>
Vectorized<int32_t> inline clamp_min(const Vectorized<int32_t>& a, const Vectorized<int32_t>& min_val) {
  return _mm256_max_epi32(min_val, a);
}

template <>
Vectorized<int16_t> inline clamp_min(const Vectorized<int16_t>& a, const Vectorized<int16_t>& min_val) {
  return _mm256_max_epi16(min_val, a);
}

template <>
Vectorized<int8_t> inline clamp_min(const Vectorized<int8_t>& a, const Vectorized<int8_t>& min_val) {
  return _mm256_max_epi8(min_val, a);
}

template <>
Vectorized<uint8_t> inline clamp_min(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& min_val) {
  return _mm256_max_epu8(min_val, a);
}

template<typename T>
std::enable_if_t<!(std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>), Vectorized<int32_t>>
inline convert_to_int32(const T* ptr, int count=Vectorized<int32_t>::size()) {
  return Vectorized<int32_t>::loadu(ptr, count);
}

template<typename T>
std::enable_if_t<std::is_same_v<T, int8_t>, Vectorized<int32_t>>
inline convert_to_int32(const int8_t* ptr, int count=Vectorized<int32_t>::size()) {
  if (count == Vectorized<int32_t>::size()) {
    return _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr)));
  } else {
    auto a = Vectorized<int8_t>::loadu(ptr, count);
    return _mm256_cvtepi8_epi32(_mm256_castsi256_si128(a));
  }
}

template<typename T>
std::enable_if_t<std::is_same_v<T, uint8_t>, Vectorized<int32_t>>
inline convert_to_int32(const uint8_t* ptr, int count=Vectorized<int32_t>::size()) {
  if (count == Vectorized<int32_t>::size()) {
    return _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr)));
  } else {
    auto a = Vectorized<uint8_t>::loadu(ptr, count);
    return _mm256_cvtepu8_epi32(_mm256_castsi256_si128(a));
  }
}

template <>
Vectorized<int64_t> inline operator/(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  return int_elementwise_binary_256(a, b, std::divides<int64_t>());
}
template <>
Vectorized<int32_t> inline operator/(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
  return int_elementwise_binary_256(a, b, std::divides<int32_t>());
}
template <>
Vectorized<int16_t> inline operator/(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
  return int_elementwise_binary_256(a, b, std::divides<int16_t>());
}
template <>
Vectorized<int8_t> inline operator/(const Vectorized<int8_t>& a, const Vectorized<int8_t>& b) {
  return int_elementwise_binary_256(a, b, std::divides<int8_t>());
}
template <>
Vectorized<uint8_t> inline operator/(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
  return int_elementwise_binary_256(a, b, std::divides<uint8_t>());
}

template<class T, typename std::enable_if_t<std::is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
inline Vectorized<T> operator&(const Vectorized<T>& a, const Vectorized<T>& b) {
  return _mm256_and_si256(a, b);
}
template<class T, typename std::enable_if_t<std::is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
inline Vectorized<T> operator|(const Vectorized<T>& a, const Vectorized<T>& b) {
  return _mm256_or_si256(a, b);
}
template<class T, typename std::enable_if_t<std::is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
inline Vectorized<T> operator^(const Vectorized<T>& a, const Vectorized<T>& b) {
  return _mm256_xor_si256(a, b);
}
template<class T, typename std::enable_if_t<std::is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
inline Vectorized<T> operator~(const Vectorized<T>& a) {
  return _mm256_xor_si256(a, _mm256_set1_epi32(-1));
}

inline Vectorized<int64_t> Vectorized<int64_t>::eq(const Vectorized<int64_t>& other) const {
  return (*this == other) & Vectorized<int64_t>(1);
}

inline Vectorized<int64_t> Vectorized<int64_t>::ne(const Vectorized<int64_t>& other) const {
  return (*this != other) & Vectorized<int64_t>(1);
}

inline Vectorized<int64_t> Vectorized<int64_t>::gt(const Vectorized<int64_t>& other) const {
  return (*this > other) & Vectorized<int64_t>(1);
}

inline Vectorized<int64_t> Vectorized<int64_t>::ge(const Vectorized<int64_t>& other) const {
  return (*this >= other) & Vectorized<int64_t>(1);
}

inline Vectorized<int64_t> Vectorized<int64_t>::lt(const Vectorized<int64_t>& other) const {
  return (*this < other) & Vectorized<int64_t>(1);
}

inline Vectorized<int64_t> Vectorized<int64_t>::le(const Vectorized<int64_t>& other) const {
  return (*this <= other) & Vectorized<int64_t>(1);
}

inline Vectorized<int32_t> Vectorized<int32_t>::eq(const Vectorized<int32_t>& other) const {
  return (*this == other) & Vectorized<int32_t>(1);
}

inline Vectorized<int32_t> Vectorized<int32_t>::ne(const Vectorized<int32_t>& other) const {
  return (*this != other) & Vectorized<int32_t>(1);
}

inline Vectorized<int32_t> Vectorized<int32_t>::gt(const Vectorized<int32_t>& other) const {
  return (*this > other) & Vectorized<int32_t>(1);
}

inline Vectorized<int32_t> Vectorized<int32_t>::ge(const Vectorized<int32_t>& other) const {
  return (*this >= other) & Vectorized<int32_t>(1);
}

inline Vectorized<int32_t> Vectorized<int32_t>::lt(const Vectorized<int32_t>& other) const {
  return (*this < other) & Vectorized<int32_t>(1);
}

inline Vectorized<int32_t> Vectorized<int32_t>::le(const Vectorized<int32_t>& other) const {
  return (*this <= other) & Vectorized<int32_t>(1);
}

inline Vectorized<int16_t> Vectorized<int16_t>::eq(const Vectorized<int16_t>& other) const {
  return (*this == other) & Vectorized<int16_t>(1);
}

inline Vectorized<int16_t> Vectorized<int16_t>::ne(const Vectorized<int16_t>& other) const {
  return (*this != other) & Vectorized<int16_t>(1);
}

inline Vectorized<int16_t> Vectorized<int16_t>::gt(const Vectorized<int16_t>& other) const {
  return (*this > other) & Vectorized<int16_t>(1);
}

inline Vectorized<int16_t> Vectorized<int16_t>::ge(const Vectorized<int16_t>& other) const {
  return (*this >= other) & Vectorized<int16_t>(1);
}

inline Vectorized<int16_t> Vectorized<int16_t>::lt(const Vectorized<int16_t>& other) const {
  return (*this < other) & Vectorized<int16_t>(1);
}

inline Vectorized<int16_t> Vectorized<int16_t>::le(const Vectorized<int16_t>& other) const {
  return (*this <= other) & Vectorized<int16_t>(1);
}

inline Vectorized<int8_t> Vectorized<int8_t>::eq(const Vectorized<int8_t>& other) const {
  return (*this == other) & Vectorized<int8_t>(1);
}

inline Vectorized<int8_t> Vectorized<int8_t>::ne(const Vectorized<int8_t>& other) const {
  return (*this != other) & Vectorized<int8_t>(1);
}

inline Vectorized<int8_t> Vectorized<int8_t>::gt(const Vectorized<int8_t>& other) const {
  return (*this > other) & Vectorized<int8_t>(1);
}

inline Vectorized<int8_t> Vectorized<int8_t>::ge(const Vectorized<int8_t>& other) const {
  return (*this >= other) & Vectorized<int8_t>(1);
}

inline Vectorized<int8_t> Vectorized<int8_t>::lt(const Vectorized<int8_t>& other) const {
  return (*this < other) & Vectorized<int8_t>(1);
}

inline Vectorized<int8_t> Vectorized<int8_t>::le(const Vectorized<int8_t>& other) const {
  return (*this <= other) & Vectorized<int8_t>(1);
}

inline Vectorized<uint8_t> Vectorized<uint8_t>::eq(const Vectorized<uint8_t>& other) const {
  return (*this == other) & Vectorized<uint8_t>(1);
}

inline Vectorized<uint8_t> Vectorized<uint8_t>::ne(const Vectorized<uint8_t>& other) const {
  return (*this != other) & Vectorized<uint8_t>(1);
}

inline Vectorized<uint8_t> Vectorized<uint8_t>::gt(const Vectorized<uint8_t>& other) const {
  return (*this > other) & Vectorized<uint8_t>(1);
}

inline Vectorized<uint8_t> Vectorized<uint8_t>::ge(const Vectorized<uint8_t>& other) const {
  return (*this >= other) & Vectorized<uint8_t>(1);
}

inline Vectorized<uint8_t> Vectorized<uint8_t>::lt(const Vectorized<uint8_t>& other) const {
  return (*this < other) & Vectorized<uint8_t>(1);
}

inline Vectorized<uint8_t> Vectorized<uint8_t>::le(const Vectorized<uint8_t>& other) const {
  return (*this <= other) & Vectorized<uint8_t>(1);
}

template <bool left_shift>
Vectorized<int16_t> inline shift_256_16(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
  // No vector instruction for shifting int16_t, so emulating it instead.

  // Control masks for shuffle operation, treating 256 bits as an
  // array of 16-bit elements, and considering pairs of neighboring
  // elements.  Specifially, a mask named "ctl_M_N" (M,N in [0,1], and
  // M!=N) is set so that shuffle will move element with index M from
  // input pair into element with index N in output pair, and element
  // with index M in output pair will be set to all 0s.
  __m256i ctl_0_1 = _mm256_set_epi8(29, 28, 0x80, 0x80, 25, 24, 0x80, 0x80,
                                    21, 20, 0x80, 0x80, 17, 16, 0x80, 0x80,
                                    13, 12, 0x80, 0x80, 9, 8, 0x80, 0x80,
                                    5, 4, 0x80, 0x80, 1, 0, 0x80, 0x80);
  __m256i ctl_1_0 = _mm256_set_epi8(0x80, 0x80, 31, 30, 0x80, 0x80, 27, 26,
                                    0x80, 0x80, 23, 22, 0x80, 0x80, 19, 18,
                                    0x80, 0x80, 15, 14, 0x80, 0x80, 11, 10,
                                    0x80, 0x80, 7, 6, 0x80, 0x80, 3, 2);

  // Masks for bitwise and operation, treating 256 bits as an array of
  // 16-bit elements, and considering them in pairs of neighboring
  // elements.  A mask named "keep_M" (M in [0,1]) is set so that
  // bitwise and will copy element with index M from input pair into
  // element with the same index in output pair, while the other
  // element in output pair will be set to all 0s.
  __m256i keep_0 = _mm256_set1_epi32(0xFFFF);
  __m256i keep_1 = _mm256_set1_epi32(0xFFFF0000);

  // Take each 16-bit element with idx%2==0 from input array to be
  // shifted and extend it to 32 bits so that 0s are added to the
  // right.  Then, perform shifting on this 32-bit number.  Upper 16
  // bits will be proper result of shifting original 16-bit number, so
  // write them to result array, into the same position from which
  // corresponding input element is taken.  Also, make sure that
  // result array elements with idx%2!=0 are set to all 0s.
  //
  // Note that number of bits to shift for is extended to 32 bits by
  // adding 0s to the left.  That means this number is not properly
  // sign-extended for negative values.  However, number of bits to
  // shift is treated as an unsigned integer by respective shift
  // intrinsics anyway so if negative then either with or without
  // proper sign extension, it will be interpreted as a number greater
  // than 32, and the shifting result will be the same.
  __m256i a0 = _mm256_shuffle_epi8(a, ctl_0_1);
  __m256i b0 = _mm256_and_si256(b, keep_0);
  __m256i c0;
  if (left_shift)
    c0 = _mm256_sllv_epi32(a0, b0);
  else
    c0 = _mm256_srav_epi32(a0, b0);
  c0 = _mm256_shuffle_epi8(c0, ctl_1_0);

  // Peform shifting the same way for input array elements with
  // idx%2==1.
  __m256i a1 = _mm256_and_si256(a, keep_1);
  __m256i b1 = _mm256_shuffle_epi8(b, ctl_1_0);
  __m256i c1;
  if (left_shift)
    c1 = _mm256_sllv_epi32(a1, b1);
  else
    c1 = _mm256_srav_epi32(a1, b1);
  c1 = _mm256_and_si256(c1, keep_1);

  // Merge partial results into the final result.
  __m256i c = _mm256_or_si256(c0, c1);

  return c;
}

template <bool left_shift, typename T, typename std::enable_if_t<std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>, int> = 0>
Vectorized<T> inline shift_256_8(const Vectorized<T>& a, const Vectorized<T>& b) {
  // No vector instruction for shifting int8_t/uint8_t, so emulating
  // it instead.

  // Control masks for shuffle operation, treating 256 bits as an
  // array of 8-bit elements, and considering quadruples of
  // neighboring elements.  Specifially, a mask named "ctl_M_N" (M,N
  // in [0,1,2,3], and M!=N) is set so that shuffle will move element
  // with index M from input quadruple into element with index N in
  // output quadruple, and other elements in output quadruple will be
  // set to all 0s.
  __m256i ctl_0_3 = _mm256_set_epi8(28, 0x80, 0x80, 0x80, 24, 0x80, 0x80, 0x80,
                                    20, 0x80, 0x80, 0x80, 16, 0x80, 0x80, 0x80,
                                    12, 0x80, 0x80, 0x80, 8, 0x80, 0x80, 0x80,
                                    4, 0x80, 0x80, 0x80, 0, 0x80, 0x80, 0x80);
  __m256i ctl_1_0 = _mm256_set_epi8(0x80, 0x80, 0x80, 29, 0x80, 0x80, 0x80, 25,
                                    0x80, 0x80, 0x80, 21, 0x80, 0x80, 0x80, 17,
                                    0x80, 0x80, 0x80, 13, 0x80, 0x80, 0x80, 9,
                                    0x80, 0x80, 0x80, 5, 0x80, 0x80, 0x80, 1);
  __m256i ctl_1_3 = _mm256_set_epi8(29, 0x80, 0x80, 0x80, 25, 0x80, 0x80, 0x80,
                                    21, 0x80, 0x80, 0x80, 17, 0x80, 0x80, 0x80,
                                    13, 0x80, 0x80, 0x80, 9, 0x80, 0x80, 0x80,
                                    5, 0x80, 0x80, 0x80, 1, 0x80, 0x80, 0x80);
  __m256i ctl_2_0 = _mm256_set_epi8(0x80, 0x80, 0x80, 30, 0x80, 0x80, 0x80, 26,
                                    0x80, 0x80, 0x80, 22, 0x80, 0x80, 0x80, 18,
                                    0x80, 0x80, 0x80, 14, 0x80, 0x80, 0x80, 10,
                                    0x80, 0x80, 0x80, 6, 0x80, 0x80, 0x80, 2);
  __m256i ctl_2_3 = _mm256_set_epi8(30, 0x80, 0x80, 0x80, 26, 0x80, 0x80, 0x80,
                                    22, 0x80, 0x80, 0x80, 18, 0x80, 0x80, 0x80,
                                    14, 0x80, 0x80, 0x80, 10, 0x80, 0x80, 0x80,
                                    6, 0x80, 0x80, 0x80, 2, 0x80, 0x80, 0x80);
  __m256i ctl_3_0 = _mm256_set_epi8(0x80, 0x80, 0x80, 31, 0x80, 0x80, 0x80, 27,
                                    0x80, 0x80, 0x80, 23, 0x80, 0x80, 0x80, 19,
                                    0x80, 0x80, 0x80, 15, 0x80, 0x80, 0x80, 11,
                                    0x80, 0x80, 0x80, 7, 0x80, 0x80, 0x80, 3);
  __m256i ctl_3_1 = _mm256_set_epi8(0x80, 0x80, 31, 0x80, 0x80, 0x80, 27, 0x80,
                                    0x80, 0x80, 23, 0x80, 0x80, 0x80, 19, 0x80,
                                    0x80, 0x80, 15, 0x80, 0x80, 0x80, 11, 0x80,
                                    0x80, 0x80, 7, 0x80, 0x80, 0x80, 3, 0x80);
  __m256i ctl_3_2 = _mm256_set_epi8(0x80, 31, 0x80, 0x80, 0x80, 27, 0x80, 0x80,
                                    0x80, 23, 0x80, 0x80, 0x80, 19, 0x80, 0x80,
                                    0x80, 15, 0x80, 0x80, 0x80, 11, 0x80, 0x80,
                                    0x80, 7, 0x80, 0x80, 0x80, 3, 0x80, 0x80);

  // Masks for bitwise and operation, treating 256 bits as an array of
  // 8-bit elements, and considering them in quadruples of neighboring
  // elements.  A mask named "keep_M" (M in [0,1,2,3]) is set so that
  // bitwise and will copy element with index M from input quadruple
  // into element with the same index in output quadruple, while the
  // other elements in output quadruple will be set to all 0s.
  __m256i keep_0 = _mm256_set1_epi32(0xFF);
  __m256i keep_3 = _mm256_set1_epi32(0xFF000000);

  // Take each 8-bit element with idx%4==0 from input array to be
  // shifted and extend it to 32 bits so that 0s are added to the
  // right.  Then, perform shifting on this 32-bit number.  Upper 8
  // bits will be proper result of shifting original 8-bit number, so
  // write them to result array, into the same position from which
  // corresponding input element is taken.  Also, make sure that
  // result array elements with idx%4!=0 are set to all 0s.
  //
  // Note that number of bits to shift for is extended to 32 bits by
  // adding 0s to the left.  That means this number is not properly
  // sign-extended for negative values.  However, number of bits to
  // shift is treated as an unsigned integer by respective shift
  // intrinsics anyway so if negative then either with or without
  // proper sign extension, it will be interpreted as a number greater
  // than 32, and the shifting result will be the same.
  __m256i a0 = _mm256_shuffle_epi8(a, ctl_0_3);
  __m256i b0 = _mm256_and_si256(b, keep_0);
  __m256i c0;
  if (left_shift)
    c0 = _mm256_sllv_epi32(a0, b0);
  else
    if constexpr (std::is_same_v<T, int8_t>)
      c0 = _mm256_srav_epi32(a0, b0);
    else
      c0 = _mm256_srlv_epi32(a0, b0);
  c0 = _mm256_shuffle_epi8(c0, ctl_3_0);

  // Peform shifting the same way for input array elements with
  // idx%4==1.
  __m256i a1 = _mm256_shuffle_epi8(a, ctl_1_3);
  __m256i b1 = _mm256_shuffle_epi8(b, ctl_1_0);
  __m256i c1;
  if (left_shift)
    c1 = _mm256_sllv_epi32(a1, b1);
  else
    if constexpr (std::is_same_v<T, int8_t>)
      c1 = _mm256_srav_epi32(a1, b1);
    else
      c1 = _mm256_srlv_epi32(a1, b1);
  c1 = _mm256_shuffle_epi8(c1, ctl_3_1);

  // Peform shifting the same way for input array elements with
  // idx%4==2.
  __m256i a2 = _mm256_shuffle_epi8(a, ctl_2_3);
  __m256i b2 = _mm256_shuffle_epi8(b, ctl_2_0);
  __m256i c2;
  if (left_shift)
    c2 = _mm256_sllv_epi32(a2, b2);
  else
    if constexpr (std::is_same_v<T, int8_t>)
      c2 = _mm256_srav_epi32(a2, b2);
    else
      c2 = _mm256_srlv_epi32(a2, b2);
  c2 = _mm256_shuffle_epi8(c2, ctl_3_2);

  // Peform shifting the same way for input array elements with
  // idx%4==3.
  __m256i a3 =  _mm256_and_si256(a, keep_3);
  __m256i b3 = _mm256_shuffle_epi8(b, ctl_3_0);
  __m256i c3;
  if (left_shift)
    c3 = _mm256_sllv_epi32(a3, b3);
  else
    if constexpr (std::is_same_v<T, int8_t>)
      c3 = _mm256_srav_epi32(a3, b3);
    else
      c3 = _mm256_srlv_epi32(a3, b3);
  c3 = _mm256_and_si256(c3, keep_3);

  // Merge partial results into the final result.
  __m256i c01 = _mm256_or_si256(c0, c1);
  __m256i c23 = _mm256_or_si256(c2, c3);
  __m256i c = _mm256_or_si256(c01, c23);

  return c;
}

template <>
Vectorized<int64_t> inline operator<<(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  return _mm256_sllv_epi64(a, b);
}

template <>
Vectorized<int32_t> inline operator<<(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
  return _mm256_sllv_epi32(a, b);
}

template <>
Vectorized<int16_t> inline operator<<(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
  return shift_256_16<true>(a, b);
}

template <>
Vectorized<int8_t> inline operator<<(const Vectorized<int8_t>& a, const Vectorized<int8_t>& b) {
  return shift_256_8<true>(a, b);
}

template <>
Vectorized<uint8_t> inline operator<<(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
  return shift_256_8<true>(a, b);
}

template <>
Vectorized<int64_t> inline operator>>(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
  // No vector instruction for right arithmetic shifting int64_t, so emulating it
  // instead.

  // Clamp the shift values such that shift values < 0 and > 64 are changed to 64
  // which results in -1 for negative input and 0 for non-negative input.
  __m256i zero = _mm256_set1_epi64x(0);
  __m256i max_shift = _mm256_set1_epi64x(64);
  __m256i mask = _mm256_or_si256(_mm256_cmpgt_epi64(zero, b), _mm256_cmpgt_epi64(b, max_shift));
  __m256i shift = _mm256_blendv_epi8(b, max_shift, mask);
  // Shift the number logically to the right, thus filling the most
  // significant bits with 0s.  Then, replace these bits with the sign
  // bit.
  __m256i sign_bits = _mm256_cmpgt_epi64(zero, a);
  __m256i sign_shift = _mm256_sub_epi64(max_shift, shift);
  __m256i sign_ext = _mm256_sllv_epi64(sign_bits, sign_shift);
  __m256i c = _mm256_srlv_epi64(a, shift);
  c = _mm256_or_si256(c, sign_ext);

  return c;
}

template <>
Vectorized<int32_t> inline operator>>(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
  return _mm256_srav_epi32(a, b);
}

template <>
Vectorized<int16_t> inline operator>>(const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) {
  return shift_256_16<false>(a, b);
}

template <>
Vectorized<int8_t> inline operator>>(const Vectorized<int8_t>& a, const Vectorized<int8_t>& b) {
  return shift_256_8<false>(a, b);
}

template <>
Vectorized<uint8_t> inline operator>>(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
  return shift_256_8<false>(a, b);
}

#endif

}} // namespace at::vec::CPU_CAPABILITY
