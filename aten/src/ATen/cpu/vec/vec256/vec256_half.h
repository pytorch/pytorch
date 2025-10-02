#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/vec256/vec256_16bit_float.h>
#include <c10/util/irange.h>

namespace at::vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

#ifdef CPU_CAPABILITY_AVX2

template <>
struct is_vec_specialized_for<Half> : std::bool_constant<true> {};

template <>
class Vectorized<Half> : public Vectorized16<Half> {
 public:
  using Vectorized16::Vectorized16;

  using value_type = Half;

  Vectorized<Half> frac() const;

  Vectorized<Half> eq(const Vectorized<Half>& other) const;
  Vectorized<Half> ne(const Vectorized<Half>& other) const;
  Vectorized<Half> gt(const Vectorized<Half>& other) const;
  Vectorized<Half> ge(const Vectorized<Half>& other) const;
  Vectorized<Half> lt(const Vectorized<Half>& other) const;
  Vectorized<Half> le(const Vectorized<Half>& other) const;
};

Vectorized<Half> inline operator+(
    const Vectorized<Half>& a,
    const Vectorized<Half>& b) {
  return binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) {
    return _mm256_add_ps(x, y);
  });
}
Vectorized<Half> inline operator-(
    const Vectorized<Half>& a,
    const Vectorized<Half>& b) {
  return binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) {
    return _mm256_sub_ps(x, y);
  });
}
Vectorized<Half> inline operator*(
    const Vectorized<Half>& a,
    const Vectorized<Half>& b) {
  return binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) {
    return _mm256_mul_ps(x, y);
  });
}
Vectorized<Half> inline operator/(
    const Vectorized<Half>& a,
    const Vectorized<Half>& b) {
  return binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) {
    return _mm256_div_ps(x, y);
  });
}
Vectorized<Half> inline operator&(
    const Vectorized<Half>& a,
    const Vectorized<Half>& b) {
  return _mm256_and_si256(a, b);
}
Vectorized<Half> inline operator|(
    const Vectorized<Half>& a,
    const Vectorized<Half>& b) {
  return _mm256_or_si256(a, b);
}
Vectorized<Half> inline operator^(
    const Vectorized<Half>& a,
    const Vectorized<Half>& b) {
  return _mm256_xor_si256(a, b);
}

inline Vectorized<Half> Vectorized<Half>::eq(
    const Vectorized<Half>& other) const {
  return (*this == other) & Vectorized<Half>(1.0f);
}
inline Vectorized<Half> Vectorized<Half>::ne(
    const Vectorized<Half>& other) const {
  return (*this != other) & Vectorized<Half>(1.0f);
}
inline Vectorized<Half> Vectorized<Half>::gt(
    const Vectorized<Half>& other) const {
  return (*this > other) & Vectorized<Half>(1.0f);
}
inline Vectorized<Half> Vectorized<Half>::ge(
    const Vectorized<Half>& other) const {
  return (*this >= other) & Vectorized<Half>(1.0f);
}
inline Vectorized<Half> Vectorized<Half>::lt(
    const Vectorized<Half>& other) const {
  return (*this < other) & Vectorized<Half>(1.0f);
}
inline Vectorized<Half> Vectorized<Half>::le(
    const Vectorized<Half>& other) const {
  return (*this <= other) & Vectorized<Half>(1.0f);
}

// frac. Implement this here so we can use subtraction
inline Vectorized<Half> Vectorized<Half>::frac() const {
  return *this - this->trunc();
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<Half> inline maximum(
    const Vectorized<Half>& a,
    const Vectorized<Half>& b) {
  __m256 a_lo, a_hi;
  __m256 b_lo, b_hi;
  cvtfp16_fp32(__m256i(a), a_lo, a_hi);
  cvtfp16_fp32(__m256i(b), b_lo, b_hi);
  auto max_lo = _mm256_max_ps(a_lo, b_lo);
  auto max_hi = _mm256_max_ps(a_hi, b_hi);
  auto nan_lo = _mm256_cmp_ps(a_lo, b_lo, _CMP_UNORD_Q);
  auto nan_hi = _mm256_cmp_ps(a_hi, b_hi, _CMP_UNORD_Q);
  // Exploit the fact that all-ones is a NaN.
  auto o1 = _mm256_or_ps(max_lo, nan_lo);
  auto o2 = _mm256_or_ps(max_hi, nan_hi);
  return cvtfp32_fp16(o1, o2);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<Half> inline minimum(
    const Vectorized<Half>& a,
    const Vectorized<Half>& b) {
  __m256 a_lo, a_hi;
  __m256 b_lo, b_hi;
  cvtfp16_fp32(__m256i(a), a_lo, a_hi);
  cvtfp16_fp32(__m256i(b), b_lo, b_hi);
  auto min_lo = _mm256_min_ps(a_lo, b_lo);
  auto min_hi = _mm256_min_ps(a_hi, b_hi);
  auto nan_lo = _mm256_cmp_ps(a_lo, b_lo, _CMP_UNORD_Q);
  auto nan_hi = _mm256_cmp_ps(a_hi, b_hi, _CMP_UNORD_Q);
  // Exploit the fact that all-ones is a NaN.
  auto o1 = _mm256_or_ps(min_lo, nan_lo);
  auto o2 = _mm256_or_ps(min_hi, nan_hi);
  return cvtfp32_fp16(o1, o2);
}

template <>
Vectorized<Half> inline clamp(
    const Vectorized<Half>& a,
    const Vectorized<Half>& min,
    const Vectorized<Half>& max) {
  __m256 a_lo, a_hi;
  __m256 min_lo, min_hi;
  __m256 max_lo, max_hi;
  cvtfp16_fp32(__m256i(a), a_lo, a_hi);
  cvtfp16_fp32(__m256i(min), min_lo, min_hi);
  cvtfp16_fp32(__m256i(max), max_lo, max_hi);
  auto o1 = _mm256_min_ps(max_lo, _mm256_max_ps(min_lo, a_lo));
  auto o2 = _mm256_min_ps(max_hi, _mm256_max_ps(min_hi, a_hi));
  return cvtfp32_fp16(o1, o2);
}

template <>
Vectorized<Half> inline clamp_max(
    const Vectorized<Half>& a,
    const Vectorized<Half>& max) {
  __m256 a_lo, a_hi;
  __m256 max_lo, max_hi;
  cvtfp16_fp32(__m256i(a), a_lo, a_hi);
  cvtfp16_fp32(__m256i(max), max_lo, max_hi);
  auto o1 = _mm256_min_ps(max_lo, a_lo);
  auto o2 = _mm256_min_ps(max_hi, a_hi);
  return cvtfp32_fp16(o1, o2);
}

template <>
Vectorized<Half> inline clamp_min(
    const Vectorized<Half>& a,
    const Vectorized<Half>& min) {
  __m256 a_lo, a_hi;
  __m256 min_lo, min_hi;
  cvtfp16_fp32(__m256i(a), a_lo, a_hi);
  cvtfp16_fp32(__m256i(min), min_lo, min_hi);
  auto o1 = _mm256_max_ps(min_lo, a_lo);
  auto o2 = _mm256_max_ps(min_hi, a_hi);
  return cvtfp32_fp16(o1, o2);
}

template <>
inline void convert(const Half* src, Half* dst, int64_t n) {
  int64_t i;
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (i = 0; i <= (n - Vectorized<Half>::size());
       i += Vectorized<Half>::size()) {
    auto vsrc =
        _mm256_loadu_si256(reinterpret_cast<__m256i*>((void*)(src + i)));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>((void*)(dst + i)), vsrc);
  }
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (; i < n; i++) {
    dst[i] = src[i];
  }
}

template <>
inline void convert(const float* src, Half* dst, int64_t n) {
  int64_t i;
  for (i = 0; i + Vectorized<Half>::size() <= n;
       i += Vectorized<Half>::size()) {
    __m256 a = _mm256_loadu_ps(&src[i]);
    __m256 b = _mm256_loadu_ps(&src[i + 8]);

    __m256i c = cvtfp32_fp16(a, b);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[i]), c);
  }
  for (; i < n; i++) {
    dst[i] = c10::convert<Half>(src[i]);
  }
}

template <>
inline void convert(const double* src, Half* dst, int64_t n) {
  auto load_float = [](const double* src) -> __m256 {
    // Load one float vector from an array of doubles
    __m128 a = _mm256_cvtpd_ps(_mm256_loadu_pd(src));
    __m128 b = _mm256_cvtpd_ps(_mm256_loadu_pd(src + 4));
    return _mm256_insertf128_ps(_mm256_castps128_ps256(a), b, 1);
  };

  int64_t i;
  for (i = 0; i + Vectorized<Half>::size() <= n;
       i += Vectorized<Half>::size()) {
    __m256 a = load_float(&src[i]);
    __m256 b = load_float(&src[i + 8]);

    __m256i c = cvtfp32_fp16(a, b);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[i]), c);
  }
  for (; i < n; i++) {
    dst[i] = c10::convert<Half>(src[i]);
  }
}

template <>
Vectorized<Half> inline fmadd(
    const Vectorized<Half>& a,
    const Vectorized<Half>& b,
    const Vectorized<Half>& c) {
  __m256 a_lo, a_hi;
  __m256 b_lo, b_hi;
  __m256 c_lo, c_hi;
  cvtfp16_fp32(__m256i(a), a_lo, a_hi);
  cvtfp16_fp32(__m256i(b), b_lo, b_hi);
  cvtfp16_fp32(__m256i(c), c_lo, c_hi);
  auto o1 = _mm256_fmadd_ps(a_lo, b_lo, c_lo);
  auto o2 = _mm256_fmadd_ps(a_hi, b_hi, c_hi);
  return cvtfp32_fp16(o1, o2);
}

CONVERT_VECTORIZED_INIT(Half, half)
LOAD_FP32_VECTORIZED_INIT(Half, fp16)

#else // defined(CPU_CAPABILITY_AVX2)

#if !defined(__aarch64__) || defined(CPU_CAPABILITY_SVE256)
CONVERT_NON_VECTORIZED_INIT(Half, half)
#endif

LOAD_FP32_NON_VECTORIZED_INIT(Half, fp16)
#endif // defined(CPU_CAPABILITY_AVX2)
} // namespace CPU_CAPABILITY
} // namespace at::vec
