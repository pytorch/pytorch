#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec256/vec256_16_base.h>
#include <c10/util/irange.h>

#if defined(CPU_CAPABILITY_AVX2)
#define SLEEF_STATIC_LIBS
#include <sleef.h>
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"

namespace at::vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX2)

#ifndef SLEEF_CONST
#if (defined(__GNUC__) || defined(__CLANG__)) && !defined(__INTEL_COMPILER)
#define SLEEF_CONST const
#else
#define SLEEF_CONST
#endif
#define SLEEF_CONST_OLD SLEEF_CONST
#else
#define SLEEF_CONST_OLD
#endif

template <>
class Vectorized<Half>: public Vectorized16<Half> {
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

Vectorized<Half> inline operator+(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  return binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) { return _mm256_add_ps(x, y); });
}
Vectorized<Half> inline operator-(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  return binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) { return _mm256_sub_ps(x, y); });
}
Vectorized<Half> inline operator*(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  return binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) { return _mm256_mul_ps(x, y); });
}
Vectorized<Half> inline operator/(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  return binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) { return _mm256_div_ps(x, y); });
}
Vectorized<Half> inline operator&(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  return _mm256_and_si256(a, b);
}
Vectorized<Half> inline operator|(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  return _mm256_or_si256(a, b);
}
Vectorized<Half> inline operator^(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  return _mm256_xor_si256(a, b);
}

inline Vectorized<Half> Vectorized<Half>::eq(const Vectorized<Half>& other) const {
  return (*this == other) & Vectorized<Half>(1.0f);
}
inline Vectorized<Half> Vectorized<Half>::ne(const Vectorized<Half>& other) const {
  return (*this != other) & Vectorized<Half>(1.0f);
}
inline Vectorized<Half> Vectorized<Half>::gt(const Vectorized<Half>& other) const {
  return (*this > other) & Vectorized<Half>(1.0f);
}
inline Vectorized<Half> Vectorized<Half>::ge(const Vectorized<Half>& other) const {
  return (*this >= other) & Vectorized<Half>(1.0f);
}
inline Vectorized<Half> Vectorized<Half>::lt(const Vectorized<Half>& other) const {
  return (*this < other) & Vectorized<Half>(1.0f);
}
inline Vectorized<Half> Vectorized<Half>::le(const Vectorized<Half>& other) const {
  return (*this <= other) & Vectorized<Half>(1.0f);
}

// frac. Implement this here so we can use subtraction
inline Vectorized<Half> Vectorized<Half>::frac() const {
  return *this - this->trunc();
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<Half> inline maximum(const Vectorized<Half>& a, const Vectorized<Half>& b) {
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
Vectorized<Half> inline minimum(const Vectorized<Half>& a, const Vectorized<Half>& b) {
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
Vectorized<Half> inline clamp(const Vectorized<Half>& a,
    const Vectorized<Half>& min, const Vectorized<Half>& max) {
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
Vectorized<Half> inline clamp_max(const Vectorized<Half>& a, const Vectorized<Half>& max) {
  __m256 a_lo, a_hi;
  __m256 max_lo, max_hi;
  cvtfp16_fp32(__m256i(a), a_lo, a_hi);
  cvtfp16_fp32(__m256i(max), max_lo, max_hi);
  auto o1 = _mm256_min_ps(max_lo, a_lo);
  auto o2 = _mm256_min_ps(max_hi, a_hi);
  return cvtfp32_fp16(o1, o2);
}

template <>
Vectorized<Half> inline clamp_min(const Vectorized<Half>& a, const Vectorized<Half>& min) {
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
  for (i = 0; i <= (n - Vectorized<Half>::size()); i += Vectorized<Half>::size()) {
    auto vsrc = _mm256_loadu_si256(reinterpret_cast<__m256i*>((void*)(src + i)));
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
  for (i = 0; i + Vectorized<Half>::size() <= n; i += Vectorized<Half>::size()) {
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
  auto load_float = [](const double *src) -> __m256 {
    // Load one float vector from an array of doubles
    __m128 a = _mm256_cvtpd_ps(_mm256_loadu_pd(src));
    __m128 b = _mm256_cvtpd_ps(_mm256_loadu_pd(src + 4));
    return _mm256_insertf128_ps(_mm256_castps128_ps256(a), b, 1);
  };

  int64_t i;
  for (i = 0; i + Vectorized<Half>::size() <= n; i += Vectorized<Half>::size()) {
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
Vectorized<Half> inline fmadd(const Vectorized<Half>& a,
    const Vectorized<Half>& b, const Vectorized<Half>& c) {
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

#else // defined(CPU_CAPABILITY_AVX2)

#if defined(__aarch64__) && !defined(C10_MOBILE) && !defined(__CUDACC__)
inline std::tuple<Vectorized<float>, Vectorized<float>> convert_half_float(const Vectorized<Half>& a) {
  static_assert(Vectorized<Half>::size() == 2 * Vectorized<float>::size());
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  float16x8x2_t arr = a;
  float16x8_t x = arr.val[0];
  float16x8_t y = arr.val[1];
#else
  auto arr = reinterpret_cast<const float16_t*>(a.operator const Half*());
  float16x8_t x = vld1q_f16(arr);
  float16x8_t y = vld1q_f16(arr + Vectorized<float>::size());
#endif
  float32x4_t x1 = vcvt_f32_f16(vget_low_f16(x));
  float32x4_t x2 = vcvt_f32_f16(vget_high_f16(x));
  float32x4_t y1 = vcvt_f32_f16(vget_low_f16(y));
  float32x4_t y2 = vcvt_f32_f16(vget_high_f16(y));
  return { Vectorized<float>(x1, x2), Vectorized<float>(y1, y2) };
}
inline Vectorized<Half> convert_float_half(const Vectorized<float>& a, const Vectorized<float>& b) {
  static_assert(Vectorized<Half>::size() == 2 * Vectorized<float>::size());
  float32x4x2_t x = a;
  float32x4x2_t y = b;
  float16x4_t x1 = vcvt_f16_f32(x.val[0]);
  float16x4_t x2 = vcvt_f16_f32(x.val[1]);
  float16x4_t y1 = vcvt_f16_f32(y.val[0]);
  float16x4_t y2 = vcvt_f16_f32(y.val[1]);
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  return Vectorized<Half>(vcombine_f16(x1, x2), vcombine_f16(y1, y2));
#else
  Vectorized<Half> rc;
  auto arr = reinterpret_cast<float16_t*>(rc.operator Half*());
  vst1q_f16(arr, vcombine_f16(x1, x2));
  vst1q_f16(arr + Vectorized<float>::size(), vcombine_f16(y1, y2));
  return rc;
#endif
}
#else
CONVERT_NON_VECTORIZED_INIT(Half, half);
#endif

#endif // defined(CPU_CAPABILITY_AVX2)

#if defined(CPU_CAPABILITY_AVX2)
LOAD_FP32_VECTORIZED_INIT(Half, fp16);
#else // defined(CPU_CAPABILITY_AVX2)
LOAD_FP32_NON_VECTORIZED_INIT(Half, fp16);

#endif
}} // namsepace at::vec::CPU_CAPABILITY

#pragma GCC diagnostic pop
