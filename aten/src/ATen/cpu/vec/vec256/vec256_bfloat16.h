#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/vec256/vec256_16bit_float.h>
#include <c10/util/irange.h>

namespace at::vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX2)

template <>
struct is_vec_specialized_for<BFloat16> : std::bool_constant<true> {};

template <>
class Vectorized<BFloat16> : public Vectorized16<BFloat16> {
 public:
  using Vectorized16::Vectorized16;

  using value_type = BFloat16;

  Vectorized<BFloat16> frac() const;

  Vectorized<BFloat16> eq(const Vectorized<BFloat16>& other) const;
  Vectorized<BFloat16> ne(const Vectorized<BFloat16>& other) const;
  Vectorized<BFloat16> gt(const Vectorized<BFloat16>& other) const;
  Vectorized<BFloat16> ge(const Vectorized<BFloat16>& other) const;
  Vectorized<BFloat16> lt(const Vectorized<BFloat16>& other) const;
  Vectorized<BFloat16> le(const Vectorized<BFloat16>& other) const;
};

Vectorized<BFloat16> inline operator+(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b) {
  return binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) {
    return _mm256_add_ps(x, y);
  });
}
Vectorized<BFloat16> inline operator-(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b) {
  return binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) {
    return _mm256_sub_ps(x, y);
  });
}
Vectorized<BFloat16> inline operator*(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b) {
  return binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) {
    return _mm256_mul_ps(x, y);
  });
}
Vectorized<BFloat16> inline operator/(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b) {
  return binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) {
    return _mm256_div_ps(x, y);
  });
}
Vectorized<BFloat16> inline operator&(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b) {
  return _mm256_and_si256(a, b);
}
Vectorized<BFloat16> inline operator|(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b) {
  return _mm256_or_si256(a, b);
}
Vectorized<BFloat16> inline operator^(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b) {
  return _mm256_xor_si256(a, b);
}

inline Vectorized<BFloat16> Vectorized<BFloat16>::eq(
    const Vectorized<BFloat16>& other) const {
  return (*this == other) & Vectorized<BFloat16>(1.0f);
}
inline Vectorized<BFloat16> Vectorized<BFloat16>::ne(
    const Vectorized<BFloat16>& other) const {
  return (*this != other) & Vectorized<BFloat16>(1.0f);
}
inline Vectorized<BFloat16> Vectorized<BFloat16>::gt(
    const Vectorized<BFloat16>& other) const {
  return (*this > other) & Vectorized<BFloat16>(1.0f);
}
inline Vectorized<BFloat16> Vectorized<BFloat16>::ge(
    const Vectorized<BFloat16>& other) const {
  return (*this >= other) & Vectorized<BFloat16>(1.0f);
}
inline Vectorized<BFloat16> Vectorized<BFloat16>::lt(
    const Vectorized<BFloat16>& other) const {
  return (*this < other) & Vectorized<BFloat16>(1.0f);
}
inline Vectorized<BFloat16> Vectorized<BFloat16>::le(
    const Vectorized<BFloat16>& other) const {
  return (*this <= other) & Vectorized<BFloat16>(1.0f);
}

// frac. Implement this here so we can use subtraction
inline Vectorized<BFloat16> Vectorized<BFloat16>::frac() const {
  return *this - this->trunc();
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<BFloat16> inline maximum(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b) {
  __m256 a_lo, a_hi;
  __m256 b_lo, b_hi;
  cvtbf16_fp32(__m256i(a), a_lo, a_hi);
  cvtbf16_fp32(__m256i(b), b_lo, b_hi);
  auto max_lo = _mm256_max_ps(a_lo, b_lo);
  auto max_hi = _mm256_max_ps(a_hi, b_hi);
  auto nan_lo = _mm256_cmp_ps(a_lo, b_lo, _CMP_UNORD_Q);
  auto nan_hi = _mm256_cmp_ps(a_hi, b_hi, _CMP_UNORD_Q);
  // Exploit the fact that all-ones is a NaN.
  auto o1 = _mm256_or_ps(max_lo, nan_lo);
  auto o2 = _mm256_or_ps(max_hi, nan_hi);
  return cvtfp32_bf16(o1, o2);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<BFloat16> inline minimum(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b) {
  __m256 a_lo, a_hi;
  __m256 b_lo, b_hi;
  cvtbf16_fp32(__m256i(a), a_lo, a_hi);
  cvtbf16_fp32(__m256i(b), b_lo, b_hi);
  auto min_lo = _mm256_min_ps(a_lo, b_lo);
  auto min_hi = _mm256_min_ps(a_hi, b_hi);
  auto nan_lo = _mm256_cmp_ps(a_lo, b_lo, _CMP_UNORD_Q);
  auto nan_hi = _mm256_cmp_ps(a_hi, b_hi, _CMP_UNORD_Q);
  // Exploit the fact that all-ones is a NaN.
  auto o1 = _mm256_or_ps(min_lo, nan_lo);
  auto o2 = _mm256_or_ps(min_hi, nan_hi);
  return cvtfp32_bf16(o1, o2);
}

template <>
Vectorized<BFloat16> inline clamp(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& min,
    const Vectorized<BFloat16>& max) {
  __m256 a_lo, a_hi;
  __m256 min_lo, min_hi;
  __m256 max_lo, max_hi;
  cvtbf16_fp32(__m256i(a), a_lo, a_hi);
  cvtbf16_fp32(__m256i(min), min_lo, min_hi);
  cvtbf16_fp32(__m256i(max), max_lo, max_hi);
  auto o1 = _mm256_min_ps(max_lo, _mm256_max_ps(min_lo, a_lo));
  auto o2 = _mm256_min_ps(max_hi, _mm256_max_ps(min_hi, a_hi));
  return cvtfp32_bf16(o1, o2);
}

template <>
Vectorized<BFloat16> inline clamp_max(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& max) {
  __m256 a_lo, a_hi;
  __m256 max_lo, max_hi;
  cvtbf16_fp32(__m256i(a), a_lo, a_hi);
  cvtbf16_fp32(__m256i(max), max_lo, max_hi);
  auto o1 = _mm256_min_ps(max_lo, a_lo);
  auto o2 = _mm256_min_ps(max_hi, a_hi);
  return cvtfp32_bf16(o1, o2);
}

template <>
Vectorized<BFloat16> inline clamp_min(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& min) {
  __m256 a_lo, a_hi;
  __m256 min_lo, min_hi;
  cvtbf16_fp32(__m256i(a), a_lo, a_hi);
  cvtbf16_fp32(__m256i(min), min_lo, min_hi);
  auto o1 = _mm256_max_ps(min_lo, a_lo);
  auto o2 = _mm256_max_ps(min_hi, a_hi);
  return cvtfp32_bf16(o1, o2);
}

template <>
inline void convert(const BFloat16* src, BFloat16* dst, int64_t n) {
  int64_t i;
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (i = 0; i <= (n - Vectorized<BFloat16>::size());
       i += Vectorized<BFloat16>::size()) {
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
inline void convert(const float* src, BFloat16* dst, int64_t n) {
  int64_t i;
  for (i = 0; i + Vectorized<BFloat16>::size() <= n;
       i += Vectorized<BFloat16>::size()) {
    __m256 a = _mm256_loadu_ps(&src[i]);
    __m256 b = _mm256_loadu_ps(&src[i + 8]);

    __m256i bf = cvtfp32_bf16(a, b);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[i]), bf);
  }
  for (; i < n; i++) {
    dst[i] = c10::convert<BFloat16>(src[i]);
  }
}

template <>
inline void convert(const double* src, BFloat16* dst, int64_t n) {
  auto load_float = [](const double* src) -> __m256 {
    // Load one float vector from an array of doubles
    __m128 a = _mm256_cvtpd_ps(_mm256_loadu_pd(src));
    __m128 b = _mm256_cvtpd_ps(_mm256_loadu_pd(src + 4));
    return _mm256_insertf128_ps(_mm256_castps128_ps256(a), b, 1);
  };

  int64_t i;
  for (i = 0; i + Vectorized<BFloat16>::size() <= n;
       i += Vectorized<BFloat16>::size()) {
    __m256 a = load_float(&src[i]);
    __m256 b = load_float(&src[i + 8]);

    __m256i bf = cvtfp32_bf16(a, b);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[i]), bf);
  }
  for (; i < n; i++) {
    dst[i] = c10::convert<BFloat16>(src[i]);
  }
}

template <>
Vectorized<BFloat16> inline fmadd(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b,
    const Vectorized<BFloat16>& c) {
  __m256 a_lo, a_hi;
  __m256 b_lo, b_hi;
  __m256 c_lo, c_hi;
  cvtbf16_fp32(__m256i(a), a_lo, a_hi);
  cvtbf16_fp32(__m256i(b), b_lo, b_hi);
  cvtbf16_fp32(__m256i(c), c_lo, c_hi);
  auto o1 = _mm256_fmadd_ps(a_lo, b_lo, c_lo);
  auto o2 = _mm256_fmadd_ps(a_hi, b_hi, c_hi);
  return cvtfp32_bf16(o1, o2);
}

CONVERT_VECTORIZED_INIT(BFloat16, bfloat16)
LOAD_FP32_VECTORIZED_INIT(BFloat16, bf16)

#else // defined(CPU_CAPABILITY_AVX2)

#if defined(__aarch64__) && !defined(C10_MOBILE) && !defined(__CUDACC__) && \
    !defined(CPU_CAPABILITY_SVE256)
// Upcasts bf16 tiles to fp32, transposes them with NEON lane shuffles, and
// downcasts back to bf16 to keep bf16 transpose results consistent on AArch64.
template <>
inline void transpose_mxn<BFloat16>(
    const BFloat16* src,
    int64_t ld_src,
    BFloat16* dst,
    int64_t ld_dst,
    int M,
    int N) {
  if (M <= 0 || N <= 0) {
    return;
  }

  constexpr int kBlock = 8;
  if (M <= kBlock && N <= kBlock) {
    auto load_row = [N](const BFloat16* row_ptr, int row_idx, int rows) {
      return row_idx < rows
          ? Vectorized<BFloat16>::loadu(row_ptr, N)
          : at_vdupq_n_bf16(0);
    };

    at_bfloat16x8_t rows[kBlock];
    for (int i = 0; i < kBlock; ++i) {
      rows[i] = load_row(src + i * ld_src, i, M);
    }

    auto bf16_to_f32 = [](at_bfloat16x8_t v,
                          float32x4_t& lo,
                          float32x4_t& hi) {
#ifdef __ARM_FEATURE_BF16
      lo = vcvt_f32_bf16(at_vget_low_bf16(v));
      hi = vcvt_f32_bf16(at_vget_high_bf16(v));
#else
      lo = vreinterpretq_f32_u32(vshll_n_u16(at_vget_low_bf16(v), 16));
      hi = vreinterpretq_f32_u32(vshll_n_u16(at_vget_high_bf16(v), 16));
#endif
    };

    auto f32_to_bf16 = [](float32x4_t lo, float32x4_t hi) {
#ifdef __ARM_FEATURE_BF16
      return at_vcombine_bf16(vcvt_bf16_f32(lo), vcvt_bf16_f32(hi));
#else
      auto round_vec = [](float32x4_t v) {
        const uint32x4_t as_u32 = vreinterpretq_u32_f32(v);
        const uint32x4_t lsb =
            vandq_u32(vshrq_n_u32(as_u32, 16), vdupq_n_u32(1));
        const uint32x4_t bias = vaddq_u32(lsb, vdupq_n_u32(0x7FFF));
        const uint32x4_t rounded = vaddq_u32(as_u32, bias);
        uint16x4_t bf16 = vshrn_n_u32(rounded, 16);
        const uint32x4_t nan_mask =
            vmvnq_u32(vreinterpretq_u32_f32(vceqq_f32(v, v)));
        const uint16x4_t bf16_nan = vdup_n_u16(0x7FC0);
        bf16 = vbsl_u16(vmovn_u32(nan_mask), bf16_nan, bf16);
        return bf16;
      };
      return at_vcombine_bf16(round_vec(lo), round_vec(hi));
#endif
    };

    auto transpose4x4 = [](float32x4_t& x0,
                           float32x4_t& x1,
                           float32x4_t& x2,
                           float32x4_t& x3) {
      const float32x4_t t0 = vtrn1q_f32(x0, x1);
      const float32x4_t t1 = vtrn2q_f32(x0, x1);
      const float32x4_t t2 = vtrn1q_f32(x2, x3);
      const float32x4_t t3 = vtrn2q_f32(x2, x3);

      x0 = vcombine_f32(vget_low_f32(t0), vget_low_f32(t2));
      x1 = vcombine_f32(vget_low_f32(t1), vget_low_f32(t3));
      x2 = vcombine_f32(vget_high_f32(t0), vget_high_f32(t2));
      x3 = vcombine_f32(vget_high_f32(t1), vget_high_f32(t3));
    };

    float32x4_t a0, a1, a2, a3, a4, a5, a6, a7;
    float32x4_t b0, b1, b2, b3, b4, b5, b6, b7;
    bf16_to_f32(rows[0], a0, b0);
    bf16_to_f32(rows[1], a1, b1);
    bf16_to_f32(rows[2], a2, b2);
    bf16_to_f32(rows[3], a3, b3);
    bf16_to_f32(rows[4], a4, b4);
    bf16_to_f32(rows[5], a5, b5);
    bf16_to_f32(rows[6], a6, b6);
    bf16_to_f32(rows[7], a7, b7);

    transpose4x4(a0, a1, a2, a3); // A^T
    transpose4x4(a4, a5, a6, a7); // C^T
    transpose4x4(b0, b1, b2, b3); // B^T
    transpose4x4(b4, b5, b6, b7); // D^T

    float32x4_t left[] = {a0, a1, a2, a3, b0, b1, b2, b3};
    float32x4_t right[] = {a4, a5, a6, a7, b4, b5, b6, b7};

    for (int row = 0; row < N; ++row) {
      Vectorized<BFloat16>(f32_to_bf16(left[row], right[row]))
          .store(dst + row * ld_dst, M);
    }
    return;
  }

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      dst[j * ld_dst + i] = src[i * ld_src + j];
    }
  }
}
#endif

#if !(                                                                      \
    defined(__aarch64__) && !defined(C10_MOBILE) && !defined(__CUDACC__) && \
    !defined(CPU_CAPABILITY_SVE256))
CONVERT_NON_VECTORIZED_INIT(BFloat16, bfloat16)
#endif

LOAD_FP32_NON_VECTORIZED_INIT(BFloat16, bf16)
#endif // defined(CPU_CAPABILITY_AVX2)
} // namespace CPU_CAPABILITY
} // namespace at::vec
