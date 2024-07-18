#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/util/irange.h>

#if defined(CPU_CAPABILITY_AVX512)
#define SLEEF_STATIC_LIBS
#include <sleef.h>
#endif

namespace at {
namespace vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX512)

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

// bfloat16 conversion
static inline void cvtbf16_fp32(const __m256i& a, __m512& o) {
  o = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(a), 16));
}

static inline void cvtbf16_fp32(const __m512i& a, __m512& o1, __m512& o2) {
  __m256i lo = _mm512_extracti32x8_epi32(a, 0);
  __m256i hi = _mm512_extracti32x8_epi32(a, 1);
  cvtbf16_fp32(lo, o1);
  cvtbf16_fp32(hi, o2);
}

static inline __m256i cvtfp32_bf16(const __m512& src) {
  __m512i value = _mm512_castps_si512(src);
  __m512i nan = _mm512_set1_epi32(0xffff);
  auto mask_value = _mm512_cmp_ps_mask(src, src, _CMP_ORD_Q);
  __m512i ones = _mm512_set1_epi32(0x1);
  __m512i vec_bias = _mm512_set1_epi32(0x7fff);
  // uint32_t lsb = (input >> 16) & 1;
  auto t_value = _mm512_and_si512(_mm512_srli_epi32(value, 16), ones);
  // uint32_t rounding_bias = 0x7fff + lsb;
  t_value = _mm512_add_epi32(t_value, vec_bias);
  // input += rounding_bias;
  t_value = _mm512_add_epi32(t_value, value);
  // input = input >> 16;
  t_value = _mm512_srli_epi32(t_value, 16);
  // Check NaN before converting back to bf16
  t_value = _mm512_mask_blend_epi32(mask_value, nan, t_value);
  return _mm512_cvtusepi32_epi16(t_value);
}

static inline __m512i cvtfp32_bf16(const __m512& a, const __m512& b) {
  __m512i lo = _mm512_castps_si512(a);
  __m512i hi = _mm512_castps_si512(b);
  __m512i nan = _mm512_set1_epi32(0xffff);
  auto mask_lo = _mm512_cmp_ps_mask(a, a, _CMP_ORD_Q);
  auto mask_hi = _mm512_cmp_ps_mask(b, b, _CMP_ORD_Q);
  __m512i ones = _mm512_set1_epi32(0x1);
  __m512i vec_bias = _mm512_set1_epi32(0x7fff);
  // uint32_t lsb = (input >> 16) & 1;
  auto t_lo = _mm512_and_si512(_mm512_srli_epi32(lo, 16), ones);
  auto t_hi = _mm512_and_si512(_mm512_srli_epi32(hi, 16), ones);
  // uint32_t rounding_bias = 0x7fff + lsb;
  t_lo = _mm512_add_epi32(t_lo, vec_bias);
  t_hi = _mm512_add_epi32(t_hi, vec_bias);
  // input += rounding_bias;
  t_lo = _mm512_add_epi32(t_lo, lo);
  t_hi = _mm512_add_epi32(t_hi, hi);
  // input = input >> 16;
  t_lo = _mm512_srli_epi32(t_lo, 16);
  t_hi = _mm512_srli_epi32(t_hi, 16);
  // Check NaN before converting back to bf16
  t_lo = _mm512_mask_blend_epi32(mask_lo, nan, t_lo);
  t_hi = _mm512_mask_blend_epi32(mask_hi, nan, t_hi);

  t_lo = _mm512_packus_epi32(t_lo, t_hi); // t_hi[4-7] t_lo[4-7] t_hi[0-4] t_lo[0-4]
  __m512i idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
  return _mm512_permutexvar_epi64(idx, t_lo);
}

static inline __m512i merge_compare_result(const __m512& a, const __m512& b) {
  __m512i lo = _mm512_castps_si512(a);
  __m512i hi = _mm512_castps_si512(b);
  lo = _mm512_srli_epi32(lo, 16);
  hi = _mm512_srli_epi32(hi, 16);
  auto out = _mm512_packus_epi32(lo, hi);
  __m512i idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
  return _mm512_permutexvar_epi64(idx, out);
}

// float16 conversion
static inline void cvtfp16_fp32(const __m256i& a, __m512& o) {
  o = _mm512_cvtph_ps(a);
}

static inline void cvtfp16_fp32(const __m512i& a, __m512& o1, __m512& o2) {
  __m256i lo = _mm512_extracti32x8_epi32(a, 0);
  __m256i hi = _mm512_extracti32x8_epi32(a, 1);
  cvtfp16_fp32(lo, o1);
  cvtfp16_fp32(hi, o2);
}

static inline __m256i cvtfp32_fp16(const __m512& src) {
  return _mm512_cvtps_ph(
      src, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

static inline __m512i cvtfp32_fp16(const __m512& a, const __m512& b) {
  __m256i lo = _mm512_cvtps_ph(
      a, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  __m256i hi = _mm512_cvtps_ph(
      b, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  __m512 t_lo = _mm512_castsi512_ps(_mm512_castsi256_si512(lo));
  __m256 t_hi = _mm256_castsi256_ps(hi);
  return _mm512_castps_si512(_mm512_insertf32x8(t_lo, t_hi, 1));
}

// dtype conversion between float16/bfloat16 and float32
template <typename T, typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline void cvt_to_fp32(const __m256i& a, __m512& o);
template <> inline void cvt_to_fp32<BFloat16>(const __m256i& a, __m512& o) {
  cvtbf16_fp32(a, o);
}
template <> inline void cvt_to_fp32<Half>(const __m256i& a, __m512& o) {
  cvtfp16_fp32(a, o);
}

template <typename T, typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline void cvt_to_fp32(const __m512i& a, __m512& o1, __m512& o2);
template <> inline void cvt_to_fp32<BFloat16>(const __m512i& a, __m512& o1, __m512& o2) {
  cvtbf16_fp32(a, o1, o2);
}
template <> inline void cvt_to_fp32<Half>(const __m512i& a, __m512& o1, __m512& o2) {
  cvtfp16_fp32(a, o1, o2);
}

template <typename T, bool is_compare_op = false,
          typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline __m512i cvt_from_fp32(const __m512& a, const __m512& b);
template <> inline __m512i cvt_from_fp32<BFloat16, false>(const __m512& a, const __m512& b) {
  return cvtfp32_bf16(a, b);
}
template <> inline __m512i cvt_from_fp32<BFloat16, true>(const __m512& a, const __m512& b) {
  return merge_compare_result(a, b);
}
template <> inline __m512i cvt_from_fp32<Half, false>(const __m512& a, const __m512& b) {
  return cvtfp32_fp16(a, b);
}
template <> inline __m512i cvt_from_fp32<Half, true>(const __m512& a, const __m512& b) {
  return cvtfp32_fp16(a, b);
}

template <typename T>
class Vectorized16 {
static_assert(
  is_reduced_floating_point_v<T>,
  "Support only float16 and bfloat16.");
private:
  __m512i values;
public:
  using value_type = uint16_t;
  using size_type = int;
  static constexpr size_type size() {
    return 32;
  }
  Vectorized16() {}
  Vectorized16(__m512i v) : values(v) {}
  Vectorized16(T val) {
    value_type uw = val.x;
    values = _mm512_set1_epi16(uw);
  }
  Vectorized16(T val1, T val2, T val3, T val4,
         T val5, T val6, T val7, T val8,
         T val9, T val10, T val11, T val12,
         T val13, T val14, T val15, T val16,
         T val17, T val18, T val19, T val20,
         T val21, T val22, T val23, T val24,
         T val25, T val26, T val27, T val28,
         T val29, T val30, T val31, T val32) {
    values = _mm512_set_epi16(
        val32.x, val31.x, val30.x, val29.x, val28.x, val27.x, val26.x, val25.x,
        val24.x, val23.x, val22.x, val21.x, val20.x, val19.x, val18.x, val17.x,
        val16.x, val15.x, val14.x, val13.x, val12.x, val11.x, val10.x, val9.x,
        val8.x, val7.x, val6.x, val5.x, val4.x, val3.x, val2.x, val1.x);
  }
  operator __m512i() const {
    return values;
  }
  T& operator[](int idx) = delete;
  const T& operator[](int idx) const  = delete;
  int zero_mask() const {
    // returns an integer mask where all zero elements are translated to 1-bit and others are translated to 0-bit
    return _mm512_cmpeq_epi16_mask(values, _mm512_set1_epi16(0));
  }
  static Vectorized<T> loadu(const void* ptr, int16_t count = size()) {
    if (count == size())
      return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr));

    __mmask32 mask = (1ULL << count) - 1;
    return _mm512_maskz_loadu_epi16(mask, ptr);
  }
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), values);
    } else if (count > 0) {
      __mmask32 mask = (1ULL << count) - 1;
      _mm512_mask_storeu_epi16(ptr, mask, values);
    }
  }
  template <int64_t mask>
  static Vectorized<T> blend(const Vectorized<T>& a, const Vectorized<T>& b) {
    __at_align__ int16_t tmp_values[size()];
    a.store(tmp_values);
    if (mask & 0x01)
      tmp_values[0] = b.values[31];
    if (mask & 0x02)
      tmp_values[1] = b.values[30];
    if (mask & 0x04)
      tmp_values[2] = b.values[29];
    if (mask & 0x08)
      tmp_values[3] = b.values[28];
    if (mask & 0x10)
      tmp_values[4] = b.values[27];
    if (mask & 0x20)
      tmp_values[5] = b.values[26];
    if (mask & 0x40)
      tmp_values[6] = b.values[25];
    if (mask & 0x80)
      tmp_values[7] = b.values[24];
    if (mask & 0x100)
      tmp_values[8] = b.values[23];
    if (mask & 0x200)
      tmp_values[9] = b.values[22];
    if (mask & 0x400)
      tmp_values[10] = b.values[21];
    if (mask & 0x800)
      tmp_values[11] = b.values[20];
    if (mask & 0x1000)
      tmp_values[12] = b.values[19];
    if (mask & 0x2000)
      tmp_values[13] = b.values[18];
    if (mask & 0x4000)
      tmp_values[14] = b.values[17];
    if (mask & 0x8000)
      tmp_values[15] = b.values[16];
    if (mask & 0x10000)
      tmp_values[16] = b.values[15];
    if (mask & 0x20000)
      tmp_values[17] = b.values[14];
    if (mask & 0x40000)
      tmp_values[18] = b.values[13];
    if (mask & 0x80000)
      tmp_values[19] = b.values[12];
    if (mask & 0x100000)
      tmp_values[20] = b.values[11];
    if (mask & 0x200000)
      tmp_values[21] = b.values[10];
    if (mask & 0x400000)
      tmp_values[22] = b.values[9];
    if (mask & 0x800000)
      tmp_values[23] = b.values[8];
    if (mask & 0x1000000)
      tmp_values[24] = b.values[7];
    if (mask & 0x2000000)
      tmp_values[25] = b.values[6];
    if (mask & 0x4000000)
      tmp_values[26] = b.values[5];
    if (mask & 0x8000000)
      tmp_values[27] = b.values[4];
    if (mask & 0x10000000)
      tmp_values[28] = b.values[3];
    if (mask & 0x20000000)
      tmp_values[29] = b.values[2];
    if (mask & 0x40000000)
      tmp_values[30] = b.values[1];
    if (mask & 0x80000000)
      tmp_values[31] = b.values[0];
    return loadu(tmp_values);
  }
  static Vectorized<T> blendv(const Vectorized<T>& a,
      const Vectorized<T>& b, const Vectorized<T>& mask) {
    auto all_ones = _mm512_set1_epi16(0xFFFF);
    auto mask_ = _mm512_cmp_epi16_mask(mask, all_ones, _MM_CMPINT_EQ);
    return _mm512_mask_blend_epi16(mask_, a.values, b.values);
  }
  template<typename step_t>
  static Vectorized<T> arange(T base = 0.f, step_t step = static_cast<step_t>(1)) {
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
  static Vectorized<T> set(const Vectorized<T>& a,
      const Vectorized<T>& b, int64_t count = size()) {
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
      case 16:
        return blend<65535>(a, b);
      case 17:
        return blend<131071>(a, b);
      case 18:
        return blend<262143>(a, b);
      case 19:
        return blend<524287>(a, b);
      case 20:
        return blend<1048575>(a, b);
      case 21:
        return blend<2097151>(a, b);
      case 22:
        return blend<4194303>(a, b);
      case 23:
        return blend<8388607>(a, b);
      case 24:
        return blend<16777215>(a, b);
      case 25:
        return blend<33554431>(a, b);
      case 26:
        return blend<67108863>(a, b);
      case 27:
        return blend<134217727>(a, b);
      case 28:
        return blend<268435455>(a, b);
      case 29:
        return blend<536870911>(a, b);
      case 30:
        return blend<1073741823>(a, b);
      case 31:
        return blend<2147483647>(a, b);
    }
    return b;
  }
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wignored-qualifiers"

  Vectorized<T> map(SLEEF_CONST __m512 (*SLEEF_CONST_OLD vop)(__m512)) const {
    __m512 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    const auto o1 = vop(lo);
    const auto o2 = vop(hi);
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> isnan() const {
    __m512 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    __mmask16 lo_mask, hi_mask;
    __m512 zero = _mm512_set1_ps(0.0);
    __m512i zeroi = _mm512_castps_si512(zero);
    lo_mask = _mm512_cmp_ps_mask(lo, zero, _CMP_UNORD_Q);
    lo = _mm512_castsi512_ps(_mm512_mask_set1_epi32(zeroi, lo_mask, 0xFFFF'FFFF));
    hi_mask = _mm512_cmp_ps_mask(hi, zero, _CMP_UNORD_Q);
    hi = _mm512_castsi512_ps(_mm512_mask_set1_epi32(zeroi, hi_mask, 0xFFFF'FFFF));
    return merge_compare_result(lo, hi);
  }
  #pragma clang diagnostic pop
  Vectorized<T> abs() const {
    return _mm512_andnot_si512(_mm512_set1_epi16(0x8000), values);
  }
  Vectorized<T> angle() const {
    __m512 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    auto angle_lambda = [](__m512 values) {
      const auto zero_vec = _mm512_set1_ps(0.f);
      const auto nan_vec = _mm512_set1_ps(NAN);
      const auto not_nan_mask = _mm512_cmp_ps_mask(values, values, _CMP_EQ_OQ);
      const auto non_nan_mask_vec = _mm512_mask_set1_epi32(_mm512_castps_si512(zero_vec),
                                                           not_nan_mask, 0xFFFFFFFF);
      const auto nan_mask = _mm512_cmp_ps_mask(_mm512_castsi512_ps(non_nan_mask_vec),
                                               zero_vec, _CMP_EQ_OQ);
      const auto pi = _mm512_set1_ps(c10::pi<float>);

      const auto neg_mask = _mm512_cmp_ps_mask(values, zero_vec, _CMP_LT_OQ);
      auto angle = _mm512_mask_blend_ps(neg_mask, zero_vec, pi);
      angle = _mm512_mask_blend_ps(nan_mask, angle, nan_vec);
      return angle;
    };
    auto o1 = angle_lambda(lo);
    auto o2 = angle_lambda(hi);
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> real() const {
    return *this;
  }
  Vectorized<T> imag() const {
    return _mm512_set1_epi16(0);
  }
  Vectorized<T> conj() const {
    return *this;
  }
  Vectorized<T> acos() const {
    return map(Sleef_acosf16_u10);
  }
  Vectorized<T> acosh() const {
    return map(Sleef_acoshf16_u10);
  }
  Vectorized<T> asin() const {
    return map(Sleef_asinf16_u10);
  }
  Vectorized<T> atan() const {
    return map(Sleef_atanf16_u10);
  }
  Vectorized<T> atanh() const {
    return map(Sleef_atanhf16_u10);
  }
  Vectorized<T> atan2(const Vectorized<T> &b) const {
    __m512 lo, hi;
    __m512 b1, b2;
    cvt_to_fp32<T>(values, lo, hi);
    cvt_to_fp32<T>(b.values, b1, b2);
    auto o1 = Sleef_atan2f16_u10(lo, b1);
    auto o2 = Sleef_atan2f16_u10(hi, b2);
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> copysign(const Vectorized<T> &sign) const {
    // copy sign bit (0x8000) from sign and remaining bits from values
    __m512i mask_value = _mm512_set1_epi32(~0x80008000);
    __m512i mask_signbit = _mm512_set1_epi32(0x80008000);
    return Vectorized<T>(
      _mm512_or_si512(
        _mm512_and_si512(values, mask_value),
        _mm512_and_si512(sign, mask_signbit)));
  }
  Vectorized<T> erf() const {
    return map(Sleef_erff16_u10);
  }
  Vectorized<T> erfc() const {
    return map(Sleef_erfcf16_u15);
  }
  Vectorized<T> erfinv() const {
    __m512 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    __at_align__ float tmp1[size() / 2], tmp2[size() / 2];
    _mm512_storeu_ps(reinterpret_cast<float*>(tmp1), lo);
    _mm512_storeu_ps(reinterpret_cast<float*>(tmp2), hi);
    for (int64_t i = 0; i < size() / 2; i++) {
      tmp1[i] = calc_erfinv(tmp1[i]);
      tmp2[i] = calc_erfinv(tmp2[i]);
    }
    auto o1 = _mm512_loadu_ps(tmp1);
    auto o2 = _mm512_loadu_ps(tmp2);
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> exp() const {
    return map(Sleef_expf16_u10);
  }
  Vectorized<T> exp2() const {
    return map(Sleef_exp2f16_u10);
  }
  Vectorized<T> expm1() const {
    return map(Sleef_expm1f16_u10);
  }
  Vectorized<T> exp_u20() const {
    return exp();
  }
  Vectorized<T> fmod(const Vectorized<T> & q) const {
    __m512 x_lo, x_hi;
    cvt_to_fp32<T>(values, x_lo, x_hi);
    __m512 q_lo, q_hi;
    cvtbf16_fp32(q.values, q_lo, q_hi);
    auto o1 = Sleef_fmodf16(x_lo, q_lo);
    auto o2 = Sleef_fmodf16(x_hi, q_hi);
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> hypot(const Vectorized<T> &b) const {
    __m512 lo, hi;
    __m512 b1, b2;
    cvt_to_fp32<T>(values, lo, hi);
    cvt_to_fp32<T>(b.values, b1, b2);
    auto o1 = Sleef_hypotf16_u05(lo, b1);
    auto o2 = Sleef_hypotf16_u05(hi, b2);
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> i0() const {
    __m512 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    __at_align__ float tmp1[size() / 2], tmp2[size() / 2];
    _mm512_storeu_ps(reinterpret_cast<float*>(tmp1), lo);
    _mm512_storeu_ps(reinterpret_cast<float*>(tmp2), hi);
    for (int64_t i = 0; i < size() / 2; i++) {
      tmp1[i] = calc_i0(tmp1[i]);
      tmp2[i] = calc_i0(tmp2[i]);
    }
    auto o1 = _mm512_loadu_ps(tmp1);
    auto o2 = _mm512_loadu_ps(tmp2);
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> i0e() const {
    __m512 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    constexpr auto sz = size();
    __at_align__ float tmp1[sz / 2], tmp2[sz / 2];
    _mm512_storeu_ps(reinterpret_cast<float*>(tmp1), lo);
    _mm512_storeu_ps(reinterpret_cast<float*>(tmp2), hi);

    for (auto i = decltype(sz){0}; i < sz / 2; i++) {
      tmp1[i] = calc_i0e(tmp1[i]);
      tmp2[i] = calc_i0e(tmp2[i]);
    }
    const auto o1 = _mm512_loadu_ps(tmp1);
    const auto o2 = _mm512_loadu_ps(tmp2);
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> digamma() const {
    __m512 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    constexpr auto sz = size();
    __at_align__ float tmp1[sz / 2], tmp2[sz / 2];
    _mm512_storeu_ps(reinterpret_cast<float*>(tmp1), lo);
    _mm512_storeu_ps(reinterpret_cast<float*>(tmp2), hi);

    for (auto i = decltype(sz){0}; i < sz / 2; i++) {
      tmp1[i] = calc_digamma(tmp1[i]);
      tmp2[i] = calc_digamma(tmp2[i]);
    }
    const auto o1 = _mm512_loadu_ps(tmp1);
    const auto o2 = _mm512_loadu_ps(tmp2);
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> igamma(const Vectorized<T> &x) const {
    __m512 lo, hi;
    __m512 xlo, xhi;
    cvt_to_fp32<T>(values, lo, hi);
    cvt_to_fp32<T>(x.values, xlo, xhi);
    __at_align__ float tmp1[size() / 2], tmp2[size() / 2];
    _mm512_storeu_ps(reinterpret_cast<float*>(tmp1), lo);
    _mm512_storeu_ps(reinterpret_cast<float*>(tmp2), hi);
    __at_align__ float tmpx1[size() / 2], tmpx2[size() / 2];
    _mm512_storeu_ps(reinterpret_cast<float*>(tmpx1), xlo);
    _mm512_storeu_ps(reinterpret_cast<float*>(tmpx2), xhi);
    for (int64_t i = 0; i < size() / 2; ++i) {
      tmp1[i] = calc_igamma(tmp1[i], tmpx1[i]);
      tmp2[i] = calc_igamma(tmp2[i], tmpx2[i]);
    }
    auto o1 = _mm512_loadu_ps(tmp1);
    auto o2 = _mm512_loadu_ps(tmp2);
    return cvt_from_fp32<T>(o1, o2);
  }

  Vectorized<T> igammac(const Vectorized<T> &x) const {
    __m512 lo, hi;
    __m512 xlo, xhi;
    cvt_to_fp32<T>(values, lo, hi);
    cvt_to_fp32<T>(x.values, xlo, xhi);
    __at_align__ float tmp1[size() / 2], tmp2[size() / 2];
    _mm512_storeu_ps(reinterpret_cast<float*>(tmp1), lo);
    _mm512_storeu_ps(reinterpret_cast<float*>(tmp2), hi);
    __at_align__ float tmpx1[size() / 2], tmpx2[size() / 2];
    _mm512_storeu_ps(reinterpret_cast<float*>(tmpx1), xlo);
    _mm512_storeu_ps(reinterpret_cast<float*>(tmpx2), xhi);
    for (int64_t i = 0; i < size() / 2; ++i) {
      tmp1[i] = calc_igammac(tmp1[i], tmpx1[i]);
      tmp2[i] = calc_igammac(tmp2[i], tmpx2[i]);
    }
    auto o1 = _mm512_loadu_ps(tmp1);
    auto o2 = _mm512_loadu_ps(tmp2);
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> log() const {
    return map(Sleef_logf16_u10);
  }
  Vectorized<T> log2() const {
    return map(Sleef_log2f16_u10);
  }
  Vectorized<T> log10() const {
    return map(Sleef_log10f16_u10);
  }
  Vectorized<T> log1p() const {
    return map(Sleef_log1pf16_u10);
  }
  Vectorized<T> sin() const {
    return map(Sleef_sinf16_u10);
  }
  Vectorized<T> sinh() const {
    return map(Sleef_sinhf16_u10);
  }
  Vectorized<T> cos() const {
    return map(Sleef_cosf16_u10);
  }
  Vectorized<T> cosh() const {
    return map(Sleef_coshf16_u10);
  }
  Vectorized<T> ceil() const {
    __m512 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    auto o1 = _mm512_ceil_ps(lo);
    auto o2 = _mm512_ceil_ps(hi);
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> floor() const {
    __m512 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    auto o1 = _mm512_floor_ps(lo);
    auto o2 = _mm512_floor_ps(hi);
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> neg() const {
    return _mm512_xor_si512(values, _mm512_set1_epi16(0x8000));
  }
  Vectorized<T> round() const {
    __m512 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    auto o1 = _mm512_roundscale_ps(lo, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    auto o2 = _mm512_roundscale_ps(hi, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> tan() const {
    return map(Sleef_tanf16_u10);
  }
  Vectorized<T> tanh() const {
    return map(Sleef_tanhf16_u10);
  }
  Vectorized<T> trunc() const {
    __m512 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    auto o1 = _mm512_roundscale_ps(lo, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
    auto o2 = _mm512_roundscale_ps(hi, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> lgamma() const {
    return map(Sleef_lgammaf16_u10);
  }
  Vectorized<T> sqrt() const {
    __m512 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    auto o1 = _mm512_sqrt_ps(lo);
    auto o2 = _mm512_sqrt_ps(hi);
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> reciprocal() const {
    __m512 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    auto ones = _mm512_set1_ps(1);
    auto o1 = _mm512_div_ps(ones, lo);
    auto o2 = _mm512_div_ps(ones, hi);
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> rsqrt() const {
    __m512 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    auto ones = _mm512_set1_ps(1);
    auto o1 = _mm512_div_ps(ones, _mm512_sqrt_ps(lo));
    auto o2 = _mm512_div_ps(ones, _mm512_sqrt_ps(hi));
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> pow(const Vectorized<T> &b) const {
    __m512 lo, hi;
    __m512 b1, b2;
    cvt_to_fp32<T>(values, lo, hi);
    cvt_to_fp32<T>(b.values, b1, b2);
    auto o1 = Sleef_powf16_u10(lo, b1);
    auto o2 = Sleef_powf16_u10(hi, b2);
    return cvt_from_fp32<T>(o1, o2);
  }
private:
  template<typename Op>
  Vectorized<T> inline binary_compare(const Vectorized<T>& b, Op op) const {
    __m512 a_lo, a_hi;
    __m512 b_lo, b_hi;
    cvt_to_fp32<T>(values, a_lo, a_hi);
    cvt_to_fp32<T>(b.values, b_lo, b_hi);
    auto o1 = op(a_lo, b_lo);
    auto o2 = op(a_hi, b_hi);
    return cvt_from_fp32<T, /*is_compare_op*/true>(o1, o2);
  }

public:
  Vectorized<T> inline operator>(const Vectorized<T>& other) const {
    return binary_compare(other, [](__m512 x, __m512 y) {
      auto zero_vec = _mm512_set1_epi32(0);
      auto cmp = _mm512_cmp_ps_mask(x, y, _CMP_GT_OQ);
      return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, cmp, 0xFFFFFFFF));
    });
  }
  Vectorized<T> inline operator<(const Vectorized<T>& other) const {
    return binary_compare(other, [](__m512 x, __m512 y) {
      auto zero_vec = _mm512_set1_epi32(0);
      auto cmp = _mm512_cmp_ps_mask(x, y, _CMP_LT_OQ);
      return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, cmp, 0xFFFFFFFF));
    });
  }
  Vectorized<T> inline operator>=(const Vectorized<T>& other) const {
    return binary_compare(other, [](__m512 x, __m512 y) {
      auto zero_vec = _mm512_set1_epi32(0);
      auto cmp = _mm512_cmp_ps_mask(x, y, _CMP_GE_OQ);
      return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, cmp, 0xFFFFFFFF));
    });
  }
  Vectorized<T> inline operator<=(const Vectorized<T>& other) const {
    return binary_compare(other, [](__m512 x, __m512 y) {
      auto zero_vec = _mm512_set1_epi32(0);
      auto cmp = _mm512_cmp_ps_mask(x, y, _CMP_LE_OQ);
      return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, cmp, 0xFFFFFFFF));
    });
  }
  Vectorized<T> inline operator==(const Vectorized<T>& other) const {
    return binary_compare(other, [](__m512 x, __m512 y) {
      auto zero_vec = _mm512_set1_epi32(0);
      auto cmp = _mm512_cmp_ps_mask(x, y, _CMP_EQ_OQ);
      return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, cmp, 0xFFFFFFFF));
    });
  }
  Vectorized<T> inline operator!=(const Vectorized<T>& other) const {
    return binary_compare(other, [](__m512 x, __m512 y) {
      auto zero_vec = _mm512_set1_epi32(0);
      auto cmp = _mm512_cmp_ps_mask(x, y, _CMP_NEQ_UQ);
      return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, cmp, 0xFFFFFFFF));
    });
  }
};

template<typename T, typename Op>
static inline Vectorized<T> binary_op_as_fp32(const Vectorized<T>& a, const Vectorized<T>& b, Op op) {
  __m512 a_lo, a_hi;
  __m512 b_lo, b_hi;
  cvt_to_fp32<T>(__m512i(a), a_lo, a_hi);
  cvt_to_fp32<T>(__m512i(b), b_lo, b_hi);
  auto o1 = op(a_lo, b_lo);
  auto o2 = op(a_hi, b_hi);
  return cvt_from_fp32<T>(o1, o2);
}

template <>
class Vectorized<BFloat16>: public Vectorized16<BFloat16> {
public:
  using Vectorized16::Vectorized16;

  Vectorized<BFloat16> frac() const;

  Vectorized<BFloat16> eq(const Vectorized<BFloat16>& other) const;
  Vectorized<BFloat16> ne(const Vectorized<BFloat16>& other) const;
  Vectorized<BFloat16> gt(const Vectorized<BFloat16>& other) const;
  Vectorized<BFloat16> ge(const Vectorized<BFloat16>& other) const;
  Vectorized<BFloat16> lt(const Vectorized<BFloat16>& other) const;
  Vectorized<BFloat16> le(const Vectorized<BFloat16>& other) const;
};

Vectorized<BFloat16> inline operator+(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
  return binary_op_as_fp32(a, b, [](const __m512& x, const __m512& y) { return _mm512_add_ps(x, y); });
}
Vectorized<BFloat16> inline operator-(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
  return binary_op_as_fp32(a, b, [](const __m512& x, const __m512& y) { return _mm512_sub_ps(x, y); });
}
Vectorized<BFloat16> inline operator*(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
  return binary_op_as_fp32(a, b, [](const __m512& x, const __m512& y) { return _mm512_mul_ps(x, y); });
}
Vectorized<BFloat16> inline operator/(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
  return binary_op_as_fp32(a, b, [](const __m512& x, const __m512& y) { return _mm512_div_ps(x, y); });
}
Vectorized<BFloat16> inline operator&(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
  return _mm512_and_si512(a, b);
}
Vectorized<BFloat16> inline operator|(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
  return _mm512_or_si512(a, b);
}
Vectorized<BFloat16> inline operator^(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
  return _mm512_xor_si512(a, b);
}

inline Vectorized<BFloat16> Vectorized<BFloat16>::eq(const Vectorized<BFloat16>& other) const {
  return (*this == other) & Vectorized<BFloat16>(1.0f);
}

inline Vectorized<BFloat16> Vectorized<BFloat16>::ne(const Vectorized<BFloat16>& other) const {
  return (*this != other) & Vectorized<BFloat16>(1.0f);
}

inline Vectorized<BFloat16> Vectorized<BFloat16>::gt(const Vectorized<BFloat16>& other) const {
  return (*this > other) & Vectorized<BFloat16>(1.0f);
}

inline Vectorized<BFloat16> Vectorized<BFloat16>::ge(const Vectorized<BFloat16>& other) const {
  return (*this >= other) & Vectorized<BFloat16>(1.0f);
}

inline Vectorized<BFloat16> Vectorized<BFloat16>::lt(const Vectorized<BFloat16>& other) const {
  return (*this < other) & Vectorized<BFloat16>(1.0f);
}

inline Vectorized<BFloat16> Vectorized<BFloat16>::le(const Vectorized<BFloat16>& other) const {
  return (*this <= other) & Vectorized<BFloat16>(1.0f);
}

// frac. Implement this here so we can use subtraction
inline Vectorized<BFloat16> Vectorized<BFloat16>::frac() const {
  return *this - this->trunc();
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<BFloat16> inline maximum(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
  __m512 a_lo, a_hi;
  __m512 b_lo, b_hi;
  cvtbf16_fp32(__m512i(a), a_lo, a_hi);
  cvtbf16_fp32(__m512i(b), b_lo, b_hi);
  auto max_lo = _mm512_max_ps(a_lo, b_lo);
  auto max_hi = _mm512_max_ps(a_hi, b_hi);
  auto nan_lo_mask = _mm512_cmp_ps_mask(a_lo, b_lo, _CMP_UNORD_Q);
  auto nan_hi_mask = _mm512_cmp_ps_mask(a_hi, b_hi, _CMP_UNORD_Q);
  auto nan_lo = _mm512_castsi512_ps(_mm512_set1_epi32(nan_lo_mask));
  auto nan_hi = _mm512_castsi512_ps(_mm512_set1_epi32(nan_hi_mask));
  // Exploit the fact that all-ones is a NaN.
  auto o1 = _mm512_or_ps(max_lo, nan_lo);
  auto o2 = _mm512_or_ps(max_hi, nan_hi);
  return cvtfp32_bf16(o1, o2);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<BFloat16> inline minimum(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
  __m512 a_lo, a_hi;
  __m512 b_lo, b_hi;
  __m512i zero_vec = _mm512_set1_epi32(0);
  cvtbf16_fp32(__m512i(a), a_lo, a_hi);
  cvtbf16_fp32(__m512i(b), b_lo, b_hi);
  auto min_lo = _mm512_min_ps(a_lo, b_lo);
  auto min_hi = _mm512_min_ps(a_hi, b_hi);
  auto nan_lo_mask = _mm512_cmp_ps_mask(a_lo, b_lo, _CMP_UNORD_Q);
  auto nan_hi_mask = _mm512_cmp_ps_mask(a_hi, b_hi, _CMP_UNORD_Q);
  auto nan_lo = _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, nan_lo_mask,
                                                           0xFFFFFFFF));
  auto nan_hi = _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, nan_hi_mask,
                                                           0xFFFFFFFF));
  // Exploit the fact that all-ones is a NaN.
  auto o1 = _mm512_or_ps(min_lo, nan_lo);
  auto o2 = _mm512_or_ps(min_hi, nan_hi);
  return cvtfp32_bf16(o1, o2);
}

template <>
Vectorized<BFloat16> inline clamp(const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& min, const Vectorized<BFloat16>& max) {
  __m512 a_lo, a_hi;
  __m512 min_lo, min_hi;
  __m512 max_lo, max_hi;
  cvtbf16_fp32(__m512i(a), a_lo, a_hi);
  cvtbf16_fp32(__m512i(min), min_lo, min_hi);
  cvtbf16_fp32(__m512i(max), max_lo, max_hi);
  auto o1 = _mm512_min_ps(max_lo, _mm512_max_ps(min_lo, a_lo));
  auto o2 = _mm512_min_ps(max_hi, _mm512_max_ps(min_hi, a_hi));
  return cvtfp32_bf16(o1, o2);
}

template <>
Vectorized<BFloat16> inline clamp_max(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& max) {
  __m512 a_lo, a_hi;
  __m512 max_lo, max_hi;
  cvtbf16_fp32(__m512i(a), a_lo, a_hi);
  cvtbf16_fp32(__m512i(max), max_lo, max_hi);
  auto o1 = _mm512_min_ps(max_lo, a_lo);
  auto o2 = _mm512_min_ps(max_hi, a_hi);
  return cvtfp32_bf16(o1, o2);
}

template <>
Vectorized<BFloat16> inline clamp_min(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& min) {
  __m512 a_lo, a_hi;
  __m512 min_lo, min_hi;
  cvtbf16_fp32(__m512i(a), a_lo, a_hi);
  cvtbf16_fp32(__m512i(min), min_lo, min_hi);
  auto o1 = _mm512_max_ps(min_lo, a_lo);
  auto o2 = _mm512_max_ps(min_hi, a_hi);
  return cvtfp32_bf16(o1, o2);
}

template <>
inline void convert(const BFloat16* src, BFloat16* dst, int64_t n) {
  int64_t i;
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (i = 0; i <= (n - Vectorized<BFloat16>::size()); i += Vectorized<BFloat16>::size()) {
    auto vsrc = _mm512_loadu_si512(reinterpret_cast<__m512i*>((void*)(src + i)));
    _mm512_storeu_si512(reinterpret_cast<__m512i*>((void*)(dst + i)), vsrc);
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
  for (i = 0; i + Vectorized<BFloat16>::size() <= n; i += Vectorized<BFloat16>::size()) {
    __m512 a = _mm512_loadu_ps(&src[i]);
    __m512 b = _mm512_loadu_ps(&src[i + 16]);

    __m512i bf = cvtfp32_bf16(a, b);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(&dst[i]), bf);
  }
  for (; i < n; i++) {
    dst[i] = c10::convert<BFloat16>(src[i]);
  }
}

template <>
inline void convert(const double* src, BFloat16* dst, int64_t n) {
  auto load_float = [](const double *src) -> __m512 {
    // Load one float vector from an array of doubles
    __m256 a = _mm512_cvtpd_ps(_mm512_loadu_pd(src));
    __m256 b = _mm512_cvtpd_ps(_mm512_loadu_pd(src + 8));
    return _mm512_insertf32x8(_mm512_castps256_ps512(a), b, 1);
  };

  int64_t i;
  for (i = 0; i + Vectorized<BFloat16>::size() <= n; i += Vectorized<BFloat16>::size()) {
    __m512 a = load_float(&src[i]);
    __m512 b = load_float(&src[i + 16]);

    __m512i bf = cvtfp32_bf16(a, b);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(&dst[i]), bf);
  }
  for (; i < n; i++) {
    dst[i] = c10::convert<BFloat16>(src[i]);
  }
}

template <>
Vectorized<BFloat16> inline fmadd(const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b, const Vectorized<BFloat16>& c) {
  __m512 a_lo, a_hi;
  __m512 b_lo, b_hi;
  __m512 c_lo, c_hi;
  cvtbf16_fp32(__m512i(a), a_lo, a_hi);
  cvtbf16_fp32(__m512i(b), b_lo, b_hi);
  cvtbf16_fp32(__m512i(c), c_lo, c_hi);
  auto o1 = _mm512_fmadd_ps(a_lo, b_lo, c_lo);
  auto o2 = _mm512_fmadd_ps(a_hi, b_hi, c_hi);
  return cvtfp32_bf16(o1, o2);
}

static inline void _transpose_mxn_half_16_16(__m256i t[], __m512i u[]) {
  __m512i r[8];
  // a0a1 a2a3 a4a5 a6a7 a8a9 a10a11 a12a13 a14a15   e0e1 e2e3 e4e5 e6e7 e8e9 e10e11 e12e13 e14e15
  // b0-b15  f0-f15
  // c0-c15  g0-g15
  // d0-d15  h0-h15
  // i0-i15  m0-m15
  // j0-j15  n0-n15
  // k0-k15  o0-o15
  // l0-l15  p0-p15
#ifndef __msvc_cl__
#pragma unroll(4)
#endif
  for (int i = 0; i < 4; i++) {
    r[i] = _mm512_inserti64x4(_mm512_castsi256_si512(t[i]), t[i + 4], 0x01);
    r[i + 4] = _mm512_inserti64x4(_mm512_castsi256_si512(t[i + 8]), t[i + 12], 0x01);
  }

  // u0: a0a1 b0b1 a2a3 b2b3 a8a9 b8b9 a10a11 b10b11   e0e1 f0f1 e2e3 f2f3 e8e9 f8f9 e10e11 f10f11
  // u1: a4a5 b4b5 a6a7 b6b7 a12a13 b12b13 a14a15 b14b15   e4e5 f4f5 e6e7 f6f7 e12e13 f12f13 e14e15 f14f15
  // u2: c0c1 d0d1 c2c3 d2d3 c8c9 d8d9 c10c11 d10d11   g0g1 h0h1 g2g3 h2h3 g8g9 h8h9 g10g11 h10h11
  // u3: c4c5 d4b5 c6c7 d6b7 c12c13 d12d13 c14c15 d14d15   g4g5 h4h5 g6g7 h6h7 g12g13 h12h13 g14g15 h14h15
  // i j  m n
  // k l  o p
#ifndef __msvc_cl__
#pragma unroll(4)
#endif
  for (int i = 0; i < 8; i += 2) {
    u[i] = _mm512_unpacklo_epi32(r[i], r[i + 1]);
    u[i + 1] = _mm512_unpackhi_epi32(r[i], r[i + 1]);
  }

  // r0: a0a1 b0b1 c0c1 d0d1 a8a9 b8b9 c8c9 d8d9  e0e1 f0f1 g0g1 h0h1 e8e9 f8f9 g8g9 h8h9
  // r1: a2a3 b2b3 c2c3 d2d3 a10a11 b10b11 c10c11 d10d11  e2e3 f2f3 g2g3 h2h3 e10e11 f10f11 g10g11 h10h11
  // r2: a4a5 b4b5 c4c5 d4b5 a12a13 b12b13 c12c13 d12d13
  // r3: a6a7 b6b7 c6c7 d6b7 a14a15 b14b15 c14c15 d14d15
  // r4: i j k l m n o p
  r[0] = _mm512_unpacklo_epi64(u[0], u[2]);
  r[1] = _mm512_unpackhi_epi64(u[0], u[2]);
  r[2] = _mm512_unpacklo_epi64(u[1], u[3]);
  r[3] = _mm512_unpackhi_epi64(u[1], u[3]);
  r[4] = _mm512_unpacklo_epi64(u[4], u[6]);
  r[5] = _mm512_unpackhi_epi64(u[4], u[6]);
  r[6] = _mm512_unpacklo_epi64(u[5], u[7]);
  r[7] = _mm512_unpackhi_epi64(u[5], u[7]);

  __m512i const1 = _mm512_set_epi32(
      0x00370035,
      0x00330031,
      0x00270025,
      0x00230021,
      0x00170015,
      0x00130011,
      0x00070005,
      0x00030001,
      0x00360034,
      0x00320030,
      0x00260024,
      0x00220020,
      0x00160014,
      0x00120010,
      0x00060004,
      0x00020000);
  __m512i const2 = _mm512_set_epi32(
      0x003f003d,
      0x003b0039,
      0x002f002d,
      0x002b0029,
      0x001f001d,
      0x001b0019,
      0x000f000d,
      0x000b0009,
      0x003e003c,
      0x003a0038,
      0x002e002c,
      0x002a0028,
      0x001e001c,
      0x001a0018,
      0x000e000c,
      0x000a0008);
  // merge values from two regs
  // 0-- 1--
  // 8-- 9--
  // 2-- 3--
  // 10-- 11--
  // 4-- 5--
  // 12-- 13--
  // 6-- 7--
  // 14-- 15--
#ifndef __msvc_cl__
#pragma unroll(4)
#endif
  for (int i = 0; i < 4; i++) {
    u[i] = _mm512_permutex2var_epi16(r[i], const1, r[i + 4]);
    u[i + 4] = _mm512_permutex2var_epi16(r[i], const2, r[i + 4]);
  }
}

// TODO(Leslie): Add the AVX2 Version of transpose_mxn for BFloat16 and Float16
// Code referred to FBGEMM:
// https://github.com/pytorch/FBGEMM/blob/39a423e4ad1a04b77fea81c7d09c3e6f8984fae9/src/UtilsAvx512.cc#L1483-L1607
template<>
inline void transpose_mxn<BFloat16, 16, 16>(
    const BFloat16* src,
    int64_t ld_src,
    BFloat16* dst,
    int64_t ld_dst) {
  __m256i t[16];
  // load from src to registers
  // a: a0  a1  a2  a3  a4  a5  a6  a7  a8  a9  a10 a11 a12 a13 a14 a15
  // b: b0  b1  b2  b3  b4  b5  b6  b7  b8  b9  b10 b11 b12 b13 b14 b15
  // c: c0  c1  c2  c3  c4  c5  c6  c7  c8  c9  c10 c11 c12 c13 c14 c15
  // d: d0  d1  d2  d3  d4  d5  d6  d7  d8  d9  d10 d11 d12 d13 d14 d15
  // e: e0  e1  e2  e3  e4  e5  e6  e7  e8  e9  e10 e11 e12 e13 e14 e15
  // f: f0  f1  f2  f3  f4  f5  f6  f7  f8  f9  f10 f11 f12 f13 f14 f15
  // g: g0  g1  g2  g3  g4  g5  g6  g7  g8  g9  g10 g11 g12 g13 g14 g15
  // h: h0  h1  h2  h3  h4  h5  h6  h7  h8  h9  h10 h11 h12 h13 h14 h15
  // i: i0  i1  i2  i3  i4  i5  i6  i7  i8  i9  i10 i11 i12 i13 i14 i15
  // j: j0  j1  j2  j3  j4  j5  j6  j7  j8  j9  j10 j11 j12 j13 j14 j15
  // k: k0  k1  k2  k3  k4  k5  k6  k7  k8  k9  k10 k11 k12 k13 k14 k15
  // l: l0  l1  l2  l3  l4  l5  l6  l7  l8  l9  l10 l11 l12 l13 l14 l15
  // m: m0  m1  m2  m3  m4  m5  m6  m7  m8  m9  m10 m11 m12 m13 m14 m15
  // n: n0  n1  n2  n3  n4  n5  n6  n7  n8  n9  n10 n11 n12 n13 n14 n15
  // o: o0  o1  o2  o3  o4  o5  o6  o7  o8  o9  o10 o11 o12 o13 o14 o15
  // p: p0  p1  p2  p3  p4  p5  p6  p7  p8  p9  p10 p11 p12 p13 p14 p15
#ifndef __msvc_cl__
#pragma unroll(16)
#endif
  for (int i = 0; i < 16; i++) {
    t[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i * ld_src));
  }

  __m512i u[8];
  _transpose_mxn_half_16_16(t, u);

#ifndef __msvc_cl__
#pragma unroll(8)
#endif
  for (int i = 0; i < 8; i++) {
    _mm256_storeu_si256(
      reinterpret_cast<__m256i*>(dst + (i * 2) * ld_dst),
      _mm512_extracti32x8_epi32(u[i], 0x0));
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(dst + (i * 2 + 1) * ld_dst),
        _mm512_extracti32x8_epi32(u[i], 0x01));
  }
}

// Code referred to FBGEMM:
// https://github.com/pytorch/FBGEMM/blob/39a423e4ad1a04b77fea81c7d09c3e6f8984fae9/src/UtilsAvx512.cc#L1483-L1607
template<>
inline void transpose_mxn<Half, 16, 16>(
    const Half* src,
    int64_t ld_src,
    Half* dst,
    int64_t ld_dst) {
  __m256i t[16];
  // load from src to registers
  // Same matrix indices as above transpose_mxn<BFloat16, 16, 16>
#ifndef __msvc_cl__
#pragma unroll(16)
#endif
  for (int i = 0; i < 16; i++) {
    t[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i * ld_src));
  }

  __m512i u[8];
  _transpose_mxn_half_16_16(t, u);

#ifndef __msvc_cl__
#pragma unroll(8)
#endif
  for (int i = 0; i < 8; i++) {
    _mm256_storeu_si256(
      reinterpret_cast<__m256i*>(dst + (i * 2) * ld_dst),
      _mm512_extracti32x8_epi32(u[i], 0x0));
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(dst + (i * 2 + 1) * ld_dst),
        _mm512_extracti32x8_epi32(u[i], 0x01));
  }
}

static inline void _transpose_mxn_half_32_32(__m512i r[], __m512i d[]) {
  // t[0]: 0 32 1 33 2 34 3 35 8 40 9 41 10 42 11 43 16 ... 59
  // t[1]: 4 36 5 37 6 38 7 39 12 44 13 45 14 46 15 47 20 ... 63
  // t[2]: 64 96 65 97 66 98 67 99 72 104 73 105 74 106 75 ... 123
  // t[3]: 68 100 69 101 70 102 71 103 76 108 77 109 78 110 79 111 84 ... 127
  // t[4]: 128 160 129 161 130 162 131 163 136 168 137 169 138 170 139 171 144 ... 187
  // t[5]: 132 164 133 165 134 166 135 167 140 172 141 173 142 174 143 175 148 ... 191
  // t[6]: 192 224 193 225 194 226 195 227 200 232 201 233 202 234 203 235 208 ... 251
  // t[7]: 196 228 197 229 198 230 199 231 204 236 205 237 206 238 207 239 212 ... 255
  // t[8]: 256 288 257 289 258 290 259 291 264 296 265 297 266 298 267 299 272 ... 315
  // t[9]: 260 292 261 293 262 294 263 295 268 300 269 301 270 302 271 303 276 ... 319
  // t[10]: 320 352 321 353 322 354 323 355 328 360 329 361 330 362 331 363 336 ... 379
  // t[11]: 324 356 325 357 326 358 327 359 332 364 333 365 334 366 335 367 340 ... 383
  // t[12]: 384 416 385 417 386 418 387 419 392 424 393 425 394 426 395 427 400 ... 443
  // t[13]: 388 420 389 421 390 422 391 423 396 428 397 429 398 430 399 431 404 ... 447
  // t[14]: 448 480 449 481 450 482 451 483 456 488 457 489 458 490 459 491 464 ... 507
  // t[15]: 452 484 453 485 454 486 455 487 460 492 461 493 462 494 463 495 468 ... 511
  // t[16]: 512 544 513 545 514 546 515 547 520 552 521 553 522 554 523 555 528 ... 571
  // ...
  // t[31]: 964 996 965 997 966 998 967 999 972 1004 973 1005 974 1006 975 1007 980 ... 1023
#ifndef __msvc_cl__
#pragma unroll(16)
#endif
  for (int i = 0; i < 16; ++i) {
    d[i * 2] = _mm512_unpacklo_epi16(r[i * 2], r[i * 2 + 1]);
    d[i * 2 + 1] = _mm512_unpackhi_epi16(r[i * 2], r[i * 2 + 1]);
  }

  // t[0]: 0 32 64 96 1 33 65 97 8 40 72 104 9 41 73 105 16 ... 121
  // t[1]: 2 34 66 98 3 35 67 99 10 42 74 106 11 43 75 107 18 ... 123
  // t[2]: 4 36 68 100 5 37 69 101 12 44 76 108 13 45 77 109 20 ... 125
  // t[3]: 6 38 70 102 7 39 71 103 14 46 78 110 15 47 79 111 22 ... 127
  // t[4]: 128 160 192 224 129 161 193 225 136 168 200 232 137 169 201 233 144 ... 249
  // t[5]: 130 162 194 226 131 163 195 227 138 170 202 234 139 171 203 235 146 ... 251
  // t[6]: 132 164 196 228 133 165 197 229 140 172 204 236 141 173 205 237 148 ... 253
  // t[7]: 134 166 198 230 135 167 199 231 142 174 206 238 143 175 207 239 150 ... 255
  // t[8]: 256 288 320 352 257 289 321 353 264 296 328 360 265 297 329 361 272 ... 377
  // t[9]: 258 290 322 354 259 291 323 355 266 298 330 362 267 299 331 363 274 ... 379
  // t[10]: 260 292 324 356 261 293 325 357 268 300 332 364 269 301 333 365 276 ... 381
  // t[11]: 262 294 326 358 263 295 327 359 270 302 334 366 271 303 335 367 278 ... 383
  // t[12]: 384 416 448 480 385 417 449 481 392 424 456 488 393 425 457 489 400 ... 505
  // t[13]: 386 418 450 482 387 419 451 483 394 426 458 490 395 427 459 491 402 ... 507
  // t[14]: 388 420 452 484 389 421 453 485 396 428 460 492 397 429 461 493 404 ... 509
  // t[15]: 390 422 454 486 391 423 455 487 398 430 462 494 399 431 463 495 406 ... 511
  // t[16]: 512 544 576 608 513 545 577 609 520 552 584 616 521 553 585 617 528 ... 633
  // ...
  // t[31]: 902 934 966 998 903 935 967 999 910 942 974 1006 911 943 975 1007 918 ... 1023
#ifndef __msvc_cl__
#pragma unroll(8)
#endif
  for (int i = 0; i < 8; ++i) {
    r[i * 4] = _mm512_unpacklo_epi32(d[i * 4], d[i * 4 + 2]);
    r[i * 4 + 1] = _mm512_unpackhi_epi32(d[i * 4], d[i * 4 + 2]);
    r[i * 4 + 2] = _mm512_unpacklo_epi32(d[i * 4 + 1], d[i * 4 + 3]);
    r[i * 4 + 3] = _mm512_unpackhi_epi32(d[i * 4 + 1], d[i * 4 + 3]);
  }

  // t[0]: 0 32 64 96 128 160 192 224 8 40 72 104 136 168 200 232 16 ... 248
  // t[1]: 1 33 65 97 129 161 193 225 9 41 73 105 137 169 201 233 17 ... 249
  // t[2]: 2 34 66 98 130 162 194 226 10 42 74 106 138 170 202 234 18 ... 250
  // t[3]: 3 35 67 99 131 163 195 227 11 43 75 107 139 171 203 235 19 ... 251
  // t[4]: 4 36 68 100 132 164 196 228 12 44 76 108 140 172 204 236 20 ... 252
  // t[5]: 5 37 69 101 133 165 197 229 13 45 77 109 141 173 205 237 21 ... 253
  // t[6]: 6 38 70 102 134 166 198 230 14 46 78 110 142 174 206 238 22 ... 254
  // t[7]: 7 39 71 103 135 167 199 231 15 47 79 111 143 175 207 239 23 ... 255
  // t[8]: 256 288 320 352 384 416 448 480 264 296 328 360 392 424 456 488 272 ... 504
  // t[9]: 257 289 321 353 385 417 449 481 265 297 329 361 393 425 457 489 273 ... 505
  // t[10]: 258 290 322 354 386 418 450 482 266 298 330 362 394 426 458 490 274 ... 506
  // t[11]: 259 291 323 355 387 419 451 483 267 299 331 363 395 427 459 491 275 ... 507
  // t[12]: 260 292 324 356 388 420 452 484 268 300 332 364 396 428 460 492 276 ... 508
  // t[13]: 261 293 325 357 389 421 453 485 269 301 333 365 397 429 461 493 277 ... 509
  // t[14]: 262 294 326 358 390 422 454 486 270 302 334 366 398 430 462 494 278 ... 510
  // t[15]: 263 295 327 359 391 423 455 487 271 303 335 367 399 431 463 495 279 ... 511
  // t[16]: 512 544 576 608 640 672 704 736 520 552 584 616 648 680 712 744 528 ... 760
  // ...
  // t[31]: 775 807 839 871 903 935 967 999 783 815 847 879 911 943 975 1007 791 ... 1023
#ifndef __msvc_cl__
#pragma unroll(4)
#endif
  for (int i = 0; i < 4; ++i) {
    d[i * 8] = _mm512_unpacklo_epi64(r[i * 8], r[i * 8 + 4]);
    d[i * 8 + 1] = _mm512_unpackhi_epi64(r[i * 8], r[i * 8 + 4]);
    d[i * 8 + 2] = _mm512_unpacklo_epi64(r[i * 8 + 1], r[i * 8 + 5]);
    d[i * 8 + 3] = _mm512_unpackhi_epi64(r[i * 8 + 1], r[i * 8 + 5]);
    d[i * 8 + 4] = _mm512_unpacklo_epi64(r[i * 8 + 2], r[i * 8 + 6]);
    d[i * 8 + 5] = _mm512_unpackhi_epi64(r[i * 8 + 2], r[i * 8 + 6]);
    d[i * 8 + 6] = _mm512_unpacklo_epi64(r[i * 8 + 3], r[i * 8 + 7]);
    d[i * 8 + 7] = _mm512_unpackhi_epi64(r[i * 8 + 3], r[i * 8 + 7]);
  }

  // t[0]: 0 32 64 96 128 160 192 224 256 288 320 352 384 416 448 480 16 ... 496
  // t[1]: 1 33 65 97 129 161 193 225 257 289 321 353 385 417 449 481 17 ... 497
  // t[2]: 2 34 66 98 130 162 194 226 258 290 322 354 386 418 450 482 18 ... 498
  // t[3]: 3 35 67 99 131 163 195 227 259 291 323 355 387 419 451 483 19 ... 499
  // t[4]: 4 36 68 100 132 164 196 228 260 292 324 356 388 420 452 484 20 ... 500
  // t[5]: 5 37 69 101 133 165 197 229 261 293 325 357 389 421 453 485 21 ... 501
  // t[6]: 6 38 70 102 134 166 198 230 262 294 326 358 390 422 454 486 22 ... 502
  // t[7]: 7 39 71 103 135 167 199 231 263 295 327 359 391 423 455 487 23 ... 503
  // t[8]: 8 40 72 104 136 168 200 232 264 296 328 360 392 424 456 488 24 ... 504
  // t[9]: 9 41 73 105 137 169 201 233 265 297 329 361 393 425 457 489 25 ... 505
  // t[10]: 10 42 74 106 138 170 202 234 266 298 330 362 394 426 458 490 26 ... 506
  // t[11]: 11 43 75 107 139 171 203 235 267 299 331 363 395 427 459 491 27 ... 507
  // t[12]: 12 44 76 108 140 172 204 236 268 300 332 364 396 428 460 492 28 ... 508
  // t[13]: 13 45 77 109 141 173 205 237 269 301 333 365 397 429 461 493 29 ... 509
  // t[14]: 14 46 78 110 142 174 206 238 270 302 334 366 398 430 462 494 30 ... 510
  // t[15]: 15 47 79 111 143 175 207 239 271 303 335 367 399 431 463 495 31 ... 511
  // t[16]: 512 544 576 608 640 672 704 736 768 800 832 864 896 928 960 992 528 ... 1008
  // ...
  // t[31]: 527 559 591 623 655 687 719 751 783 815 847 879 911 943 975 1007 543 ... 1023
  __m512i const1 = _mm512_set_epi64(
      0x000000000000000d,
      0x000000000000000c,
      0x0000000000000005,
      0x0000000000000004,
      0x0000000000000009,
      0x0000000000000008,
      0x0000000000000001,
      0x0000000000000000);
  __m512i const2 = _mm512_set_epi64(
      0x000000000000000f,
      0x000000000000000e,
      0x0000000000000007,
      0x0000000000000006,
      0x000000000000000b,
      0x000000000000000a,
      0x0000000000000003,
      0x0000000000000002);
#ifndef __msvc_cl__
#pragma unroll(8)
#endif
  for (int i = 0; i < 8; ++i) {
    r[i] = _mm512_permutex2var_epi64(d[i], /*idx*/const1, d[i + 8]);
    r[i + 8] = _mm512_permutex2var_epi64(d[i], /*idx*/const2, d[i + 8]);
    r[i + 16] = _mm512_permutex2var_epi64(d[i + 16], /*idx*/const1, d[i + 24]);
    r[i + 24] = _mm512_permutex2var_epi64(d[i + 16], /*idx*/const2, d[i + 24]);
  }

  // t[0]: 0 32 64 96 128 160 192 224 256 288 320 352 384 416 448 480 512 544 ... 992
  // t[1]: 1 33 65 97 129 161 193 225 257 289 321 353 385 417 449 481 513 545 ... 993
  // t[2]: 2 34 66 98 130 162 194 226 258 290 322 354 386 418 450 482 514 546 ... 994
  // t[3]: 3 35 67 99 131 163 195 227 259 291 323 355 387 419 451 483 515 547 ... 995
  // t[4]: 4 36 68 100 132 164 196 228 260 292 324 356 388 420 452 484 516 548 ... 996
  // t[5]: 5 37 69 101 133 165 197 229 261 293 325 357 389 421 453 485 517 549 ... 997
  // t[6]: 6 38 70 102 134 166 198 230 262 294 326 358 390 422 454 486 518 550 ... 998
  // t[7]: 7 39 71 103 135 167 199 231 263 295 327 359 391 423 455 487 519 551 ... 999
  // t[8]: 8 40 72 104 136 168 200 232 264 296 328 360 392 424 456 488 520 552 ... 1000
  // t[9]: 9 41 73 105 137 169 201 233 265 297 329 361 393 425 457 489 521 553 ... 1001
  // t[10]: 10 42 74 106 138 170 202 234 266 298 330 362 394 426 458 490 522 554 ... 1002
  // t[11]: 11 43 75 107 139 171 203 235 267 299 331 363 395 427 459 491 523 555 ... 1003
  // t[12]: 12 44 76 108 140 172 204 236 268 300 332 364 396 428 460 492 524 556 ... 1004
  // t[13]: 13 45 77 109 141 173 205 237 269 301 333 365 397 429 461 493 525 557 ... 1005
  // t[14]: 14 46 78 110 142 174 206 238 270 302 334 366 398 430 462 494 526 558 ... 1006
  // t[15]: 15 47 79 111 143 175 207 239 271 303 335 367 399 431 463 495 527 559 ... 1007
  // t[16]: 16 48 80 112 144 176 208 240 272 304 336 368 400 432 464 496 528 560 ... 1008
  // ...
  // t[31]: 31 63 95 127 159 191 223 255 287 319 351 383 415 447 479 511 543 575 ... 1023
  __m512i const3 = _mm512_set_epi64(
      0x000000000000000b,
      0x000000000000000a,
      0x0000000000000009,
      0x0000000000000008,
      0x0000000000000003,
      0x0000000000000002,
      0x0000000000000001,
      0x0000000000000000);
  __m512i const4 = _mm512_set_epi64(
      0x000000000000000f,
      0x000000000000000e,
      0x000000000000000d,
      0x000000000000000c,
      0x0000000000000007,
      0x0000000000000006,
      0x0000000000000005,
      0x0000000000000004);
#ifndef __msvc_cl__
#pragma unroll(16)
#endif
  for (int i = 0; i < 16; ++i) {
    d[i] = _mm512_permutex2var_epi64(r[i], /*idx*/const3, r[i + 16]);
    d[i + 16] = _mm512_permutex2var_epi64(r[i], /*idx*/const4, r[i + 16]);
  }
}

// Code referred to FBGEMM:
// https://github.com/pytorch/FBGEMM/blob/39a423e4ad1a04b77fea81c7d09c3e6f8984fae9/src/UtilsAvx512.cc#LL19C6-L19C6
template<>
inline void transpose_mxn<BFloat16, 32, 32>(
    const BFloat16* src,
    int64_t ld_src,
    BFloat16* dst,
    int64_t ld_dst) {
  // Load from memory
  __m512i r[32];
#ifndef __msvc_cl__
#pragma unroll(32)
#endif
  for (int i = 0; i < 32; ++i) {
    r[i] = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + i* ld_src));
  }

  __m512i d[32];
  _transpose_mxn_half_32_32(r, d);

  // Store to dst
#ifndef __msvc_cl__
#pragma unroll(32)
#endif
  for (int i = 0; i < 32; ++i) {
    _mm512_storeu_si512(dst + i* ld_dst, d[i]);
  }
}

template <typename T, int M, int N,
          typename std::enable_if_t<std::is_same<T, BFloat16>::value && ((M < 32 && M != 16) || (N < 32 && N != 16)), int> = 0>
inline void transpose_mxn(const BFloat16* src, int64_t ld_src, BFloat16* dst, int64_t ld_dst) {
  // load from src
  __mmask32 src_mask = (1 << N) - 1;
  __m512i r[32];
  int i;
  for (i = 0; i < M; ++i) {
    r[i] = _mm512_maskz_loadu_epi16(src_mask, &src[i * ld_src]);
  }
  for (; i < 32; ++i) {
    r[i] = _mm512_setzero_si512();
  }

  __m512i d[32];
  _transpose_mxn_half_32_32(r, d);

  // store to dst
  __mmask32 dst_mask = (1 << M) - 1;
  for (i = 0; i < N; ++i) {
    _mm512_mask_storeu_epi16(&dst[i * ld_dst], dst_mask, d[i]);
  }
}

template<>
inline void transpose_mxn<Half, 32, 32>(
    const Half* src,
    int64_t ld_src,
    Half* dst,
    int64_t ld_dst) {
  // Load from memory
  __m512i r[32];
#ifndef __msvc_cl__
#pragma unroll(32)
#endif
  for (int i = 0; i < 32; ++i) {
    r[i] = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + i* ld_src));
  }

  __m512i d[32];
  _transpose_mxn_half_32_32(r, d);

  // Store to dst
#ifndef __msvc_cl__
#pragma unroll(32)
#endif
  for (int i = 0; i < 32; ++i) {
    _mm512_storeu_si512(dst + i* ld_dst, d[i]);
  }
}

template <typename T, int M, int N,
          typename std::enable_if_t<std::is_same<T, Half>::value && ((M < 32 && M != 16) || (N < 32 && N != 16)), int> = 0>
inline void transpose_mxn(const Half* src, int64_t ld_src, Half* dst, int64_t ld_dst) {
  // load from src
  __mmask32 src_mask = (1 << N) - 1;
  __m512i r[32];
  int i;
  for (i = 0; i < M; ++i) {
    r[i] = _mm512_maskz_loadu_epi16(src_mask, &src[i * ld_src]);
  }
  for (; i < 32; ++i) {
    r[i] = _mm512_setzero_si512();
  }

  __m512i d[32];
  _transpose_mxn_half_32_32(r, d);

  // store to dst
  __mmask32 dst_mask = (1 << M) - 1;
  for (i = 0; i < N; ++i) {
    _mm512_mask_storeu_epi16(&dst[i * ld_dst], dst_mask, d[i]);
  }
}

template <>
class Vectorized<Half>: public Vectorized16<Half> {
public:
  using Vectorized16::Vectorized16;

  Vectorized<Half> frac() const;

  Vectorized<Half> eq(const Vectorized<Half>& other) const;
  Vectorized<Half> ne(const Vectorized<Half>& other) const;
  Vectorized<Half> gt(const Vectorized<Half>& other) const;
  Vectorized<Half> ge(const Vectorized<Half>& other) const;
  Vectorized<Half> lt(const Vectorized<Half>& other) const;
  Vectorized<Half> le(const Vectorized<Half>& other) const;
};

Vectorized<Half> inline operator+(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  return binary_op_as_fp32(a, b, [](const __m512& x, const __m512& y) { return _mm512_add_ps(x, y); });
}
Vectorized<Half> inline operator-(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  return binary_op_as_fp32(a, b, [](const __m512& x, const __m512& y) { return _mm512_sub_ps(x, y); });
}
Vectorized<Half> inline operator*(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  return binary_op_as_fp32(a, b, [](const __m512& x, const __m512& y) { return _mm512_mul_ps(x, y); });
}
Vectorized<Half> inline operator/(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  return binary_op_as_fp32(a, b, [](const __m512& x, const __m512& y) { return _mm512_div_ps(x, y); });
}

Vectorized<Half> inline operator&(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  return _mm512_and_si512(a, b);
}
Vectorized<Half> inline operator|(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  return _mm512_or_si512(a, b);
}
Vectorized<Half> inline operator^(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  return _mm512_xor_si512(a, b);
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
  __m512 a_lo, a_hi;
  __m512 b_lo, b_hi;
  cvtfp16_fp32(__m512i(a), a_lo, a_hi);
  cvtfp16_fp32(__m512i(b), b_lo, b_hi);
  auto max_lo = _mm512_max_ps(a_lo, b_lo);
  auto max_hi = _mm512_max_ps(a_hi, b_hi);
  auto nan_lo_mask = _mm512_cmp_ps_mask(a_lo, b_lo, _CMP_UNORD_Q);
  auto nan_hi_mask = _mm512_cmp_ps_mask(a_hi, b_hi, _CMP_UNORD_Q);
  auto nan_lo = _mm512_castsi512_ps(_mm512_set1_epi32(nan_lo_mask));
  auto nan_hi = _mm512_castsi512_ps(_mm512_set1_epi32(nan_hi_mask));
  // Exploit the fact that all-ones is a NaN.
  auto o1 = _mm512_or_ps(max_lo, nan_lo);
  auto o2 = _mm512_or_ps(max_hi, nan_hi);
  return cvtfp32_fp16(o1, o2);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<Half> inline minimum(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  __m512 a_lo, a_hi;
  __m512 b_lo, b_hi;
  __m512i zero_vec = _mm512_set1_epi32(0);
  cvtfp16_fp32(__m512i(a), a_lo, a_hi);
  cvtfp16_fp32(__m512i(b), b_lo, b_hi);
  auto min_lo = _mm512_min_ps(a_lo, b_lo);
  auto min_hi = _mm512_min_ps(a_hi, b_hi);
  auto nan_lo_mask = _mm512_cmp_ps_mask(a_lo, b_lo, _CMP_UNORD_Q);
  auto nan_hi_mask = _mm512_cmp_ps_mask(a_hi, b_hi, _CMP_UNORD_Q);
  auto nan_lo = _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, nan_lo_mask,
                                                           0xFFFFFFFF));
  auto nan_hi = _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, nan_hi_mask,
                                                           0xFFFFFFFF));
  // Exploit the fact that all-ones is a NaN.
  auto o1 = _mm512_or_ps(min_lo, nan_lo);
  auto o2 = _mm512_or_ps(min_hi, nan_hi);
  return cvtfp32_fp16(o1, o2);
}

template <>
Vectorized<Half> inline clamp(const Vectorized<Half>& a,
    const Vectorized<Half>& min, const Vectorized<Half>& max) {
  __m512 a_lo, a_hi;
  __m512 min_lo, min_hi;
  __m512 max_lo, max_hi;
  cvtfp16_fp32(__m512i(a), a_lo, a_hi);
  cvtfp16_fp32(__m512i(min), min_lo, min_hi);
  cvtfp16_fp32(__m512i(max), max_lo, max_hi);
  auto o1 = _mm512_min_ps(max_lo, _mm512_max_ps(min_lo, a_lo));
  auto o2 = _mm512_min_ps(max_hi, _mm512_max_ps(min_hi, a_hi));
  return cvtfp32_fp16(o1, o2);
}

template <>
Vectorized<Half> inline clamp_max(const Vectorized<Half>& a, const Vectorized<Half>& max) {
  __m512 a_lo, a_hi;
  __m512 max_lo, max_hi;
  cvtfp16_fp32(__m512i(a), a_lo, a_hi);
  cvtfp16_fp32(__m512i(max), max_lo, max_hi);
  auto o1 = _mm512_min_ps(max_lo, a_lo);
  auto o2 = _mm512_min_ps(max_hi, a_hi);
  return cvtfp32_fp16(o1, o2);
}

template <>
Vectorized<Half> inline clamp_min(const Vectorized<Half>& a, const Vectorized<Half>& min) {
  __m512 a_lo, a_hi;
  __m512 min_lo, min_hi;
  cvtfp16_fp32(__m512i(a), a_lo, a_hi);
  cvtfp16_fp32(__m512i(min), min_lo, min_hi);
  auto o1 = _mm512_max_ps(min_lo, a_lo);
  auto o2 = _mm512_max_ps(min_hi, a_hi);
  return cvtfp32_fp16(o1, o2);
}

template <>
inline void convert(const Half* src, Half* dst, int64_t n) {
  int64_t i;
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (i = 0; i <= (n - Vectorized<Half>::size()); i += Vectorized<Half>::size()) {
    auto vsrc = _mm512_loadu_si512(reinterpret_cast<__m512i*>((void*)(src + i)));
    _mm512_storeu_si512(reinterpret_cast<__m512i*>((void*)(dst + i)), vsrc);
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
    __m512 a = _mm512_loadu_ps(&src[i]);
    __m512 b = _mm512_loadu_ps(&src[i + 16]);

    __m512i bf = cvtfp32_fp16(a, b);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(&dst[i]), bf);
  }
  for (; i < n; i++) {
    dst[i] = c10::convert<Half>(src[i]);
  }
}

template <>
inline void convert(const double* src, Half* dst, int64_t n) {
  auto load_float = [](const double *src) -> __m512 {
    // Load one float vector from an array of doubles
    __m256 a = _mm512_cvtpd_ps(_mm512_loadu_pd(src));
    __m256 b = _mm512_cvtpd_ps(_mm512_loadu_pd(src + 8));
    return _mm512_insertf32x8(_mm512_castps256_ps512(a), b, 1);
  };

  int64_t i;
  for (i = 0; i + Vectorized<Half>::size() <= n; i += Vectorized<Half>::size()) {
    __m512 a = load_float(&src[i]);
    __m512 b = load_float(&src[i + 16]);

    __m512i bf = cvtfp32_fp16(a, b);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(&dst[i]), bf);
  }
  for (; i < n; i++) {
    dst[i] = c10::convert<Half>(src[i]);
  }
}

template <>
Vectorized<Half> inline fmadd(const Vectorized<Half>& a,
    const Vectorized<Half>& b, const Vectorized<Half>& c) {
  __m512 a_lo, a_hi;
  __m512 b_lo, b_hi;
  __m512 c_lo, c_hi;
  cvtfp16_fp32(__m512i(a), a_lo, a_hi);
  cvtfp16_fp32(__m512i(b), b_lo, b_hi);
  cvtfp16_fp32(__m512i(c), c_lo, c_hi);
  auto o1 = _mm512_fmadd_ps(a_lo, b_lo, c_lo);
  auto o2 = _mm512_fmadd_ps(a_hi, b_hi, c_hi);
  return cvtfp32_fp16(o1, o2);
}

#define CONVERT_VECTORIZED_INIT(type, name) \
inline std::tuple<Vectorized<float>, Vectorized<float>> convert_##name##_float(const Vectorized<type>& a) { \
  __m512 o1, o2; \
  cvt_to_fp32<type>(__m512i(a), o1, o2); \
  return std::make_tuple(o1, o2); \
} \
\
inline Vectorized<type> convert_float_##name(const Vectorized<float>& a, const Vectorized<float>& b) { \
 return cvt_from_fp32<type>(__m512(a), __m512(b)); \
}
CONVERT_VECTORIZED_INIT(BFloat16, bfloat16);
CONVERT_VECTORIZED_INIT(Half, half);

#else //defined(CPU_CAPABILITY_AVX512)

#define CONVERT_NON_VECTORIZED_INIT(type, name) \
inline std::tuple<Vectorized<float>, Vectorized<float>> convert_##name##_float(const Vectorized<type>& a) { \
  constexpr int64_t K = Vectorized<type>::size(); \
  __at_align__ float arr[K]; \
  __at_align__ type arr2[K]; \
  a.store(arr2); \
  for (const auto k : c10::irange(K)) { \
    arr[k] = c10::convert<float>(arr2[k]); \
  } \
  return std::make_tuple( \
      Vectorized<float>::loadu(arr), \
      Vectorized<float>::loadu(arr + Vectorized<float>::size())); \
} \
\
inline Vectorized<type> convert_float_##name(const Vectorized<float>& a, const Vectorized<float>& b) { \
  constexpr int64_t K = Vectorized<type>::size(); \
  __at_align__ float arr[K]; \
  __at_align__ type arr2[K]; \
  a.store(arr); \
  b.store(arr + Vectorized<float>::size()); \
  for (const auto k : c10::irange(K)) { \
    arr2[k] = c10::convert<type>(arr[k]); \
  } \
  return Vectorized<type>::loadu(arr2); \
}
CONVERT_NON_VECTORIZED_INIT(BFloat16, bfloat16);
CONVERT_NON_VECTORIZED_INIT(Half, half);

#endif // defined(CPU_CAPABILITY_AVX512)

#if defined(CPU_CAPABILITY_AVX512)
#define LOAD_FP32_VECTORIZED_INIT(type, name) \
inline void load_fp32_from_##name(const type *data, Vectorized<float>& out) { \
  auto values = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data)); \
  __m512 out_values; \
  cvt_to_fp32<type>(values, out_values); \
  out = out_values; \
} \
\
inline void load_fp32_from_##name(const type *data, Vectorized<float>& out1, Vectorized<float>& out2) { \
  auto vec = Vectorized<type>::loadu(data); \
  __m512 out1_values, out2_values; \
  cvt_to_fp32<type>(vec, out1_values, out2_values); \
  out1 = out1_values; \
  out2 = out2_values; \
}
LOAD_FP32_VECTORIZED_INIT(BFloat16, bf16);
LOAD_FP32_VECTORIZED_INIT(Half, fp16);

#else // defined(CPU_CAPABILITY_AVX512)
#define LOAD_FP32_NON_VECTORIZED_INIT(type, name) \
inline void load_fp32_from_##name(const type *data, Vectorized<float>& out) { \
  __at_align__ float values[Vectorized<float>::size()]; \
  for (const auto k : c10::irange(Vectorized<float>::size())) { \
    values[k] = data[k]; \
  } \
  out = Vectorized<float>::loadu(values); \
} \
\
inline void load_fp32_from_##name(const type *data, Vectorized<float>& out1, Vectorized<float>& out2) { \
  load_fp32_from_##name(data, out1); \
  data += Vectorized<float>::size(); \
  load_fp32_from_##name(data, out2); \
}
LOAD_FP32_NON_VECTORIZED_INIT(BFloat16, bf16);
LOAD_FP32_NON_VECTORIZED_INIT(Half, fp16);

#endif
}}}
