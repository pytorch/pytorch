#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/util/irange.h>

#if defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)
#include <sleef.h>
#endif

namespace at {
namespace vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)

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

    __at_align__ int16_t tmp_values[size()];
    std::memcpy(tmp_values, ptr, count * sizeof(int16_t));
    return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(tmp_values));
  }
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), values);
    } else if (count > 0) {
      __at_align__ int16_t tmp_values[size()];
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(int16_t));
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
  Vectorized<T> map(const __m512 (*const vop)(__m512)) const {
    __m512 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    const auto o1 = vop(lo);
    const auto o2 = vop(hi);
    return cvt_from_fp32<T>(o1, o2);
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
#pragma unroll
  for (i = 0; i <= (n - Vectorized<BFloat16>::size()); i += Vectorized<BFloat16>::size()) {
    auto vsrc = _mm512_loadu_si512(reinterpret_cast<__m512i*>((void*)(src + i)));
    _mm512_storeu_si512(reinterpret_cast<__m512i*>((void*)(dst + i)), vsrc);
  }
#pragma unroll
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
#pragma unroll
  for (i = 0; i <= (n - Vectorized<Half>::size()); i += Vectorized<Half>::size()) {
    auto vsrc = _mm512_loadu_si512(reinterpret_cast<__m512i*>((void*)(src + i)));
    _mm512_storeu_si512(reinterpret_cast<__m512i*>((void*)(dst + i)), vsrc);
  }
#pragma unroll
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

#else //defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)

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

#endif // defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)

#if defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)
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

#else // defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)
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
