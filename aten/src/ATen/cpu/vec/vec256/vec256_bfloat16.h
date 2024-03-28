#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/util/irange.h>

#if defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)
#include <sleef.h>
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"

namespace at::vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)

// bfloat16 conversion
static inline void cvtbf16_fp32(const __m128i& a, __m256& o) {
  o = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(a), 16));
}

static inline void cvtbf16_fp32(const __m256i& a, __m256& o1, __m256& o2) {
  __m128i lo = _mm256_extractf128_si256(a, 0);
  __m128i hi = _mm256_extractf128_si256(a, 1);
  cvtbf16_fp32(lo, o1);
  cvtbf16_fp32(hi, o2);
}

static inline __m128i cvtfp32_bf16(const __m256& src) {
  __m256i value = _mm256_castps_si256(src);
  __m256i nan = _mm256_set1_epi32(0xffff);
  __m256i mask = _mm256_castps_si256(_mm256_cmp_ps(src, src, _CMP_ORD_Q));
  __m256i ones = _mm256_set1_epi32(0x1);
  __m256i vec_bias = _mm256_set1_epi32(0x7fff);
  // uint32_t lsb = (input >> 16) & 1;
  auto t_value = _mm256_and_si256(_mm256_srli_epi32(value, 16), ones);
  // uint32_t rounding_bias = 0x7fff + lsb;
  t_value = _mm256_add_epi32(t_value, vec_bias);
  // input += rounding_bias;
  t_value = _mm256_add_epi32(t_value, value);
  // input = input >> 16;
  t_value = _mm256_srli_epi32(t_value, 16);
  // Check NaN before converting back to bf16
  t_value = _mm256_blendv_epi8(nan, t_value, mask);
  t_value = _mm256_packus_epi32(t_value, t_value);   // t[4-7] t[4-7] t[0-4] t[0-4]
  t_value = _mm256_permute4x64_epi64(t_value, 0xd8); // 11     01     10     00
  return _mm256_castsi256_si128(t_value);
}

static inline __m256i cvtfp32_bf16(const __m256& a, const __m256& b) {
  __m256i lo = _mm256_castps_si256(a);
  __m256i hi = _mm256_castps_si256(b);
  __m256i nan = _mm256_set1_epi32(0xffff);
  __m256i mask_lo = _mm256_castps_si256(_mm256_cmp_ps(a, a, _CMP_ORD_Q));
  __m256i mask_hi = _mm256_castps_si256(_mm256_cmp_ps(b, b, _CMP_ORD_Q));
  __m256i ones = _mm256_set1_epi32(0x1);
  __m256i vec_bias = _mm256_set1_epi32(0x7fff);
  // uint32_t lsb = (input >> 16) & 1;
  auto t_lo = _mm256_and_si256(_mm256_srli_epi32(lo, 16), ones);
  auto t_hi = _mm256_and_si256(_mm256_srli_epi32(hi, 16), ones);
  // uint32_t rounding_bias = 0x7fff + lsb;
  t_lo = _mm256_add_epi32(t_lo, vec_bias);
  t_hi = _mm256_add_epi32(t_hi, vec_bias);
  // input += rounding_bias;
  t_lo = _mm256_add_epi32(t_lo, lo);
  t_hi = _mm256_add_epi32(t_hi, hi);
  // input = input >> 16;
  t_lo = _mm256_srli_epi32(t_lo, 16);
  t_hi = _mm256_srli_epi32(t_hi, 16);
  // Check NaN before converting back to bf16
  t_lo = _mm256_blendv_epi8(nan, t_lo, mask_lo);
  t_hi = _mm256_blendv_epi8(nan, t_hi, mask_hi);

  t_lo = _mm256_packus_epi32(t_lo, t_hi);      // t_hi[4-7] t_lo[4-7] t_hi[0-4] t_lo[0-4]
  return _mm256_permute4x64_epi64(t_lo, 0xd8); // 11        01        10        00
}

static inline __m256i merge_compare_result(const __m256& a, const __m256& b) {
  __m256i lo = _mm256_castps_si256(a);
  __m256i hi = _mm256_castps_si256(b);
  lo = _mm256_srli_epi32(lo, 16);
  hi = _mm256_srli_epi32(hi, 16);
  auto out = _mm256_packus_epi32(lo, hi);
  return _mm256_permute4x64_epi64(out, 0xd8);
}

// float16 conversion
static inline void cvtfp16_fp32(const __m128i& a, __m256& o) {
  o = _mm256_cvtph_ps(a);
}

static inline void cvtfp16_fp32(const __m256i& a, __m256& o1, __m256& o2) {
  __m128i lo = _mm256_extractf128_si256(a, 0);
  __m128i hi = _mm256_extractf128_si256(a, 1);
  cvtfp16_fp32(lo, o1);
  cvtfp16_fp32(hi, o2);
}

static inline __m128i cvtfp32_fp16(const __m256& src) {
  return _mm256_cvtps_ph(
      src, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

static inline __m256i cvtfp32_fp16(const __m256& a, const __m256& b) {
  __m128i lo = _mm256_cvtps_ph(
      a, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  __m128i hi = _mm256_cvtps_ph(
      b, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  return _mm256_insertf128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

// dtype conversion between float16/bfloat16 and float32
template <typename T, typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline void cvt_to_fp32(const __m128i& a, __m256& o);
template <> inline void cvt_to_fp32<BFloat16>(const __m128i& a, __m256& o) {
  cvtbf16_fp32(a, o);
};
template <> inline void cvt_to_fp32<Half>(const __m128i& a, __m256& o) {
  cvtfp16_fp32(a, o);
}

template <typename T, typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline void cvt_to_fp32(const __m256i& a, __m256& o1, __m256& o2);
template <> inline void cvt_to_fp32<BFloat16>(const __m256i& a, __m256& o1, __m256& o2) {
  cvtbf16_fp32(a, o1, o2);
}
template <> inline void cvt_to_fp32<Half>(const __m256i& a, __m256& o1, __m256& o2) {
  cvtfp16_fp32(a, o1, o2);
}

template <typename T, bool is_compare_op = false,
          typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline __m256i cvt_from_fp32(const __m256& a, const __m256& b);
template <> inline __m256i cvt_from_fp32<BFloat16, false>(const __m256& a, const __m256& b) {
  return cvtfp32_bf16(a, b);
}
template <> inline __m256i cvt_from_fp32<BFloat16, true>(const __m256& a, const __m256& b) {
  return merge_compare_result(a, b);
}
template <> inline __m256i cvt_from_fp32<Half, false>(const __m256& a, const __m256& b) {
  return cvtfp32_fp16(a, b);
}
template <> inline __m256i cvt_from_fp32<Half, true>(const __m256& a, const __m256& b) {
  return cvtfp32_fp16(a, b);
}

template <typename T>
class Vectorized16 {
static_assert(
  is_reduced_floating_point_v<T>,
  "Support only float16 and bfloat16.");
protected:
  __m256i values;
public:
  using value_type = uint16_t;
  using size_type = int;
  static constexpr size_type size() {
    return 16;
  }
  Vectorized16() {}
  Vectorized16(__m256i v) : values(v) {}
  Vectorized16(T val) {
    value_type uw = val.x;
    values = _mm256_set1_epi16(uw);
  }
  Vectorized16(T val1, T val2, T val3, T val4,
         T val5, T val6, T val7, T val8,
         T val9, T val10, T val11, T val12,
         T val13, T val14, T val15, T val16) {
    values = _mm256_setr_epi16(
        val1.x, val2.x, val3.x, val4.x, val5.x, val6.x, val7.x, val8.x,
        val9.x, val10.x, val11.x, val12.x, val13.x, val14.x, val15.x, val16.x);
  }
  operator __m256i() const {
    return values;
  }
  T& operator[](int idx) = delete;
  const T& operator[](int idx) const  = delete;
  int zero_mask() const {
    // returns an integer mask where all zero elements are translated to 1-bit and others are translated to 0-bit
    __m256i cmp = _mm256_cmpeq_epi16(values, _mm256_set1_epi16(0));
    return _mm256_movemask_epi8(cmp);
  }
  static Vectorized<T> loadu(const void* ptr, int16_t count = size()) {
    if (count == size())
      return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));

    __at_align__ int16_t tmp_values[size()];
    std::memcpy(tmp_values, ptr, count * sizeof(int16_t));
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tmp_values));
  }
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
    } else if (count > 0) {
      __at_align__ int16_t tmp_values[size()];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(int16_t));
    }
  }
  template <int64_t mask>
  static Vectorized<T> blend(const Vectorized<T>& a, const Vectorized<T>& b) {
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
  static Vectorized<T> blendv(const Vectorized<T>& a,
      const Vectorized<T>& b, const Vectorized<T>& mask) {
    return _mm256_blendv_epi8(a.values, b.values, mask.values);
  }
  template<typename step_t>
  static Vectorized<T> arange(T base = 0.f, step_t step = static_cast<step_t>(1)) {
    return Vectorized<T>(
      base,             base +      step, base +  2 * step, base +  3 * step,
      base +  4 * step, base +  5 * step, base +  6 * step, base +  7 * step,
      base +  8 * step, base +  9 * step, base + 10 * step, base + 11 * step,
      base + 12 * step, base + 13 * step, base + 14 * step, base + 15 * step);
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
    }
    return b;
  }
  Vectorized<T> map(const __m256 (*const vop)(__m256)) const {
    __m256 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    const auto o1 = vop(lo);
    const auto o2 = vop(hi);
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> isnan() const {
    __m256 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    lo = _mm256_cmp_ps(lo, _mm256_set1_ps(0.0f), _CMP_UNORD_Q);
    hi = _mm256_cmp_ps(hi, _mm256_set1_ps(0.0f), _CMP_UNORD_Q);
    return merge_compare_result(lo, hi);
  }
  Vectorized<T> abs() const {
    return _mm256_andnot_si256(_mm256_set1_epi16(0x8000), values);
  }
  Vectorized<T> angle() const {
    __m256 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    auto angle_lambda = [](__m256 values_2) {
      const auto zero_vec = _mm256_set1_ps(0.f);
      const auto nan_vec = _mm256_set1_ps(NAN);
      const auto not_nan_mask = _mm256_cmp_ps(values_2, values_2, _CMP_EQ_OQ);
      const auto nan_mask = _mm256_cmp_ps(not_nan_mask, zero_vec, _CMP_EQ_OQ);
      const auto pi = _mm256_set1_ps(c10::pi<float>);

      const auto neg_mask = _mm256_cmp_ps(values_2, zero_vec, _CMP_LT_OQ);
      auto angle = _mm256_blendv_ps(zero_vec, pi, neg_mask);
      angle = _mm256_blendv_ps(angle, nan_vec, nan_mask);
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
    return _mm256_set1_epi16(0);
  }
  Vectorized<T> conj() const {
    return *this;
  }
  Vectorized<T> acos() const {
    return map(Sleef_acosf8_u10);
  }
  Vectorized<T> acosh() const {
    return map(Sleef_acoshf8_u10);
  }
  Vectorized<T> asin() const {
    return map(Sleef_asinf8_u10);
  }
  Vectorized<T> atan() const {
    return map(Sleef_atanf8_u10);
  }
  Vectorized<T> atanh() const {
    return map(Sleef_atanhf8_u10);
  }
  Vectorized<T> atan2(const Vectorized<T> &b) const {
    __m256 lo, hi;
    __m256 b1, b2;
    cvt_to_fp32<T>(values, lo, hi);
    cvt_to_fp32<T>(b.values, b1, b2);
    auto o1 = Sleef_atan2f8_u10(lo, b1);
    auto o2 = Sleef_atan2f8_u10(hi, b2);
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> copysign(const Vectorized<T> &sign) const {
    // copy sign bit (0x8000) from sign and remaining bits from values
    __m256i mask_value = _mm256_set1_epi32(~0x80008000);
    __m256i mask_signbit = _mm256_set1_epi32(0x80008000);
    return Vectorized<T>(
      _mm256_or_si256(
        _mm256_and_si256(values, mask_value),
        _mm256_and_si256(sign, mask_signbit)));
  }
  Vectorized<T> erf() const {
    return map(Sleef_erff8_u10);
  }
  Vectorized<T> erfc() const {
    return map(Sleef_erfcf8_u15);
  }
  Vectorized<T> erfinv() const {
    __m256 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    __at_align__ float tmp1[size() / 2], tmp2[size() / 2];
    _mm256_storeu_ps(reinterpret_cast<float*>(tmp1), lo);
    _mm256_storeu_ps(reinterpret_cast<float*>(tmp2), hi);
    for (int64_t i = 0; i < size() / 2; i++) {
      tmp1[i] = calc_erfinv(tmp1[i]);
      tmp2[i] = calc_erfinv(tmp2[i]);
    }
    auto o1 = _mm256_loadu_ps(tmp1);
    auto o2 = _mm256_loadu_ps(tmp2);
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> exp() const {
    return map(Sleef_expf8_u10);
  }
  Vectorized<T> exp2() const {
    return map(Sleef_exp2f8_u10);
  }
  Vectorized<T> expm1() const {
    return map(Sleef_expm1f8_u10);
  }
  Vectorized<T> exp_u20() const {
    return exp();
  }
  Vectorized<T> fmod(const Vectorized<T> & q) const {
    __m256 x_lo, x_hi;
    cvt_to_fp32<T>(values, x_lo, x_hi);
    __m256 q_lo, q_hi;
    cvt_to_fp32<T>(q.values, q_lo, q_hi);
    auto o1 = Sleef_fmodf8(x_lo, q_lo);
    auto o2 = Sleef_fmodf8(x_hi, q_hi);
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> hypot(const Vectorized<T> &b) const {
    __m256 lo, hi;
    __m256 b1, b2;
    cvt_to_fp32<T>(values, lo, hi);
    cvt_to_fp32<T>(b.values, b1, b2);
    auto o1 = Sleef_hypotf8_u05(lo, b1);
    auto o2 = Sleef_hypotf8_u05(hi, b2);
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> i0() const {
    __m256 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    __at_align__ float tmp1[size() / 2], tmp2[size() / 2];
    _mm256_storeu_ps(reinterpret_cast<float*>(tmp1), lo);
    _mm256_storeu_ps(reinterpret_cast<float*>(tmp2), hi);
    for (int64_t i = 0; i < size() / 2; i++) {
      tmp1[i] = calc_i0(tmp1[i]);
      tmp2[i] = calc_i0(tmp2[i]);
    }
    auto o1 = _mm256_loadu_ps(tmp1);
    auto o2 = _mm256_loadu_ps(tmp2);
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> i0e() const {
    __m256 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    constexpr auto sz = size();
    __at_align__ float tmp1[sz / 2], tmp2[sz / 2];
    _mm256_storeu_ps(reinterpret_cast<float*>(tmp1), lo);
    _mm256_storeu_ps(reinterpret_cast<float*>(tmp2), hi);

    for (auto i = decltype(sz){0}; i < sz / 2; i++) {
      tmp1[i] = calc_i0e(tmp1[i]);
      tmp2[i] = calc_i0e(tmp2[i]);
    }
    const auto o1 = _mm256_loadu_ps(tmp1);
    const auto o2 = _mm256_loadu_ps(tmp2);
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> digamma() const {
    __m256 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    constexpr auto sz = size();
    __at_align__ float tmp1[sz / 2], tmp2[sz / 2];
    _mm256_storeu_ps(reinterpret_cast<float*>(tmp1), lo);
    _mm256_storeu_ps(reinterpret_cast<float*>(tmp2), hi);

    for (auto i = decltype(sz){0}; i < sz / 2; i++) {
      tmp1[i] = calc_digamma(tmp1[i]);
      tmp2[i] = calc_digamma(tmp2[i]);
    }
    const auto o1 = _mm256_loadu_ps(tmp1);
    const auto o2 = _mm256_loadu_ps(tmp2);
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> igamma(const Vectorized<T> &x) const {
    __m256 lo, hi;
    __m256 xlo, xhi;
    cvt_to_fp32<T>(values, lo, hi);
    cvt_to_fp32<T>(x.values, xlo, xhi);
    __at_align__ float tmp1[size() / 2], tmp2[size() / 2];
    _mm256_storeu_ps(reinterpret_cast<float*>(tmp1), lo);
    _mm256_storeu_ps(reinterpret_cast<float*>(tmp2), hi);
    __at_align__ float tmpx1[size() / 2], tmpx2[size() / 2];
    _mm256_storeu_ps(reinterpret_cast<float*>(tmpx1), xlo);
    _mm256_storeu_ps(reinterpret_cast<float*>(tmpx2), xhi);
    for (int64_t i = 0; i < size() / 2; ++i) {
      tmp1[i] = calc_igamma(tmp1[i], tmpx1[i]);
      tmp2[i] = calc_igamma(tmp2[i], tmpx2[i]);
    }
    auto o1 = _mm256_loadu_ps(tmp1);
    auto o2 = _mm256_loadu_ps(tmp2);
    return cvt_from_fp32<T>(o1, o2);
  }

  Vectorized<T> igammac(const Vectorized<T> &x) const {
    __m256 lo, hi;
    __m256 xlo, xhi;
    cvt_to_fp32<T>(values, lo, hi);
    cvt_to_fp32<T>(x.values, xlo, xhi);
    __at_align__ float tmp1[size() / 2], tmp2[size() / 2];
    _mm256_storeu_ps(reinterpret_cast<float*>(tmp1), lo);
    _mm256_storeu_ps(reinterpret_cast<float*>(tmp2), hi);
    __at_align__ float tmpx1[size() / 2], tmpx2[size() / 2];
    _mm256_storeu_ps(reinterpret_cast<float*>(tmpx1), xlo);
    _mm256_storeu_ps(reinterpret_cast<float*>(tmpx2), xhi);
    for (int64_t i = 0; i < size() / 2; ++i) {
      tmp1[i] = calc_igammac(tmp1[i], tmpx1[i]);
      tmp2[i] = calc_igammac(tmp2[i], tmpx2[i]);
    }
    auto o1 = _mm256_loadu_ps(tmp1);
    auto o2 = _mm256_loadu_ps(tmp2);
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> log() const {
    return map(Sleef_logf8_u10);
  }
  Vectorized<T> log2() const {
    return map(Sleef_log2f8_u10);
  }
  Vectorized<T> log10() const {
    return map(Sleef_log10f8_u10);
  }
  Vectorized<T> log1p() const {
    return map(Sleef_log1pf8_u10);
  }
  Vectorized<T> sin() const {
    return map(Sleef_sinf8_u10);
  }
  Vectorized<T> sinh() const {
    return map(Sleef_sinhf8_u10);
  }
  Vectorized<T> cos() const {
    return map(Sleef_cosf8_u10);
  }
  Vectorized<T> cosh() const {
    return map(Sleef_coshf8_u10);
  }
  Vectorized<T> ceil() const {
    __m256 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    auto o1 = _mm256_ceil_ps(lo);
    auto o2 = _mm256_ceil_ps(hi);
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> floor() const {
    __m256 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    auto o1 = _mm256_floor_ps(lo);
    auto o2 = _mm256_floor_ps(hi);
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> neg() const {
    return _mm256_xor_si256(values, _mm256_set1_epi16(0x8000));
  }
  Vectorized<T> round() const {
    __m256 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    auto o1 = _mm256_round_ps(lo, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    auto o2 = _mm256_round_ps(hi, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> tan() const {
    return map(Sleef_tanf8_u10);
  }
  Vectorized<T> tanh() const {
    return map(Sleef_tanhf8_u10);
  }
  Vectorized<T> trunc() const {
    __m256 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    auto o1 = _mm256_round_ps(lo, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
    auto o2 = _mm256_round_ps(hi, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> lgamma() const {
    return map(Sleef_lgammaf8_u10);
  }
  Vectorized<T> sqrt() const {
    __m256 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    auto o1 = _mm256_sqrt_ps(lo);
    auto o2 = _mm256_sqrt_ps(hi);
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> reciprocal() const {
    __m256 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    auto ones = _mm256_set1_ps(1);
    auto o1 = _mm256_div_ps(ones, lo);
    auto o2 = _mm256_div_ps(ones, hi);
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> rsqrt() const {
    __m256 lo, hi;
    cvt_to_fp32<T>(values, lo, hi);
    auto ones = _mm256_set1_ps(1);
    auto o1 = _mm256_div_ps(ones, _mm256_sqrt_ps(lo));
    auto o2 = _mm256_div_ps(ones, _mm256_sqrt_ps(hi));
    return cvt_from_fp32<T>(o1, o2);
  }
  Vectorized<T> pow(const Vectorized<T> &b) const {
    __m256 lo, hi;
    __m256 b1, b2;
    cvt_to_fp32<T>(values, lo, hi);
    cvt_to_fp32<T>(b.values, b1, b2);
    auto o1 = Sleef_powf8_u10(lo, b1);
    auto o2 = Sleef_powf8_u10(hi, b2);
    return cvt_from_fp32<T>(o1, o2);
  }
private:
  template<typename Op>
  Vectorized<T> inline binary_compare(const Vectorized<T>& b, Op op) const {
    __m256 a_lo, a_hi;
    __m256 b_lo, b_hi;
    cvt_to_fp32<T>(values, a_lo, a_hi);
    cvt_to_fp32<T>(b.values, b_lo, b_hi);
    auto o1 = op(a_lo, b_lo);
    auto o2 = op(a_hi, b_hi);
    return cvt_from_fp32<T, /*is_compare_op*/true>(o1, o2);
  }

public:
  Vectorized<T> inline operator>(const Vectorized<T>& other) const {
    return binary_compare(other, [](__m256 x, __m256 y) { return _mm256_cmp_ps(x, y, _CMP_GT_OQ); });
  }
  Vectorized<T> inline operator<(const Vectorized<T>& other) const {
    return binary_compare(other, [](__m256 x, __m256 y) { return _mm256_cmp_ps(x, y, _CMP_LT_OQ); });
  }
  Vectorized<T> inline operator>=(const Vectorized<T>& other) const {
    return binary_compare(other, [](__m256 x, __m256 y) { return _mm256_cmp_ps(x, y, _CMP_GE_OQ); });
  }
  Vectorized<T> inline operator<=(const Vectorized<T>& other) const {
    return binary_compare(other, [](__m256 x, __m256 y) { return _mm256_cmp_ps(x, y, _CMP_LE_OQ); });
  }
  Vectorized<T> inline operator==(const Vectorized<T>& other) const {
    return binary_compare(other, [](__m256 x, __m256 y) { return _mm256_cmp_ps(x, y, _CMP_EQ_OQ); });
  }
  Vectorized<T> inline operator!=(const Vectorized<T>& other) const {
    return binary_compare(other, [](__m256 x, __m256 y) { return _mm256_cmp_ps(x, y, _CMP_NEQ_UQ); });
  }
};

template<typename T, typename Op>
static inline Vectorized<T> binary_op_as_fp32(const Vectorized<T>& a, const Vectorized<T>& b, Op op) {
  __m256 a_lo, a_hi;
  __m256 b_lo, b_hi;
  cvt_to_fp32<T>(__m256i(a), a_lo, a_hi);
  cvt_to_fp32<T>(__m256i(b), b_lo, b_hi);
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
  return binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) { return _mm256_add_ps(x, y); });
}
Vectorized<BFloat16> inline operator-(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
  return binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) { return _mm256_sub_ps(x, y); });
}
Vectorized<BFloat16> inline operator*(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
  return binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) { return _mm256_mul_ps(x, y); });
}
Vectorized<BFloat16> inline operator/(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
  return binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) { return _mm256_div_ps(x, y); });
}
Vectorized<BFloat16> inline operator&(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
  return _mm256_and_si256(a, b);
}
Vectorized<BFloat16> inline operator|(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
  return _mm256_or_si256(a, b);
}
Vectorized<BFloat16> inline operator^(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
  return _mm256_xor_si256(a, b);
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
Vectorized<BFloat16> inline minimum(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& b) {
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
Vectorized<BFloat16> inline clamp(const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& min, const Vectorized<BFloat16>& max) {
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
Vectorized<BFloat16> inline clamp_max(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& max) {
  __m256 a_lo, a_hi;
  __m256 max_lo, max_hi;
  cvtbf16_fp32(__m256i(a), a_lo, a_hi);
  cvtbf16_fp32(__m256i(max), max_lo, max_hi);
  auto o1 = _mm256_min_ps(max_lo, a_lo);
  auto o2 = _mm256_min_ps(max_hi, a_hi);
  return cvtfp32_bf16(o1, o2);
}

template <>
Vectorized<BFloat16> inline clamp_min(const Vectorized<BFloat16>& a, const Vectorized<BFloat16>& min) {
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
#pragma unroll
  for (i = 0; i <= (n - Vectorized<BFloat16>::size()); i += Vectorized<BFloat16>::size()) {
    auto vsrc = _mm256_loadu_si256(reinterpret_cast<__m256i*>((void*)(src + i)));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>((void*)(dst + i)), vsrc);
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
  auto load_float = [](const double *src) -> __m256 {
    // Load one float vector from an array of doubles
    __m128 a = _mm256_cvtpd_ps(_mm256_loadu_pd(src));
    __m128 b = _mm256_cvtpd_ps(_mm256_loadu_pd(src + 4));
    return _mm256_insertf128_ps(_mm256_castps128_ps256(a), b, 1);
  };

  int64_t i;
  for (i = 0; i + Vectorized<BFloat16>::size() <= n; i += Vectorized<BFloat16>::size()) {
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
Vectorized<BFloat16> inline fmadd(const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b, const Vectorized<BFloat16>& c) {
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
#pragma unroll
  for (i = 0; i <= (n - Vectorized<Half>::size()); i += Vectorized<Half>::size()) {
    auto vsrc = _mm256_loadu_si256(reinterpret_cast<__m256i*>((void*)(src + i)));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>((void*)(dst + i)), vsrc);
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

#define CONVERT_VECTORIZED_INIT(type, name) \
inline std::tuple<Vectorized<float>, Vectorized<float>> convert_##name##_float(const Vectorized<type>& a) { \
  __m256 o1, o2; \
  cvt_to_fp32<type>(__m256i(a), o1, o2); \
  return std::make_tuple(o1, o2); \
} \
inline Vectorized<type> convert_float_##name(const Vectorized<float>& a, const Vectorized<float>& b) { \
  return cvt_from_fp32<type>(__m256(a), __m256(b)); \
}
CONVERT_VECTORIZED_INIT(BFloat16, bfloat16);
CONVERT_VECTORIZED_INIT(Half, half);

#else // defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)

#define CONVERT_NON_VECTORIZED_INIT(type, name) \
inline std::tuple<Vectorized<float>, Vectorized<float>> convert_##name##_float(const Vectorized<type>& a) { \
  constexpr int64_t K = Vectorized<type>::size(); \
  __at_align__ float arr[K]; \
  __at_align__ type arr2[K]; \
  a.store(arr2); \
  convert(arr2, arr, K); \
  return std::make_tuple( \
      Vectorized<float>::loadu(arr), \
      Vectorized<float>::loadu(arr + Vectorized<float>::size())); \
} \
inline Vectorized<type> convert_float_##name(const Vectorized<float>& a, const Vectorized<float>& b) { \
  constexpr int64_t K = Vectorized<type>::size(); \
  __at_align__ float arr[K]; \
  __at_align__ type arr2[K]; \
  a.store(arr); \
  b.store(arr + Vectorized<float>::size()); \
  convert(arr, arr2, K); \
  return Vectorized<type>::loadu(arr2); \
}
CONVERT_NON_VECTORIZED_INIT(BFloat16, bfloat16);
#if defined(__aarch64__) && !defined(C10_MOBILE) && !defined(__CUDACC__)
inline std::tuple<Vectorized<float>, Vectorized<float>> convert_half_float(const Vectorized<Half>& a) {
  static_assert(Vectorized<Half>::size() == 2 * Vectorized<float>::size());
  auto arr = reinterpret_cast<const float16_t*>(a.operator const Half*());
  float16x8_t x = vld1q_f16(arr);
  float32x4_t x1 = vcvt_f32_f16(vget_low_f16(x));
  float32x4_t x2 = vcvt_f32_f16(vget_high_f16(x));
  float16x8_t y = vld1q_f16(arr + Vectorized<float>::size());
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
  Vectorized<Half> rc;
  auto arr = reinterpret_cast<float16_t*>(rc.operator Half*());
  vst1q_f16(arr, vcombine_f16(x1, x2));
  vst1q_f16(arr + Vectorized<float>::size(), vcombine_f16(y1, y2));
  return rc;
}
#else
CONVERT_NON_VECTORIZED_INIT(Half, half);
#endif

#endif // defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)

#if defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)
#define LOAD_FP32_VECTORIZED_INIT(type, name) \
inline void load_fp32_from_##name(const type *data, Vectorized<float>& out) { \
  auto values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data)); \
  __m256 out_values; \
  cvt_to_fp32<type>(values, out_values); \
  out = out_values; \
} \
\
inline void load_fp32_from_##name(const type *data, Vectorized<float>& out1, Vectorized<float>& out2) { \
  auto vec = Vectorized<type>::loadu(data); \
  __m256 out1_values, out2_values; \
  cvt_to_fp32<type>(vec, out1_values, out2_values); \
  out1 = out1_values; \
  out2 = out2_values; \
}
LOAD_FP32_VECTORIZED_INIT(BFloat16, bf16);
LOAD_FP32_VECTORIZED_INIT(Half, fp16);

#else // defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)
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
}} // namsepace at::vec::CPU_CAPABILITY

#pragma GCC diagnostic pop
