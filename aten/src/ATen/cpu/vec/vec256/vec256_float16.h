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

namespace at {
namespace vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)

static inline void cvtfp16_fp32(const __m128i& a, __m256& o) {
  o = _mm256_cvtph_ps(a);
}

static inline void cvtfp16_fp32(const __m256i& a, __m256& o1, __m256& o2) {
  __m128i lo = _mm256_extractf128_si256(a, 0);
  __m128i hi = _mm256_extractf128_si256(a, 1);
  cvtfp16_fp32(lo, o1);
  cvtfp16_fp32(hi, o2);
}

static inline __m256i cvtfp32_fp16(const __m256& a, const __m256& b) {
  __m128i lo = _mm256_cvtps_ph(
      a, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  __m128i hi = _mm256_cvtps_ph(
      b, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  return _mm256_insertf128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

template <> class Vectorized<Half> {
private:
  __m256i values;
public:
  using value_type = uint16_t;
  using size_type = int;
  static constexpr size_type size() {
    return 16;
  }
  Vectorized() {}
  Vectorized(__m256i v) : values(v) {}
  Vectorized(Half val) {
    value_type uw = val.x;
    values = _mm256_set1_epi16(uw);
  }
  Vectorized(Half val1, Half val2, Half val3, Half val4,
         Half val5, Half val6, Half val7, Half val8,
         Half val9, Half val10, Half val11, Half val12,
         Half val13, Half val14, Half val15, Half val16) {
    values = _mm256_setr_epi16(
        val1.x, val2.x, val3.x, val4.x, val5.x, val6.x, val7.x, val8.x,
        val9.x, val10.x, val11.x, val12.x, val13.x, val14.x, val15.x, val16.x);
  }
  operator __m256i() const {
    return values;
  }
  Half& operator[](int idx) = delete;
  const Half& operator[](int idx) const  = delete;
  int zero_mask() const {
    // returns an integer mask where all zero elements are translated to 1-bit and others are translated to 0-bit
    __m256i cmp = _mm256_cmpeq_epi16(values, _mm256_set1_epi16(0));
    return _mm256_movemask_epi8(cmp);
  }
  static Vectorized<Half> loadu(const void* ptr) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
  }
  static Vectorized<Half> loadu(const void* ptr, int16_t count) {
    __at_align__ int16_t tmp_values[size()];
    std::memcpy(tmp_values, ptr, count * sizeof(int16_t));
    return loadu(tmp_values);
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
  static Vectorized<Half> blend(const Vectorized<Half>& a, const Vectorized<Half>& b) {
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
  static Vectorized<Half> blendv(const Vectorized<Half>& a,
      const Vectorized<Half>& b, const Vectorized<Half>& mask) {
    return _mm256_blendv_epi8(a.values, b.values, mask.values);
  }
  template<typename step_t>
  static Vectorized<Half> arange(Half base = 0.f, step_t step = static_cast<step_t>(1)) {
    return Vectorized<Half>(
      base,             base +      step, base +  2 * step, base +  3 * step,
      base +  4 * step, base +  5 * step, base +  6 * step, base +  7 * step,
      base +  8 * step, base +  9 * step, base + 10 * step, base + 11 * step,
      base + 12 * step, base + 13 * step, base + 14 * step, base + 15 * step);
  }
  static Vectorized<Half> set(const Vectorized<Half>& a,
      const Vectorized<Half>& b, int64_t count = size()) {
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
  Vectorized<Half> map(const __m256 (*const vop)(__m256)) const {
    __m256 lo, hi;
    cvtfp16_fp32(values, lo, hi);
    const auto o1 = vop(lo);
    const auto o2 = vop(hi);
    return cvtfp32_fp16(o1, o2);
  }
  Vectorized<Half> abs() const {
    __m256 lo, hi;
    cvtfp16_fp32(values, lo, hi);
    const auto mask = _mm256_set1_ps(-0.f);
    const auto o1 = _mm256_andnot_ps(mask, lo);
    const auto o2 = _mm256_andnot_ps(mask, hi);
    return cvtfp32_fp16(o1, o2);
  }
  Vectorized<Half> angle() const {
    __m256 lo, hi;
    cvtfp16_fp32(values, lo, hi);
    auto angle_lambda = [](__m256 values) {
      const auto zero_vec = _mm256_set1_ps(0.f);
      const auto nan_vec = _mm256_set1_ps(NAN);
      const auto not_nan_mask = _mm256_cmp_ps(values, values, _CMP_EQ_OQ);
      const auto nan_mask = _mm256_cmp_ps(not_nan_mask, zero_vec, _CMP_EQ_OQ);
      const auto pi = _mm256_set1_ps(c10::pi<float>);

      const auto neg_mask = _mm256_cmp_ps(values, zero_vec, _CMP_LT_OQ);
      auto angle = _mm256_blendv_ps(zero_vec, pi, neg_mask);
      angle = _mm256_blendv_ps(angle, nan_vec, nan_mask);
      return angle;
    };
    auto o1 = angle_lambda(lo);
    auto o2 = angle_lambda(hi);
    return cvtfp32_fp16(o1, o2);
  }
  Vectorized<Half> real() const {
    return *this;
  }
  Vectorized<Half> imag() const {
    return _mm256_set1_epi16(0);
  }
  Vectorized<Half> conj() const {
    return *this;
  }
  Vectorized<Half> acos() const {
    return map(Sleef_acosf8_u10);
  }
  Vectorized<Half> asin() const {
    return map(Sleef_asinf8_u10);
  }
  Vectorized<Half> atan() const {
    return map(Sleef_atanf8_u10);
  }
  Vectorized<Half> atan2(const Vectorized<Half> &b) const {
    __m256 lo, hi;
    __m256 b1, b2;
    cvtfp16_fp32(values, lo, hi);
    cvtfp16_fp32(b.values, b1, b2);
    auto o1 = Sleef_atan2f8_u10(lo, b1);
    auto o2 = Sleef_atan2f8_u10(hi, b2);
    return cvtfp32_fp16(o1, o2);
  }
  Vectorized<Half> copysign(const Vectorized<Half> &sign) const {
    // copy sign bit (0x8000) from sign and remaining bits from values
    __m256i mask_value = _mm256_set1_epi32(~0x80008000);
    __m256i mask_signbit = _mm256_set1_epi32(0x80008000);
    return Vectorized<Half>(
      _mm256_or_si256(
        _mm256_and_si256(values, mask_value),
        _mm256_and_si256(sign, mask_signbit)));
  }
  Vectorized<Half> erf() const {
    return map(Sleef_erff8_u10);
  }
  Vectorized<Half> erfc() const {
    return map(Sleef_erfcf8_u15);
  }
  Vectorized<Half> erfinv() const {
    __m256 lo, hi;
    cvtfp16_fp32(values, lo, hi);
    __at_align__ float tmp1[size() / 2], tmp2[size() / 2];
    _mm256_storeu_ps(reinterpret_cast<float*>(tmp1), lo);
    _mm256_storeu_ps(reinterpret_cast<float*>(tmp2), hi);
    for (int64_t i = 0; i < size() / 2; i++) {
      tmp1[i] = calc_erfinv(tmp1[i]);
      tmp2[i] = calc_erfinv(tmp2[i]);
    }
    auto o1 = _mm256_loadu_ps(tmp1);
    auto o2 = _mm256_loadu_ps(tmp2);
    return cvtfp32_fp16(o1, o2);
  }
  Vectorized<Half> exp() const {
    return map(Sleef_expf8_u10);
  }
  Vectorized<Half> exp2() const {
    return map(Sleef_exp2f8_u10);
  }
  Vectorized<Half> expm1() const {
    return map(Sleef_expm1f8_u10);
  }
  Vectorized<Half> fmod(const Vectorized<Half> & q) const {
    __m256 x_lo, x_hi;
    cvtfp16_fp32(values, x_lo, x_hi);
    __m256 q_lo, q_hi;
    cvtfp16_fp32(q.values, q_lo, q_hi);
    auto o1 = Sleef_fmodf8(x_lo, q_lo);
    auto o2 = Sleef_fmodf8(x_hi, q_hi);
    return cvtfp32_fp16(o1, o2);
  }
  Vectorized<Half> hypot(const Vectorized<Half> &b) const {
    __m256 lo, hi;
    __m256 b1, b2;
    cvtfp16_fp32(values, lo, hi);
    cvtfp16_fp32(b.values, b1, b2);
    auto o1 = Sleef_hypotf8_u05(lo, b1);
    auto o2 = Sleef_hypotf8_u05(hi, b2);
    return cvtfp32_fp16(o1, o2);
  }
  Vectorized<Half> i0() const {
    __m256 lo, hi;
    cvtfp16_fp32(values, lo, hi);
    __at_align__ float tmp1[size() / 2], tmp2[size() / 2];
    _mm256_storeu_ps(reinterpret_cast<float*>(tmp1), lo);
    _mm256_storeu_ps(reinterpret_cast<float*>(tmp2), hi);
    for (int64_t i = 0; i < size() / 2; i++) {
      tmp1[i] = calc_i0(tmp1[i]);
      tmp2[i] = calc_i0(tmp2[i]);
    }
    auto o1 = _mm256_loadu_ps(tmp1);
    auto o2 = _mm256_loadu_ps(tmp2);
    return cvtfp32_fp16(o1, o2);
  }
  Vectorized<Half> i0e() const {
    __m256 lo, hi;
    cvtfp16_fp32(values, lo, hi);
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
    return cvtfp32_fp16(o1, o2);
  }
  Vectorized<Half> igamma(const Vectorized<Half> &x) const {
    __m256 lo, hi;
    __m256 xlo, xhi;
    cvtfp16_fp32(values, lo, hi);
    cvtfp16_fp32(x.values, xlo, xhi);
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
    return cvtfp32_fp16(o1, o2);
  }

  Vectorized<Half> igammac(const Vectorized<Half> &x) const {
    __m256 lo, hi;
    __m256 xlo, xhi;
    cvtfp16_fp32(values, lo, hi);
    cvtfp16_fp32(x.values, xlo, xhi);
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
    return cvtfp32_fp16(o1, o2);
  }
  Vectorized<Half> log() const {
    return map(Sleef_logf8_u10);
  }
  Vectorized<Half> log2() const {
    return map(Sleef_log2f8_u10);
  }
  Vectorized<Half> log10() const {
    return map(Sleef_log10f8_u10);
  }
  Vectorized<Half> log1p() const {
    return map(Sleef_log1pf8_u10);
  }
  Vectorized<Half> frac() const;
  Vectorized<Half> sin() const {
    return map(Sleef_sinf8_u10);
  }
  Vectorized<Half> sinh() const {
    return map(Sleef_sinhf8_u10);
  }
  Vectorized<Half> cos() const {
    return map(Sleef_cosf8_u10);
  }
  Vectorized<Half> cosh() const {
    return map(Sleef_coshf8_u10);
  }
  Vectorized<Half> ceil() const {
    __m256 lo, hi;
    cvtfp16_fp32(values, lo, hi);
    auto o1 = _mm256_ceil_ps(lo);
    auto o2 = _mm256_ceil_ps(hi);
    return cvtfp32_fp16(o1, o2);
  }
  Vectorized<Half> floor() const {
    __m256 lo, hi;
    cvtfp16_fp32(values, lo, hi);
    auto o1 = _mm256_floor_ps(lo);
    auto o2 = _mm256_floor_ps(hi);
    return cvtfp32_fp16(o1, o2);
  }
  Vectorized<Half> neg() const {
    __m256 lo, hi;
    cvtfp16_fp32(values, lo, hi);
    auto mask = _mm256_set1_ps(-0.f);
    auto o1 = _mm256_xor_ps(mask, lo);
    auto o2 = _mm256_xor_ps(mask, hi);
    return cvtfp32_fp16(o1, o2);
  }
  Vectorized<Half> round() const {
    __m256 lo, hi;
    cvtfp16_fp32(values, lo, hi);
    auto o1 = _mm256_round_ps(lo, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    auto o2 = _mm256_round_ps(hi, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    return cvtfp32_fp16(o1, o2);
  }
  Vectorized<Half> tan() const {
    return map(Sleef_tanf8_u10);
  }
  Vectorized<Half> tanh() const {
    return map(Sleef_tanhf8_u10);
  }
  Vectorized<Half> trunc() const {
    __m256 lo, hi;
    cvtfp16_fp32(values, lo, hi);
    auto o1 = _mm256_round_ps(lo, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
    auto o2 = _mm256_round_ps(hi, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
    return cvtfp32_fp16(o1, o2);
  }
  Vectorized<Half> lgamma() const {
    return map(Sleef_lgammaf8_u10);
  }
  Vectorized<Half> sqrt() const {
    __m256 lo, hi;
    cvtfp16_fp32(values, lo, hi);
    auto o1 = _mm256_sqrt_ps(lo);
    auto o2 = _mm256_sqrt_ps(hi);
    return cvtfp32_fp16(o1, o2);
  }
  Vectorized<Half> reciprocal() const {
    __m256 lo, hi;
    cvtfp16_fp32(values, lo, hi);
    auto ones = _mm256_set1_ps(1);
    auto o1 = _mm256_div_ps(ones, lo);
    auto o2 = _mm256_div_ps(ones, hi);
    return cvtfp32_fp16(o1, o2);
  }
  Vectorized<Half> rsqrt() const {
    __m256 lo, hi;
    cvtfp16_fp32(values, lo, hi);
    auto ones = _mm256_set1_ps(1);
    auto o1 = _mm256_div_ps(ones, _mm256_sqrt_ps(lo));
    auto o2 = _mm256_div_ps(ones, _mm256_sqrt_ps(hi));
    return cvtfp32_fp16(o1, o2);
  }
  Vectorized<Half> pow(const Vectorized<Half> &b) const {
    __m256 lo, hi;
    __m256 b1, b2;
    cvtfp16_fp32(values, lo, hi);
    cvtfp16_fp32(b.values, b1, b2);
    auto o1 = Sleef_powf8_u10(lo, b1);
    auto o2 = Sleef_powf8_u10(hi, b2);
    return cvtfp32_fp16(o1, o2);
  }

  Vectorized<Half> inline operator>(const Vectorized<Half>& other) const;
  Vectorized<Half> inline operator<(const Vectorized<Half>& other) const;
  Vectorized<Half> inline operator>=(const Vectorized<Half>& other) const;
  Vectorized<Half> inline operator<=(const Vectorized<Half>& other) const;
  Vectorized<Half> inline operator==(const Vectorized<Half>& other) const;
  Vectorized<Half> inline operator!=(const Vectorized<Half>& other) const;

  Vectorized<Half> eq(const Vectorized<Half>& other) const;
  Vectorized<Half> ne(const Vectorized<Half>& other) const;
  Vectorized<Half> gt(const Vectorized<Half>& other) const;
  Vectorized<Half> ge(const Vectorized<Half>& other) const;
  Vectorized<Half> lt(const Vectorized<Half>& other) const;
  Vectorized<Half> le(const Vectorized<Half>& other) const;
};

template<typename Op>
Vectorized<Half> static inline fp16_binary_op_as_fp32(const Vectorized<Half>& a, const Vectorized<Half>& b, Op op) {
  __m256 a_lo, a_hi;
  __m256 b_lo, b_hi;
  cvtfp16_fp32(__m256i(a), a_lo, a_hi);
  cvtfp16_fp32(__m256i(b), b_lo, b_hi);
  auto o1 = op(a_lo, b_lo);
  auto o2 = op(a_hi, b_hi);
  return cvtfp32_fp16(o1, o2);
}

Vectorized<Half> inline Vectorized<Half>::operator>(const Vectorized<Half>& other) const {
  return fp16_binary_op_as_fp32(*this, other, [](__m256 x, __m256 y) {
    return _mm256_cmp_ps(x, y, _CMP_GT_OQ);
  });
}
Vectorized<Half> inline Vectorized<Half>::operator<(const Vectorized<Half>& other) const {
  return fp16_binary_op_as_fp32(*this, other, [](__m256 x, __m256 y) {
    return _mm256_cmp_ps(x, y, _CMP_LT_OQ);
  });
}
Vectorized<Half> inline Vectorized<Half>::operator>=(const Vectorized<Half>& other) const {
  return fp16_binary_op_as_fp32(*this, other, [](__m256 x, __m256 y) {
    return _mm256_cmp_ps(x, y, _CMP_GE_OQ);
  });
}
Vectorized<Half> inline Vectorized<Half>::operator<=(const Vectorized<Half>& other) const {
  return fp16_binary_op_as_fp32(*this, other, [](__m256 x, __m256 y) {
    return _mm256_cmp_ps(x, y, _CMP_LE_OQ);
  });
}
Vectorized<Half> inline Vectorized<Half>::operator==(const Vectorized<Half>& other) const {
  return fp16_binary_op_as_fp32(*this, other, [](__m256 x, __m256 y) {
    return _mm256_cmp_ps(x, y, _CMP_EQ_OQ);
  });
}
Vectorized<Half> inline Vectorized<Half>::operator!=(const Vectorized<Half>& other) const {
  return fp16_binary_op_as_fp32(*this, other, [](__m256 x, __m256 y) {
    return _mm256_cmp_ps(x, y, _CMP_NEQ_UQ);
  });
}

Vectorized<Half> inline operator+(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  return fp16_binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) { return _mm256_add_ps(x, y); });
}
Vectorized<Half> inline operator-(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  return fp16_binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) { return _mm256_sub_ps(x, y); });
}
Vectorized<Half> inline operator*(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  return fp16_binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) { return _mm256_mul_ps(x, y); });
}
Vectorized<Half> inline operator/(const Vectorized<Half>& a, const Vectorized<Half>& b) {
  return fp16_binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) { return _mm256_div_ps(x, y); });
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

inline std::tuple<Vectorized<float>, Vectorized<float>> convert_half_float(const Vectorized<Half>& a) {
  __m256 o1, o2;
  cvtfp16_fp32(__m256i(a), o1, o2);
  return std::make_tuple(o1, o2);
}

inline Vectorized<Half> convert_float_half(const Vectorized<float>& a, const Vectorized<float>& b) {
 return cvtfp32_fp16(__m256(a), __m256(b));
}


#else // defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)

inline std::tuple<Vectorized<float>, Vectorized<float>> convert_half_float(const Vectorized<Half>& a) {
  constexpr int64_t K = Vectorized<Half>::size();
  __at_align__ float arr[K];
  __at_align__ Half arr2[K];
  a.store(arr2);
  convert(arr2, arr, K);
  return std::make_tuple(
      Vectorized<float>::loadu(arr),
      Vectorized<float>::loadu(arr + Vectorized<float>::size()));
}

inline Vectorized<Half> convert_float_half(const Vectorized<float>& a, const Vectorized<float>& b) {
  constexpr int64_t K = Vectorized<Half>::size();
  __at_align__ float arr[K];
  __at_align__ Half arr2[K];
  a.store(arr);
  b.store(arr + Vectorized<float>::size());
  convert(arr, arr2, K);
  return Vectorized<Half>::loadu(arr2);
}

#endif // defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)

#if defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)
inline void load_fp32_from_fp16(const c10::Half *data, Vectorized<float>& out) {
  auto values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data));
  __m256 out_values;
  cvtfp16_fp32(values, out_values);
  out = out_values;
}

inline void load_fp32_from_fp16(const c10::Half *data, Vectorized<float>& out1, Vectorized<float>& out2) {
  auto vec = Vectorized<c10::Half>::loadu(data);
  __m256 out1_values, out2_values;
  cvtfp16_fp32(vec, out1_values, out2_values);
  out1 = out1_values;
  out2 = out2_values;
}
#else // defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)
inline void load_fp32_from_fp16(const c10::Half *data, Vectorized<float>& out) {
  __at_align__ float values[Vectorized<float>::size()];
  for (const auto k : c10::irange(Vectorized<float>::size())) {
    values[k] = data[k];
  }
  out = Vectorized<float>::loadu(values);
}

inline void load_fp32_from_fp16(const c10::Half *data, Vectorized<float>& out1, Vectorized<float>& out2) {
  load_fp32_from_fp16(data, out1);
  data += Vectorized<float>::size();
  load_fp32_from_fp16(data, out2);
}
#endif

}}}

#pragma GCC diagnostic pop
