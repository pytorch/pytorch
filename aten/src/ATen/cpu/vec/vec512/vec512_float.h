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

template <> class Vectorized<float> {
private:
  static constexpr __m512i zero_vec {0, 0, 0, 0, 0, 0, 0, 0};
public:
  __m512 values;
  using value_type = float;
  using size_type = int;
  static constexpr size_type size() {
    return 16;
  }
  Vectorized() {}
  Vectorized(__m512 v) : values(v) {}
  Vectorized(float val) {
    values = _mm512_set1_ps(val);
  }
  Vectorized(float val1, float val2, float val3, float val4,
         float val5, float val6, float val7, float val8,
         float val9, float val10, float val11, float val12,
         float val13, float val14, float val15, float val16) {
    values = _mm512_setr_ps(val1, val2, val3, val4, val5, val6, val7, val8,
                            val9, val10, val11, val12, val13, val14, val15, val16);
  }
  operator __m512() const {
    return values;
  }
  template <int64_t mask>
  static Vectorized<float> blend(const Vectorized<float>& a, const Vectorized<float>& b) {
    return _mm512_mask_blend_ps(mask, a.values, b.values);
  }
  static Vectorized<float> blendv(const Vectorized<float>& a, const Vectorized<float>& b,
                              const Vectorized<float>& mask) {
    auto all_ones = _mm512_set1_epi32(0xFFFFFFFF);
    auto mmask = _mm512_cmp_epi32_mask(_mm512_castps_si512(mask.values), all_ones, _MM_CMPINT_EQ);
    return _mm512_mask_blend_ps(mmask, a.values, b.values);
  }
  template<typename step_t>
  static Vectorized<float> arange(float base = 0.f, step_t step = static_cast<step_t>(1)) {
    return Vectorized<float>(
      base,            base +     step, base + 2 * step, base + 3 * step,
      base + 4 * step, base + 5 * step, base + 6 * step, base + 7 * step,
      base + 8 * step, base + 9 * step, base + 10 * step, base + 11 * step,
      base + 12 * step, base + 13 * step, base + 14 * step, base + 15 * step);
  }
  static Vectorized<float> set(const Vectorized<float>& a, const Vectorized<float>& b,
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
  static Vectorized<float> loadu(const void* ptr, int64_t count = size()) {
    if (count == size())
      return _mm512_loadu_ps(reinterpret_cast<const float*>(ptr));
    __at_align__ float tmp_values[size()];
    // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
    // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
    // instructions while a loop would be compiled to one instruction.
    for (const auto i : c10::irange(size())) {
      tmp_values[i] = 0.0;
    }
    std::memcpy(
        tmp_values, reinterpret_cast<const float*>(ptr), count * sizeof(float));
    return _mm512_loadu_ps(tmp_values);
  }
  void store(void* ptr, int64_t count = size()) const {
    if (count == size()) {
      _mm512_storeu_ps(reinterpret_cast<float*>(ptr), values);
    } else if (count > 0) {
      float tmp_values[size()];
      _mm512_storeu_ps(reinterpret_cast<float*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(float));
    }
  }
  const float& operator[](int idx) const  = delete;
  float& operator[](int idx) = delete;
  int zero_mask() const {
    // returns an integer mask where all zero elements are translated to 1-bit and others are translated to 0-bit
    __mmask16 cmp = _mm512_cmp_ps_mask(values, _mm512_set1_ps(0.0), _CMP_EQ_OQ);
    return static_cast<int32_t>(cmp);
  }
  Vectorized<float> isnan() const {
    auto mask =  _mm512_cmp_ps_mask(values, _mm512_set1_ps(0.0), _CMP_UNORD_Q);
    return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, mask,
                                                      0xFFFFFFFF));
  }
  bool has_inf_nan() const {
    __m512 self_sub  = _mm512_sub_ps(values, values);
    return (_mm512_movepi8_mask(_mm512_castps_si512(self_sub)) & 0x7777777777777777) != 0;
  }
  Vectorized<float> map(float (*const f)(float)) const {
    __at_align__ float tmp[size()];
    store(tmp);
    for (const auto i : c10::irange(size())) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  Vectorized<float> abs() const {
    auto mask = _mm512_set1_ps(-0.f);
    return _mm512_andnot_ps(mask, values);
  }
  Vectorized<float> angle() const {
    __m512 zero_vec = _mm512_set1_ps(0.f);
    const auto nan_vec = _mm512_set1_ps(NAN);
    const auto not_nan_mask = _mm512_cmp_ps_mask(values, values, _CMP_EQ_OQ);
    const auto not_nan_vec = _mm512_mask_set1_epi32(_mm512_castps_si512(zero_vec),
                                                    not_nan_mask, 0xFFFFFFFF);
    const auto nan_mask = _mm512_cmp_ps_mask(_mm512_castsi512_ps(not_nan_vec),
                                             zero_vec, _CMP_EQ_OQ);
    const auto pi = _mm512_set1_ps(c10::pi<double>);

    const auto neg_mask = _mm512_cmp_ps_mask(values, zero_vec, _CMP_LT_OQ);
    auto angle = _mm512_mask_blend_ps(neg_mask, zero_vec, pi);
    angle = _mm512_mask_blend_ps(nan_mask, angle, nan_vec);
    return angle;
  }
  Vectorized<float> real() const {
    return *this;
  }
  Vectorized<float> imag() const {
    return _mm512_set1_ps(0);
  }
  Vectorized<float> conj() const {
    return *this;
  }
  Vectorized<float> acos() const {
    return Vectorized<float>(Sleef_acosf16_u10(values));
  }
  Vectorized<float> asin() const {
    return Vectorized<float>(Sleef_asinf16_u10(values));
  }
  Vectorized<float> atan() const {
    return Vectorized<float>(Sleef_atanf16_u10(values));
  }
  Vectorized<float> atanh() const {
    return Vectorized<float>(Sleef_atanhf16_u10(values));
  }
  Vectorized<float> atan2(const Vectorized<float> &b) const {
    return Vectorized<float>(Sleef_atan2f16_u10(values, b));
  }
  Vectorized<float> copysign(const Vectorized<float> &sign) const {
    return Vectorized<float>(Sleef_copysignf16(values, sign));
  }
  Vectorized<float> erf() const {
    // constants
    const auto neg_zero_vec = _mm512_set1_ps(-0.f);
    const auto one_vec = _mm512_set1_ps(1.0f);
    const auto p = _mm512_set1_ps(0.3275911f);
    const auto p1 = _mm512_set1_ps(0.254829592f);
    const auto p2 = _mm512_set1_ps(-0.284496736f);
    const auto p3 = _mm512_set1_ps(1.421413741f);
    const auto p4 = _mm512_set1_ps(-1.453152027f);
    const auto p5 = _mm512_set1_ps(1.061405429f);
    // sign(x)
    auto sign_mask = _mm512_and_ps(neg_zero_vec, values);
    auto abs_vec = _mm512_abs_ps(values);
    // t = 1 / (p * abs(x) + 1)
    auto tmp0 = _mm512_fmadd_ps(p, abs_vec, one_vec);
    auto t = _mm512_div_ps(one_vec, tmp0);
    // r = p5 * t ^ 4 + p4 * t ^ 3 + p3 * t ^ 2 + p2 * t + p1
    auto tmp1 = _mm512_fmadd_ps(p5, t, p4);
    auto tmp2 = _mm512_fmadd_ps(tmp1, t, p3);
    auto tmp3 = _mm512_fmadd_ps(tmp2, t, p2);
    auto r = _mm512_fmadd_ps(tmp3, t, p1);
    // - exp(- x * x)
    auto pow_2 = _mm512_mul_ps(values, values);
    auto neg_pow_2 = _mm512_xor_ps(neg_zero_vec, pow_2);
    // auto tmp4 = exp(neg_pow_2);
    auto tmp4 = Vectorized<float>(Sleef_expf16_u10(neg_pow_2));
    auto tmp5 = _mm512_xor_ps(neg_zero_vec, tmp4);
    // erf(x) = sign(x) * (1 - r * t * exp(- x * x))
    auto tmp6 = _mm512_mul_ps(tmp5, t);
    auto tmp7 = _mm512_fmadd_ps(tmp6, r, one_vec);
    return _mm512_xor_ps(sign_mask, tmp7);
  }
  Vectorized<float> erfc() const {
    return Vectorized<float>(Sleef_erfcf16_u15(values));
  }
  Vectorized<float> erfinv() const {
    return map(calc_erfinv);
  }
  Vectorized<float> exp() const {
    return Vectorized<float>(Sleef_expf16_u10(values));
  }
  Vectorized<float> exp2() const {
    return Vectorized<float>(Sleef_exp2f16_u10(values));
  }
  Vectorized<float> expm1() const {
    return Vectorized<float>(Sleef_expm1f16_u10(values));
  }
  Vectorized<float> exp_u20() const {
    // A faster version of exp with ULP=20
    static __m512 vec_factorial_1 =
        _mm512_set1_ps(0.999999701f); // 1/factorial(1)
    static __m512 vec_factorial_2 =
        _mm512_set1_ps(0.499991506f); // 1/factorial(2)
    static __m512 vec_factorial_3 =
        _mm512_set1_ps(0.166676521f); // 1/factorial(3)
    static __m512 vec_factorial_4 =
        _mm512_set1_ps(0.0418978221f); // 1/factorial(4)
    static __m512 vec_factorial_5 =
        _mm512_set1_ps(0.00828929059f); // 1/factorial(5)
    static __m512 vec_exp_log2ef =
        (__m512)_mm512_set1_epi32(0x3fb8aa3b); // log2(e)
    static __m512 vec_half = _mm512_set1_ps(0.5f);
    static __m512 vec_one = _mm512_set1_ps(1.f);
    static __m512 vec_zero = _mm512_set1_ps(0.f);
    static __m512 vec_two = _mm512_set1_ps(2.f);
    static __m512 vec_ln2f = (__m512)_mm512_set1_epi32(0x3f317218); // ln(2)
    static __m512 vec_ln_flt_min = (__m512)_mm512_set1_epi32(0xc2aeac50);
    static __m512 vec_ln_flt_max = (__m512)_mm512_set1_epi32(0x42b17218);
    static __m512i vec_127 = _mm512_set1_epi32(0x0000007f);
    static int n_mantissa_bits = 23;

    // exp(x) =
    // = exp(n * ln(2) + r) // divide x by ln(2) and get quot and rem
    // = 2^n * exp(r) // simplify the exp(n*ln(2)) expression

    auto less_ln_flt_min_mask =
        _mm512_cmp_ps_mask(values, vec_ln_flt_min, 1 /*_CMP_LT_OS*/);
    auto vec_src = _mm512_min_ps(values, vec_ln_flt_max);
    vec_src = _mm512_max_ps(vec_src, vec_ln_flt_min);

    // fx = floorf(x * log2ef + 0.5)
    auto vec_fx = _mm512_fmadd_ps(vec_src, vec_exp_log2ef, vec_half);
    auto vec_fx_i = _mm512_cvt_roundps_epi32(
        vec_fx, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    vec_fx = _mm512_cvtepi32_ps(vec_fx_i);

    // x = x - fx * ln2
    auto vec_exp_poly = _mm512_fnmadd_ps(vec_fx, vec_ln2f, vec_src);

    // compute polynomial
    auto vec_res =
        _mm512_fmadd_ps(vec_exp_poly, vec_factorial_5, vec_factorial_4);
    vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_3);
    vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_2);
    vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_1);
    vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_one);

    // compute 2^(n-1)
    auto vec_exp_number = _mm512_sub_ps(vec_fx, vec_one);
    auto vec_exp_number_i = _mm512_cvtps_epi32(vec_exp_number);
    auto vec_two_pow_n_i = _mm512_add_epi32(vec_exp_number_i, vec_127);
    vec_two_pow_n_i = _mm512_slli_epi32(vec_two_pow_n_i, n_mantissa_bits);
    auto vec_two_pow_n = (__m512)vec_two_pow_n_i;
    vec_two_pow_n =
        _mm512_mask_blend_ps(less_ln_flt_min_mask, vec_two_pow_n, vec_zero);

    // y = y * 2^n
    vec_res = _mm512_mul_ps(vec_res, vec_two_pow_n);
    vec_res = _mm512_mul_ps(vec_res, vec_two);
    return vec_res;
  }
  Vectorized<float> fmod(const Vectorized<float>& q) const {
    return Vectorized<float>(Sleef_fmodf16(values, q));
  }
  Vectorized<float> log() const {
    return Vectorized<float>(Sleef_logf16_u10(values));
  }
  Vectorized<float> log2() const {
    return Vectorized<float>(Sleef_log2f16_u10(values));
  }
  Vectorized<float> log10() const {
    return Vectorized<float>(Sleef_log10f16_u10(values));
  }
  Vectorized<float> log1p() const {
    return Vectorized<float>(Sleef_log1pf16_u10(values));
  }
  Vectorized<float> frac() const;
  Vectorized<float> sin() const {
    return Vectorized<float>(Sleef_sinf16_u35(values));
  }
  Vectorized<float> sinh() const {
    return Vectorized<float>(Sleef_sinhf16_u10(values));
  }
  Vectorized<float> cos() const {
    return Vectorized<float>(Sleef_cosf16_u35(values));
  }
  Vectorized<float> cosh() const {
    return Vectorized<float>(Sleef_coshf16_u10(values));
  }
  Vectorized<float> ceil() const {
    return _mm512_ceil_ps(values);
  }
  Vectorized<float> floor() const {
    return _mm512_floor_ps(values);
  }
  Vectorized<float> hypot(const Vectorized<float> &b) const {
    return Vectorized<float>(Sleef_hypotf16_u05(values, b));
  }
  Vectorized<float> i0() const {
    return map(calc_i0);
  }
  Vectorized<float> i0e() const {
    return map(calc_i0e);
  }
  Vectorized<float> digamma() const {
    return map(calc_digamma);
  }
  Vectorized<float> igamma(const Vectorized<float> &x) const {
    __at_align__ float tmp[size()];
    __at_align__ float tmp_x[size()];
    store(tmp);
    x.store(tmp_x);
    for (const auto i : c10::irange(size())) {
      tmp[i] = calc_igamma(tmp[i], tmp_x[i]);
    }
    return loadu(tmp);
  }
  Vectorized<float> igammac(const Vectorized<float> &x) const {
    __at_align__ float tmp[size()];
    __at_align__ float tmp_x[size()];
    store(tmp);
    x.store(tmp_x);
    for (const auto i : c10::irange(size())) {
      tmp[i] = calc_igammac(tmp[i], tmp_x[i]);
    }
    return loadu(tmp);
  }
  Vectorized<float> neg() const {
    return _mm512_xor_ps(_mm512_set1_ps(-0.f), values);
  }
  Vectorized<float> nextafter(const Vectorized<float> &b) const {
    return Vectorized<float>(Sleef_nextafterf16(values, b));
  }
  Vectorized<float> round() const {
    return _mm512_roundscale_ps(values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
  Vectorized<float> tan() const {
    return Vectorized<float>(Sleef_tanf16_u10(values));
  }
  Vectorized<float> tanh() const {
    return Vectorized<float>(Sleef_tanhf16_u10(values));
  }
  Vectorized<float> trunc() const {
    return _mm512_roundscale_ps(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  }
  Vectorized<float> lgamma() const {
    return Vectorized<float>(Sleef_lgammaf16_u10(values));
  }
  Vectorized<float> sqrt() const {
    return _mm512_sqrt_ps(values);
  }
  Vectorized<float> reciprocal() const {
    return _mm512_div_ps(_mm512_set1_ps(1), values);
  }
  Vectorized<float> rsqrt() const {
    return _mm512_div_ps(_mm512_set1_ps(1), _mm512_sqrt_ps(values));
  }
  Vectorized<float> pow(const Vectorized<float> &b) const {
    return Vectorized<float>(Sleef_powf16_u10(values, b));
  }
  // Comparison using the _CMP_**_OQ predicate.
  //   `O`: get false if an operand is NaN
  //   `Q`: do not raise if an operand is NaN
  Vectorized<float> operator==(const Vectorized<float>& other) const {
    auto mask = _mm512_cmp_ps_mask(values, other.values, _CMP_EQ_OQ);
    return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, mask,
                                                      0xFFFFFFFF));
  }

  Vectorized<float> operator!=(const Vectorized<float>& other) const {
    auto mask = _mm512_cmp_ps_mask(values, other.values, _CMP_NEQ_UQ);
    return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, mask,
                                                      0xFFFFFFFF));
  }

  Vectorized<float> operator<(const Vectorized<float>& other) const {
    auto mask = _mm512_cmp_ps_mask(values, other.values, _CMP_LT_OQ);
    return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, mask,
                                                      0xFFFFFFFF));
  }

  Vectorized<float> operator<=(const Vectorized<float>& other) const {
    auto mask = _mm512_cmp_ps_mask(values, other.values, _CMP_LE_OQ);
    return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, mask,
                                                      0xFFFFFFFF));
  }

  Vectorized<float> operator>(const Vectorized<float>& other) const {
    auto mask = _mm512_cmp_ps_mask(values, other.values, _CMP_GT_OQ);
    return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, mask,
                                                      0xFFFFFFFF));
  }

  Vectorized<float> operator>=(const Vectorized<float>& other) const {
    auto mask = _mm512_cmp_ps_mask(values, other.values, _CMP_GE_OQ);
    return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, mask,
                                                      0xFFFFFFFF));
  }

  Vectorized<float> eq(const Vectorized<float>& other) const;
  Vectorized<float> ne(const Vectorized<float>& other) const;
  Vectorized<float> gt(const Vectorized<float>& other) const;
  Vectorized<float> ge(const Vectorized<float>& other) const;
  Vectorized<float> lt(const Vectorized<float>& other) const;
  Vectorized<float> le(const Vectorized<float>& other) const;
};

template <>
Vectorized<float> inline operator+(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm512_add_ps(a, b);
}

template <>
Vectorized<float> inline operator-(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm512_sub_ps(a, b);
}

template <>
Vectorized<float> inline operator*(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm512_mul_ps(a, b);
}

template <>
Vectorized<float> inline operator/(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm512_div_ps(a, b);
}

// frac. Implement this here so we can use subtraction
inline Vectorized<float> Vectorized<float>::frac() const {
  return *this - this->trunc();
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<float> inline maximum(const Vectorized<float>& a, const Vectorized<float>& b) {
  auto zero_vec = _mm512_set1_epi32(0);
  auto max = _mm512_max_ps(a, b);
  auto isnan_mask = _mm512_cmp_ps_mask(a, b, _CMP_UNORD_Q);
  auto isnan = _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, isnan_mask,
                                                          0xFFFFFFFF));
  // Exploit the fact that all-ones is a NaN.
  return _mm512_or_ps(max, isnan);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<float> inline minimum(const Vectorized<float>& a, const Vectorized<float>& b) {
  auto zero_vec = _mm512_set1_epi32(0);
  auto min = _mm512_min_ps(a, b);
  auto isnan_mask = _mm512_cmp_ps_mask(a, b, _CMP_UNORD_Q);
  auto isnan = _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, isnan_mask,
                                                          0xFFFFFFFF));
  // Exploit the fact that all-ones is a NaN.
  return _mm512_or_ps(min, isnan);
}

template <>
Vectorized<float> inline clamp(const Vectorized<float>& a, const Vectorized<float>& min, const Vectorized<float>& max) {
  return _mm512_min_ps(max, _mm512_max_ps(min, a));
}

template <>
Vectorized<float> inline clamp_max(const Vectorized<float>& a, const Vectorized<float>& max) {
  return _mm512_min_ps(max, a);
}

template <>
Vectorized<float> inline clamp_min(const Vectorized<float>& a, const Vectorized<float>& min) {
  return _mm512_max_ps(min, a);
}

template <>
Vectorized<float> inline operator&(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm512_and_ps(a, b);
}

template <>
Vectorized<float> inline operator|(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm512_or_ps(a, b);
}

template <>
Vectorized<float> inline operator^(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm512_xor_ps(a, b);
}

inline Vectorized<float> Vectorized<float>::eq(const Vectorized<float>& other) const {
  return (*this == other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::ne(const Vectorized<float>& other) const {
  return (*this != other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::gt(const Vectorized<float>& other) const {
  return (*this > other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::ge(const Vectorized<float>& other) const {
  return (*this >= other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::lt(const Vectorized<float>& other) const {
  return (*this < other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::le(const Vectorized<float>& other) const {
  return (*this <= other) & Vectorized<float>(1.0f);
}

template <>
inline void convert(const float* src, float* dst, int64_t n) {
  int64_t i;
#pragma unroll
  for (i = 0; i <= (n - Vectorized<float>::size()); i += Vectorized<float>::size()) {
    _mm512_storeu_ps(dst + i, _mm512_loadu_ps(src + i));
  }
#pragma unroll
  for (; i < n; i++) {
    dst[i] = src[i];
  }
}

template <>
Vectorized<float> inline fmadd(const Vectorized<float>& a, const Vectorized<float>& b, const Vectorized<float>& c) {
  return _mm512_fmadd_ps(a, b, c);
}

template <>
Vectorized<float> inline fmsub(const Vectorized<float>& a, const Vectorized<float>& b, const Vectorized<float>& c) {
  return _mm512_fmsub_ps(a, b, c);
}

// TODO(jgong5): rewrite with ATEN vectorized (need to add unpack and shuffle)
// Used by Inductor CPP codegen
// Code referred to FBGEMM:
// https://github.com/pytorch/FBGEMM/blob/39a423e4ad1a04b77fea81c7d09c3e6f8984fae9/src/UtilsAvx512.cc#LL19C6-L19C6
// 16 * 6 = 96 instructions
template<>
inline void transpose_mxn<float, 16, 16>(
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
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
  __m512 a = _mm512_loadu_ps(&src[0 * ld_src]);
  __m512 b = _mm512_loadu_ps(&src[1 * ld_src]);
  __m512 c = _mm512_loadu_ps(&src[2 * ld_src]);
  __m512 d = _mm512_loadu_ps(&src[3 * ld_src]);
  __m512 e = _mm512_loadu_ps(&src[4 * ld_src]);
  __m512 f = _mm512_loadu_ps(&src[5 * ld_src]);
  __m512 g = _mm512_loadu_ps(&src[6 * ld_src]);
  __m512 h = _mm512_loadu_ps(&src[7 * ld_src]);
  __m512 i = _mm512_loadu_ps(&src[8 * ld_src]);
  __m512 j = _mm512_loadu_ps(&src[9 * ld_src]);
  __m512 k = _mm512_loadu_ps(&src[10 * ld_src]);
  __m512 l = _mm512_loadu_ps(&src[11 * ld_src]);
  __m512 m = _mm512_loadu_ps(&src[12 * ld_src]);
  __m512 n = _mm512_loadu_ps(&src[13 * ld_src]);
  __m512 o = _mm512_loadu_ps(&src[14 * ld_src]);
  __m512 p = _mm512_loadu_ps(&src[15 * ld_src]);

  __m512 ta, tb, tc, td, te, tf, tg, th, ti, tj, tk, tl, tm, tn, to, tq;
  // unpacking and interleaving 32-bit elements
  // a0  b0  a1  b1  a4  b4  a5  b5  a8  b8  a9  b9  a12  b12 a13 b13
  // a2  b2  a3  b3  a6  b6  a7  b7  a10 b10 a11 b11 a14  b14 a15 b15
  // c0  d0  c1  d1 ...
  // c2  d2  c3  d3 ...
  // e0  f0  e1  f1 ...
  // e2  f2  e3  f3 ...
  // g0  h0  g1  h1 ...
  // g2  h2  g3  h3 ...
  // i0  ...
  // i2  ...
  // k0  ...
  // k2  ...
  // m0  ...
  // m2  ...
  // o0  ...
  // o1  ...
  ta = _mm512_unpacklo_ps(a, b);
  tb = _mm512_unpackhi_ps(a, b);
  tc = _mm512_unpacklo_ps(c, d);
  td = _mm512_unpackhi_ps(c, d);
  te = _mm512_unpacklo_ps(e, f);
  tf = _mm512_unpackhi_ps(e, f);
  tg = _mm512_unpacklo_ps(g, h);
  th = _mm512_unpackhi_ps(g, h);
  ti = _mm512_unpacklo_ps(i, j);
  tj = _mm512_unpackhi_ps(i, j);
  tk = _mm512_unpacklo_ps(k, l);
  tl = _mm512_unpackhi_ps(k, l);
  tm = _mm512_unpacklo_ps(m, n);
  tn = _mm512_unpackhi_ps(m, n);
  to = _mm512_unpacklo_ps(o, p);
  tq = _mm512_unpackhi_ps(o, p);

  // unpacking and interleaving 64-bit elements
  //  a0  b0  c0  d0  a4  b4  c4  d4  a8  b8  c8  d8  a12 b12 c12 d12
  //  a1  b1  c1  d1 ...
  //  a2  b2  c2  d2 ...
  //  a3  b3  c3  d3 ...
  //  e0  f0  g0  h0  e4  f4  g4  h4  e8  f8  g8  h8  e12 f12 g12 h12
  //  e1  f1  g1  h1 ...
  //  e2  f2  g2  h2 ...
  //  e3  f3  g3  h3 ...
  //  i0  j0  k0  l0 ...
  //  i1  j1  k1  l1 ...
  //  i2  j2  k2  l2 ...
  //  i3  j3  k3  l3 ...
  //  m0  n0  o0  p0 ...
  //  m1  n1  o1  p1 ...
  //  m2  n2  o2  p2 ...
  //  m3  n3  o3  p3 ...
  a = _mm512_castpd_ps(
      _mm512_unpacklo_pd(_mm512_castps_pd(ta), _mm512_castps_pd(tc)));
  b = _mm512_castpd_ps(
      _mm512_unpackhi_pd(_mm512_castps_pd(ta), _mm512_castps_pd(tc)));
  c = _mm512_castpd_ps(
      _mm512_unpacklo_pd(_mm512_castps_pd(tb), _mm512_castps_pd(td)));
  d = _mm512_castpd_ps(
      _mm512_unpackhi_pd(_mm512_castps_pd(tb), _mm512_castps_pd(td)));
  e = _mm512_castpd_ps(
      _mm512_unpacklo_pd(_mm512_castps_pd(te), _mm512_castps_pd(tg)));
  f = _mm512_castpd_ps(
      _mm512_unpackhi_pd(_mm512_castps_pd(te), _mm512_castps_pd(tg)));
  g = _mm512_castpd_ps(
      _mm512_unpacklo_pd(_mm512_castps_pd(tf), _mm512_castps_pd(th)));
  h = _mm512_castpd_ps(
      _mm512_unpackhi_pd(_mm512_castps_pd(tf), _mm512_castps_pd(th)));
  i = _mm512_castpd_ps(
      _mm512_unpacklo_pd(_mm512_castps_pd(ti), _mm512_castps_pd(tk)));
  j = _mm512_castpd_ps(
      _mm512_unpackhi_pd(_mm512_castps_pd(ti), _mm512_castps_pd(tk)));
  k = _mm512_castpd_ps(
      _mm512_unpacklo_pd(_mm512_castps_pd(tj), _mm512_castps_pd(tl)));
  l = _mm512_castpd_ps(
      _mm512_unpackhi_pd(_mm512_castps_pd(tj), _mm512_castps_pd(tl)));
  m = _mm512_castpd_ps(
      _mm512_unpacklo_pd(_mm512_castps_pd(tm), _mm512_castps_pd(to)));
  n = _mm512_castpd_ps(
      _mm512_unpackhi_pd(_mm512_castps_pd(tm), _mm512_castps_pd(to)));
  o = _mm512_castpd_ps(
      _mm512_unpacklo_pd(_mm512_castps_pd(tn), _mm512_castps_pd(tq)));
  p = _mm512_castpd_ps(
      _mm512_unpackhi_pd(_mm512_castps_pd(tn), _mm512_castps_pd(tq)));

  //  shuffle 128-bits (composed of 4 32-bit elements)
  //  a0  b0  c0  d0  a8  b8  c8  d8  e0  f0  g0  h0  e8  f8  g8  h8
  //  a1  b1  c1  d1 ...
  //  a2  b2  c2  d2 ...
  //  a3  b3  c3  d3 ...
  //  a4  b4  c4  d4 ...
  //  a5  b5  c5  d5 ...
  //  a6  b6  c6  d6 ...
  //  a7  b7  c7  d7 ...
  //  i0  j0  k0  l0  i8  j8  k8  l8  m0  n0  o0  p0  m8  n8  o8  p8
  //  i1  j1  k1  l1 ...
  //  i2  j2  k2  l2 ...
  //  i3  j3  k3  l3 ...
  //  i4  j4  k4  l4 ...
  //  i5  j5  k5  l5 ...
  //  i6  j6  k6  l6 ...
  //  i7  j7  k7  l7 ...
  ta = _mm512_shuffle_f32x4(a, e, 0x88);
  tb = _mm512_shuffle_f32x4(b, f, 0x88);
  tc = _mm512_shuffle_f32x4(c, g, 0x88);
  td = _mm512_shuffle_f32x4(d, h, 0x88);
  te = _mm512_shuffle_f32x4(a, e, 0xdd);
  tf = _mm512_shuffle_f32x4(b, f, 0xdd);
  tg = _mm512_shuffle_f32x4(c, g, 0xdd);
  th = _mm512_shuffle_f32x4(d, h, 0xdd);
  ti = _mm512_shuffle_f32x4(i, m, 0x88);
  tj = _mm512_shuffle_f32x4(j, n, 0x88);
  tk = _mm512_shuffle_f32x4(k, o, 0x88);
  tl = _mm512_shuffle_f32x4(l, p, 0x88);
  tm = _mm512_shuffle_f32x4(i, m, 0xdd);
  tn = _mm512_shuffle_f32x4(j, n, 0xdd);
  to = _mm512_shuffle_f32x4(k, o, 0xdd);
  tq = _mm512_shuffle_f32x4(l, p, 0xdd);

  //  shuffle 128-bits (composed of 4 32-bit elements)
  //  a0  b0  c0  d0  ...  o0
  //  a1  b1  c1  d1  ...  o1
  //  a2  b2  c2  d2  ...  o2
  //  a3  b3  c3  d3  ...  o3
  //  a4  ...
  //  a5  ...
  //  a6  ...
  //  a7  ...
  //  a8  ...
  //  a9  ...
  //  a10 ...
  //  a11 ...
  //  a12 ...
  //  a13 ...
  //  a14 ...
  //  a15 b15 c15 d15 ...  o15
  a = _mm512_shuffle_f32x4(ta, ti, 0x88);
  b = _mm512_shuffle_f32x4(tb, tj, 0x88);
  c = _mm512_shuffle_f32x4(tc, tk, 0x88);
  d = _mm512_shuffle_f32x4(td, tl, 0x88);
  e = _mm512_shuffle_f32x4(te, tm, 0x88);
  f = _mm512_shuffle_f32x4(tf, tn, 0x88);
  g = _mm512_shuffle_f32x4(tg, to, 0x88);
  h = _mm512_shuffle_f32x4(th, tq, 0x88);
  i = _mm512_shuffle_f32x4(ta, ti, 0xdd);
  j = _mm512_shuffle_f32x4(tb, tj, 0xdd);
  k = _mm512_shuffle_f32x4(tc, tk, 0xdd);
  l = _mm512_shuffle_f32x4(td, tl, 0xdd);
  m = _mm512_shuffle_f32x4(te, tm, 0xdd);
  n = _mm512_shuffle_f32x4(tf, tn, 0xdd);
  o = _mm512_shuffle_f32x4(tg, to, 0xdd);
  p = _mm512_shuffle_f32x4(th, tq, 0xdd);

  // store from registers to dst
  _mm512_storeu_ps(&dst[0 * ld_dst], a);
  _mm512_storeu_ps(&dst[1 * ld_dst], b);
  _mm512_storeu_ps(&dst[2 * ld_dst], c);
  _mm512_storeu_ps(&dst[3 * ld_dst], d);
  _mm512_storeu_ps(&dst[4 * ld_dst], e);
  _mm512_storeu_ps(&dst[5 * ld_dst], f);
  _mm512_storeu_ps(&dst[6 * ld_dst], g);
  _mm512_storeu_ps(&dst[7 * ld_dst], h);
  _mm512_storeu_ps(&dst[8 * ld_dst], i);
  _mm512_storeu_ps(&dst[9 * ld_dst], j);
  _mm512_storeu_ps(&dst[10 * ld_dst], k);
  _mm512_storeu_ps(&dst[11 * ld_dst], l);
  _mm512_storeu_ps(&dst[12 * ld_dst], m);
  _mm512_storeu_ps(&dst[13 * ld_dst], n);
  _mm512_storeu_ps(&dst[14 * ld_dst], o);
  _mm512_storeu_ps(&dst[15 * ld_dst], p);
}

#endif

}}}
