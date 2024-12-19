#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/util/irange.h>
#if (defined(CPU_CAPABILITY_AVX512))
#define SLEEF_STATIC_LIBS
#include <sleef.h>
#endif

namespace at {
namespace vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX512)

template <> class Vectorized<double> {
private:
  static constexpr __m512i zero_vector {0, 0, 0, 0, 0, 0, 0, 0};
public:
  // values needs to be public for compilation with clang
  // as vec512.h uses it
  __m512d values;
  using value_type = double;
  using size_type = int;
  static constexpr size_type size() {
    return 8;
  }
  Vectorized() {}
  Vectorized(__m512d v) : values(v) {}
  Vectorized(double val) {
    values = _mm512_set1_pd(val);
  }
  Vectorized(double val1, double val2, double val3, double val4,
         double val5, double val6, double val7, double val8) {
    values = _mm512_setr_pd(val1, val2, val3, val4, val5, val6, val7, val8);
  }
  operator __m512d() const {
    return values;
  }
  template <int64_t mask>
  static Vectorized<double> blend(const Vectorized<double>& a, const Vectorized<double>& b) {
    return _mm512_mask_blend_pd(mask, a.values, b.values);
  }
  static Vectorized<double> blendv(const Vectorized<double>& a, const Vectorized<double>& b,
                               const Vectorized<double>& mask) {
    auto all_ones = _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF);
    auto mmask = _mm512_cmp_epi64_mask(_mm512_castpd_si512(mask.values), all_ones, _MM_CMPINT_EQ);
    return _mm512_mask_blend_pd(mmask, a.values, b.values);
  }
  template<typename step_t>
  static Vectorized<double> arange(double base = 0., step_t step = static_cast<step_t>(1)) {
    return Vectorized<double>(base, base + step, base + 2 * step, base + 3 * step,
                          base + 4 * step, base + 5 * step, base + 6 * step,
                          base + 7 * step);
  }
  static Vectorized<double> set(const Vectorized<double>& a, const Vectorized<double>& b,
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
    }
    return b;
  }
  static Vectorized<double> loadu(const void* ptr, int64_t count = size()) {
    if (count == size())
      return _mm512_loadu_pd(reinterpret_cast<const double*>(ptr));

    __mmask8 mask = (1ULL << count) - 1;
    return _mm512_maskz_loadu_pd(mask, ptr);
  }
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      _mm512_storeu_pd(reinterpret_cast<double*>(ptr), values);
    } else if (count > 0) {
      __mmask8 mask = (1ULL << count) - 1;
      _mm512_mask_storeu_pd(reinterpret_cast<double*>(ptr), mask, values);
    }
  }
  const double& operator[](int idx) const  = delete;
  double& operator[](int idx) = delete;
  int zero_mask() const {
    // returns an integer mask where all zero elements are translated to 1-bit and others are translated to 0-bit
    __mmask8 cmp = _mm512_cmp_pd_mask(values, _mm512_set1_pd(0.0), _CMP_EQ_OQ);
    return static_cast<int32_t>(cmp);
  }
  Vectorized<double> isnan() const {
    auto cmp_mask = _mm512_cmp_pd_mask(values, _mm512_set1_pd(0.0), _CMP_UNORD_Q);
    return _mm512_castsi512_pd(_mm512_mask_set1_epi64(zero_vector, cmp_mask,
                                                      0xFFFFFFFFFFFFFFFF));
  }
  bool has_inf_nan() const {
    __m512d self_sub  = _mm512_sub_pd(values, values);
    return (_mm512_movepi8_mask(_mm512_castpd_si512(self_sub)) & 0x7777777777777777) != 0;
  }
  Vectorized<double> map(double (*const f)(double)) const {
    __at_align__ double tmp[size()];
    store(tmp);
    for (const auto i : c10::irange(size())) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  Vectorized<double> abs() const {
    auto mask = _mm512_set1_pd(-0.f);
    return _mm512_andnot_pd(mask, values);
  }
  Vectorized<double> angle() const {
    const auto zero_vec = _mm512_castsi512_pd(zero_vector);
    const auto nan_vec = _mm512_set1_pd(NAN);
    const auto not_nan_mask = _mm512_cmp_pd_mask(values, values, _CMP_EQ_OQ);
    const auto not_nan = _mm512_mask_set1_epi64(zero_vector, not_nan_mask,
                                                0xFFFFFFFFFFFFFFFF);
    const auto nan_mask = _mm512_cmp_pd_mask(_mm512_castsi512_pd(not_nan),
                                             zero_vec, _CMP_EQ_OQ);
    const auto pi = _mm512_set1_pd(c10::pi<double>);

    const auto neg_mask = _mm512_cmp_pd_mask(values, zero_vec, _CMP_LT_OQ);
    auto angle = _mm512_mask_blend_pd(neg_mask, zero_vec, pi);
    angle = _mm512_mask_blend_pd(nan_mask, angle, nan_vec);
    return angle;
  }
  Vectorized<double> real() const {
    return *this;
  }
  Vectorized<double> imag() const {
    return _mm512_set1_pd(0);
  }
  Vectorized<double> conj() const {
    return *this;
  }
  Vectorized<double> acos() const {
    return Vectorized<double>(Sleef_acosd8_u10(values));
  }
  Vectorized<double> acosh() const {
    return Vectorized<double>(Sleef_acoshd8_u10(values));
  }
  Vectorized<double> asin() const {
    return Vectorized<double>(Sleef_asind8_u10(values));
  }
  Vectorized<double> asinh() const {
    return Vectorized<double>(Sleef_asinhd8_u10(values));
  }
  Vectorized<double> atan() const {
    return Vectorized<double>(Sleef_atand8_u10(values));
  }
  Vectorized<double> atanh() const {
    return Vectorized<double>(Sleef_atanhd8_u10(values));
  }
  Vectorized<double> atan2(const Vectorized<double> &b) const {
    return Vectorized<double>(Sleef_atan2d8_u10(values, b));
  }
  Vectorized<double> copysign(const Vectorized<double> &sign) const {
    return Vectorized<double>(Sleef_copysignd8(values, sign));
  }
  Vectorized<double> erf() const {
    return Vectorized<double>(Sleef_erfd8_u10(values));
  }
  Vectorized<double> erfc() const {
    return Vectorized<double>(Sleef_erfcd8_u15(values));
  }
  Vectorized<double> erfinv() const {
    return map(calc_erfinv);
  }
  Vectorized<double> exp() const {
    return Vectorized<double>(Sleef_expd8_u10(values));
  }
  Vectorized<double> exp2() const {
    return Vectorized<double>(Sleef_exp2d8_u10(values));
  }
  Vectorized<double> expm1() const {
    return Vectorized<double>(Sleef_expm1d8_u10(values));
  }
  Vectorized<double> exp_u20() const {
    return exp();
  }
  Vectorized<double> fmod(const Vectorized<double>& q) const {
    return Vectorized<double>(Sleef_fmodd8(values, q));
  }
  Vectorized<double> hypot(const Vectorized<double> &b) const {
    return Vectorized<double>(Sleef_hypotd8_u05(values, b));
  }
  Vectorized<double> i0() const {
    return map(calc_i0);
  }
  Vectorized<double> i0e() const {
    return map(calc_i0e);
  }
  Vectorized<double> digamma() const {
    return map(calc_digamma);
  }
  Vectorized<double> igamma(const Vectorized<double> &x) const {
    __at_align__ double tmp[size()];
    __at_align__ double tmp_x[size()];
    store(tmp);
    x.store(tmp_x);
    for (const auto i : c10::irange(size())) {
      tmp[i] = calc_igamma(tmp[i], tmp_x[i]);
    }
    return loadu(tmp);
  }
  Vectorized<double> igammac(const Vectorized<double> &x) const {
    __at_align__ double tmp[size()];
    __at_align__ double tmp_x[size()];
    store(tmp);
    x.store(tmp_x);
    for (const auto i : c10::irange(size())) {
      tmp[i] = calc_igammac(tmp[i], tmp_x[i]);
    }
    return loadu(tmp);
  }
  Vectorized<double> log() const {
    return Vectorized<double>(Sleef_logd8_u10(values));
  }
  Vectorized<double> log2() const {
    return Vectorized<double>(Sleef_log2d8_u10(values));
  }
  Vectorized<double> log10() const {
    return Vectorized<double>(Sleef_log10d8_u10(values));
  }
  Vectorized<double> log1p() const {
    return Vectorized<double>(Sleef_log1pd8_u10(values));
  }
  Vectorized<double> sin() const {
    return Vectorized<double>(Sleef_sind8_u10(values));
  }
  Vectorized<double> sinh() const {
    return Vectorized<double>(Sleef_sinhd8_u10(values));
  }
  Vectorized<double> cos() const {
    return Vectorized<double>(Sleef_cosd8_u10(values));
  }
  Vectorized<double> cosh() const {
    return Vectorized<double>(Sleef_coshd8_u10(values));
  }
  Vectorized<double> ceil() const {
    return _mm512_ceil_pd(values);
  }
  Vectorized<double> floor() const {
    return _mm512_floor_pd(values);
  }
  Vectorized<double> frac() const;
  Vectorized<double> neg() const {
    return _mm512_xor_pd(_mm512_set1_pd(-0.), values);
  }
  Vectorized<double> nextafter(const Vectorized<double> &b) const {
    return Vectorized<double>(Sleef_nextafterd8(values, b));
  }
  Vectorized<double> round() const {
    return _mm512_roundscale_pd(values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
  Vectorized<double> tan() const {
    return Vectorized<double>(Sleef_tand8_u10(values));
  }
  Vectorized<double> tanh() const {
    return Vectorized<double>(Sleef_tanhd8_u10(values));
  }
  Vectorized<double> trunc() const {
    return _mm512_roundscale_pd(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  }
  Vectorized<double> lgamma() const {
    return Vectorized<double>(Sleef_lgammad8_u10(values));
  }
  Vectorized<double> sqrt() const {
    return _mm512_sqrt_pd(values);
  }
  Vectorized<double> reciprocal() const {
    return _mm512_div_pd(_mm512_set1_pd(1), values);
  }
  Vectorized<double> rsqrt() const {
    return _mm512_div_pd(_mm512_set1_pd(1), _mm512_sqrt_pd(values));
  }
  Vectorized<double> pow(const Vectorized<double> &b) const {
    return Vectorized<double>(Sleef_powd8_u10(values, b));
  }
  // Comparison using the _CMP_**_OQ predicate.
  //   `O`: get false if an operand is NaN
  //   `Q`: do not raise if an operand is NaN
  Vectorized<double> operator==(const Vectorized<double>& other) const {
    auto cmp_mask = _mm512_cmp_pd_mask(values, other.values, _CMP_EQ_OQ);
    return _mm512_castsi512_pd(_mm512_mask_set1_epi64(zero_vector, cmp_mask,
                                                      0xFFFFFFFFFFFFFFFF));
  }

  Vectorized<double> operator!=(const Vectorized<double>& other) const {
    auto cmp_mask = _mm512_cmp_pd_mask(values, other.values, _CMP_NEQ_UQ);
    return _mm512_castsi512_pd(_mm512_mask_set1_epi64(zero_vector, cmp_mask,
                                                      0xFFFFFFFFFFFFFFFF));
  }

  Vectorized<double> operator<(const Vectorized<double>& other) const {
    auto cmp_mask = _mm512_cmp_pd_mask(values, other.values, _CMP_LT_OQ);
    return _mm512_castsi512_pd(_mm512_mask_set1_epi64(zero_vector, cmp_mask,
                                                      0xFFFFFFFFFFFFFFFF));
  }

  Vectorized<double> operator<=(const Vectorized<double>& other) const {
    auto cmp_mask = _mm512_cmp_pd_mask(values, other.values, _CMP_LE_OQ);
    return _mm512_castsi512_pd(_mm512_mask_set1_epi64(zero_vector, cmp_mask,
                                                      0xFFFFFFFFFFFFFFFF));
  }

  Vectorized<double> operator>(const Vectorized<double>& other) const {
    auto cmp_mask = _mm512_cmp_pd_mask(values, other.values, _CMP_GT_OQ);
    return _mm512_castsi512_pd(_mm512_mask_set1_epi64(zero_vector, cmp_mask,
                                                      0xFFFFFFFFFFFFFFFF));
  }

  Vectorized<double> operator>=(const Vectorized<double>& other) const {
    auto cmp_mask = _mm512_cmp_pd_mask(values, other.values, _CMP_GE_OQ);
    return _mm512_castsi512_pd(_mm512_mask_set1_epi64(zero_vector, cmp_mask,
                                                      0xFFFFFFFFFFFFFFFF));
  }

  Vectorized<double> eq(const Vectorized<double>& other) const;
  Vectorized<double> ne(const Vectorized<double>& other) const;
  Vectorized<double> lt(const Vectorized<double>& other) const;
  Vectorized<double> le(const Vectorized<double>& other) const;
  Vectorized<double> gt(const Vectorized<double>& other) const;
  Vectorized<double> ge(const Vectorized<double>& other) const;
};

template <>
Vectorized<double> inline operator+(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm512_add_pd(a, b);
}

template <>
Vectorized<double> inline operator-(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm512_sub_pd(a, b);
}

template <>
Vectorized<double> inline operator*(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm512_mul_pd(a, b);
}

template <>
Vectorized<double> inline operator/(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm512_div_pd(a, b);
}

// frac. Implement this here so we can use subtraction.
inline Vectorized<double> Vectorized<double>::frac() const {
  return *this - this->trunc();
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<double> inline maximum(const Vectorized<double>& a, const Vectorized<double>& b) {
  auto zero_vec = _mm512_set1_epi64(0);
  Vectorized<double> max = _mm512_max_pd(a, b);
  auto isnan_mask = _mm512_cmp_pd_mask(a, b, _CMP_UNORD_Q);
  auto isnan = _mm512_castsi512_pd(_mm512_mask_set1_epi64(zero_vec, isnan_mask,
                                                          0xFFFFFFFFFFFFFFFF));
  // Exploit the fact that all-ones is a NaN.
  return _mm512_or_pd(max, isnan);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<double> inline minimum(const Vectorized<double>& a, const Vectorized<double>& b) {
  auto zero_vec = _mm512_set1_epi64(0);
  Vectorized<double> min = _mm512_min_pd(a, b);
  auto isnan_mask = _mm512_cmp_pd_mask(a, b, _CMP_UNORD_Q);
  auto isnan = _mm512_castsi512_pd(_mm512_mask_set1_epi64(zero_vec, isnan_mask,
                                                          0xFFFFFFFFFFFFFFFF));
  // Exploit the fact that all-ones is a NaN.
  return _mm512_or_pd(min, isnan);
}

template <>
Vectorized<double> inline clamp(const Vectorized<double>& a, const Vectorized<double>& min, const Vectorized<double>& max) {
  return _mm512_min_pd(max, _mm512_max_pd(min, a));
}

template <>
Vectorized<double> inline clamp_min(const Vectorized<double>& a, const Vectorized<double>& min) {
  return _mm512_max_pd(min, a);
}

template <>
Vectorized<double> inline clamp_max(const Vectorized<double>& a, const Vectorized<double>& max) {
  return _mm512_min_pd(max, a);
}

template <>
Vectorized<double> inline operator&(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm512_and_pd(a, b);
}

template <>
Vectorized<double> inline operator|(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm512_or_pd(a, b);
}

template <>
Vectorized<double> inline operator^(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm512_xor_pd(a, b);
}

inline Vectorized<double> Vectorized<double>::eq(const Vectorized<double>& other) const {
  return (*this == other) & Vectorized<double>(1.0);
}

inline Vectorized<double> Vectorized<double>::ne(const Vectorized<double>& other) const {
  return (*this != other) & Vectorized<double>(1.0);
}

inline Vectorized<double> Vectorized<double>::gt(const Vectorized<double>& other) const {
  return (*this > other) & Vectorized<double>(1.0);
}

inline Vectorized<double> Vectorized<double>::ge(const Vectorized<double>& other) const {
  return (*this >= other) & Vectorized<double>(1.0);
}

inline Vectorized<double> Vectorized<double>::lt(const Vectorized<double>& other) const {
  return (*this < other) & Vectorized<double>(1.0);
}

inline Vectorized<double> Vectorized<double>::le(const Vectorized<double>& other) const {
  return (*this <= other) & Vectorized<double>(1.0);
}

template <>
inline void convert(const double* src, double* dst, int64_t n) {
  int64_t i;
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (i = 0; i <= (n - Vectorized<double>::size()); i += Vectorized<double>::size()) {
    _mm512_storeu_pd(dst + i, _mm512_loadu_pd(src + i));
  }
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (; i < n; i++) {
    dst[i] = src[i];
  }
}

template <>
Vectorized<double> inline fmadd(const Vectorized<double>& a, const Vectorized<double>& b, const Vectorized<double>& c) {
  return _mm512_fmadd_pd(a, b, c);
}

template <>
Vectorized<double> inline fmsub(const Vectorized<double>& a, const Vectorized<double>& b, const Vectorized<double>& c) {
  return _mm512_fmsub_pd(a, b, c);
}

#endif

}}}
