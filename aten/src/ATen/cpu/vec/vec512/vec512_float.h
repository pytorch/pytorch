#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/vec512/intrinsics.h>
#include <ATen/cpu/vec/vec512/vec512_base.h>
#if defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)
#include <sleef.h>
#endif

namespace at {
namespace vec {
// See Note [Acceptable use of anonymous namespace in header]
namespace {

#if defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)

template <> class Vectorize<float> {
private:
  __m512 values;
  static constexpr __m512i zero_vec {0, 0, 0, 0, 0, 0, 0, 0};
public:
  using value_type = float;
  using size_type = int;
  static constexpr size_type size() {
    return 16;
  }
  Vectorize() {}
  Vectorize(__m512 v) : values(v) {}
  Vectorize(float val) {
    values = _mm512_set1_ps(val);
  }
  Vectorize(float val1, float val2, float val3, float val4,
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
  static Vectorize<float> blend(const Vectorize<float>& a, const Vectorize<float>& b) {
    return _mm512_mask_blend_ps(mask, a.values, b.values);
  }
  static Vectorize<float> blendv(const Vectorize<float>& a, const Vectorize<float>& b,
                              const Vectorize<float>& mask) {
    auto all_ones = _mm512_set1_epi32(0xFFFFFFFF);
    auto mmask = _mm512_cmp_epi32_mask(_mm512_castps_si512(mask.values), all_ones, _MM_CMPINT_EQ);
    return _mm512_mask_blend_ps(mmask, a.values, b.values);
  }
  template<typename step_t>
  static Vectorize<float> arange(float base = 0.f, step_t step = static_cast<step_t>(1)) {
    return Vectorize<float>(
      base,            base +     step, base + 2 * step, base + 3 * step,
      base + 4 * step, base + 5 * step, base + 6 * step, base + 7 * step,
      base + 8 * step, base + 9 * step, base + 10 * step, base + 11 * step,
      base + 12 * step, base + 13 * step, base + 14 * step, base + 15 * step);
  }
  static Vectorize<float> set(const Vectorize<float>& a, const Vectorize<float>& b,
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
  static Vectorize<float> loadu(const void* ptr, int64_t count = size()) {
    if (count == size())
      return _mm512_loadu_ps(reinterpret_cast<const float*>(ptr));
    __at_align64__ float tmp_values[size()];
    // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
    // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
    // instructions while a loop would be compiled to one instruction.
    for (auto i = 0; i < size(); ++i) {
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
  Vectorize<float> isnan() const {
    auto mask =  _mm512_cmp_ps_mask(values, _mm512_set1_ps(0.0f), _CMP_UNORD_Q);
    return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, mask,
                                                      0xFFFFFFFF));
  }
  Vectorize<float> map(float (*f)(float)) const {
    __at_align64__ float tmp[size()];
    store(tmp);
    for (int64_t i = 0; i < size(); i++) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  Vectorize<float> abs() const {
    auto mask = _mm512_set1_ps(-0.f);
    return _mm512_andnot_ps(mask, values);
  }
  Vectorize<float> angle() const {
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
  Vectorize<float> real() const {
    return *this;
  }
  Vectorize<float> imag() const {
    return _mm512_set1_ps(0);
  }
  Vectorize<float> conj() const {
    return *this;
  }
  Vectorize<float> acos() const {
    return Vectorize<float>(Sleef_acosf16_u10(values));
  }
  Vectorize<float> asin() const {
    return Vectorize<float>(Sleef_asinf16_u10(values));
  }
  Vectorize<float> atan() const {
    return Vectorize<float>(Sleef_atanf16_u10(values));
  }
  Vectorize<float> atan2(const Vectorize<float> &b) const {
    return Vectorize<float>(Sleef_atan2f16_u10(values, b));
  }
  Vectorize<float> copysign(const Vectorize<float> &sign) const {
    return Vectorize<float>(Sleef_copysignf16(values, sign));
  }
  Vectorize<float> erf() const {
    return Vectorize<float>(Sleef_erff16_u10(values));
  }
  Vectorize<float> erfc() const {
    return Vectorize<float>(Sleef_erfcf16_u15(values));
  }
  Vectorize<float> erfinv() const {
    return map(calc_erfinv);
  }
  Vectorize<float> exp() const {
    return Vectorize<float>(Sleef_expf16_u10(values));
  }
  Vectorize<float> expm1() const {
    return Vectorize<float>(Sleef_expm1f16_u10(values));
  }
  Vectorize<float> fmod(const Vectorize<float>& q) const {
    return Vectorize<float>(Sleef_fmodf16(values, q));
  }
  Vectorize<float> log() const {
    return Vectorize<float>(Sleef_logf16_u10(values));
  }
  Vectorize<float> log2() const {
    return Vectorize<float>(Sleef_log2f16_u10(values));
  }
  Vectorize<float> log10() const {
    return Vectorize<float>(Sleef_log10f16_u10(values));
  }
  Vectorize<float> log1p() const {
    return Vectorize<float>(Sleef_log1pf16_u10(values));
  }
  Vectorize<float> frac() const;
  Vectorize<float> sin() const {
    return Vectorize<float>(Sleef_sinf16_u10(values));
  }
  Vectorize<float> sinh() const {
    return Vectorize<float>(Sleef_sinhf16_u10(values));
  }
  Vectorize<float> cos() const {
    return Vectorize<float>(Sleef_cosf16_u10(values));
  }
  Vectorize<float> cosh() const {
    return Vectorize<float>(Sleef_coshf16_u10(values));
  }
  Vectorize<float> ceil() const {
    return _mm512_ceil_ps(values);
  }
  Vectorize<float> floor() const {
    return _mm512_floor_ps(values);
  }
  Vectorize<float> hypot(const Vectorize<float> &b) const {
    return Vectorize<float>(Sleef_hypotf16_u05(values, b));
  }
  Vectorize<float> i0() const {
    return map(calc_i0);
  }
  Vectorize<float> i0e() const {
    return map(calc_i0e);
  }
  Vectorize<float> igamma(const Vectorize<float> &x) const {
    __at_align64__ float tmp[size()];
    __at_align64__ float tmp_x[size()];
    store(tmp);
    x.store(tmp_x);
    for (int64_t i = 0; i < size(); i++) {
      tmp[i] = calc_igamma(tmp[i], tmp_x[i]);
    }
    return loadu(tmp);
  }
  Vectorize<float> igammac(const Vectorize<float> &x) const {
    __at_align64__ float tmp[size()];
    __at_align64__ float tmp_x[size()];
    store(tmp);
    x.store(tmp_x);
    for (int64_t i = 0; i < size(); i++) {
      tmp[i] = calc_igammac(tmp[i], tmp_x[i]);
    }
    return loadu(tmp);
  }
  Vectorize<float> neg() const {
    return _mm512_xor_ps(_mm512_set1_ps(-0.f), values);
  }
  Vectorize<float> nextafter(const Vectorize<float> &b) const {
    return Vectorize<float>(Sleef_nextafterf16(values, b));
  }
  Vectorize<float> round() const {
    return _mm512_roundscale_ps(values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
  Vectorize<float> tan() const {
    return Vectorize<float>(Sleef_tanf16_u10(values));
  }
  Vectorize<float> tanh() const {
    return Vectorize<float>(Sleef_tanhf16_u10(values));
  }
  Vectorize<float> trunc() const {
    return _mm512_roundscale_ps(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  }
  Vectorize<float> lgamma() const {
    return Vectorize<float>(Sleef_lgammaf16_u10(values));
  }
  Vectorize<float> sqrt() const {
    return _mm512_sqrt_ps(values);
  }
  Vectorize<float> reciprocal() const {
    return _mm512_div_ps(_mm512_set1_ps(1), values);
  }
  Vectorize<float> rsqrt() const {
    return _mm512_div_ps(_mm512_set1_ps(1), _mm512_sqrt_ps(values));
  }
  Vectorize<float> pow(const Vectorize<float> &b) const {
    return Vectorize<float>(Sleef_powf16_u10(values, b));
  }
  // Comparison using the _CMP_**_OQ predicate.
  //   `O`: get false if an operand is NaN
  //   `Q`: do not raise if an operand is NaN
  Vectorize<float> operator==(const Vectorize<float>& other) const {
    auto mask = _mm512_cmp_ps_mask(values, other.values, _CMP_EQ_OQ);
    return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, mask,
                                                      0xFFFFFFFF));
  }

  Vectorize<float> operator!=(const Vectorize<float>& other) const {
    auto mask = _mm512_cmp_ps_mask(values, other.values, _CMP_NEQ_OQ);
    return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, mask,
                                                      0xFFFFFFFF));
  }

  Vectorize<float> operator<(const Vectorize<float>& other) const {
    auto mask = _mm512_cmp_ps_mask(values, other.values, _CMP_LT_OQ);
    return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, mask,
                                                      0xFFFFFFFF));
  }

  Vectorize<float> operator<=(const Vectorize<float>& other) const {
    auto mask = _mm512_cmp_ps_mask(values, other.values, _CMP_LE_OQ);
    return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, mask,
                                                      0xFFFFFFFF));
  }

  Vectorize<float> operator>(const Vectorize<float>& other) const {
    auto mask = _mm512_cmp_ps_mask(values, other.values, _CMP_GT_OQ);
    return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, mask,
                                                      0xFFFFFFFF));
  }

  Vectorize<float> operator>=(const Vectorize<float>& other) const {
    auto mask = _mm512_cmp_ps_mask(values, other.values, _CMP_GE_OQ);
    return _mm512_castsi512_ps(_mm512_mask_set1_epi32(zero_vec, mask,
                                                      0xFFFFFFFF));
  }

  Vectorize<float> eq(const Vectorize<float>& other) const;
  Vectorize<float> ne(const Vectorize<float>& other) const;
  Vectorize<float> gt(const Vectorize<float>& other) const;
  Vectorize<float> ge(const Vectorize<float>& other) const;
  Vectorize<float> lt(const Vectorize<float>& other) const;
  Vectorize<float> le(const Vectorize<float>& other) const;
};

template <>
Vectorize<float> inline operator+(const Vectorize<float>& a, const Vectorize<float>& b) {
  return _mm512_add_ps(a, b);
}

template <>
Vectorize<float> inline operator-(const Vectorize<float>& a, const Vectorize<float>& b) {
  return _mm512_sub_ps(a, b);
}

template <>
Vectorize<float> inline operator*(const Vectorize<float>& a, const Vectorize<float>& b) {
  return _mm512_mul_ps(a, b);
}

template <>
Vectorize<float> inline operator/(const Vectorize<float>& a, const Vectorize<float>& b) {
  return _mm512_div_ps(a, b);
}

// frac. Implement this here so we can use subtraction
Vectorize<float> Vectorize<float>::frac() const {
  return *this - this->trunc();
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorize<float> inline maximum(const Vectorize<float>& a, const Vectorize<float>& b) {
  Vectorize<float> max = _mm512_max_ps(a, b);
  Vectorize<float> isnan = _mm512_cmp_ps_mask(a, b, _CMP_UNORD_Q);
  // Exploit the fact that all-ones is a NaN.
  return _mm512_or_ps(max, isnan);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorize<float> inline minimum(const Vectorize<float>& a, const Vectorize<float>& b) {
  Vectorize<float> min = _mm512_min_ps(a, b);
  Vectorize<float> isnan = _mm512_cmp_ps_mask(a, b, _CMP_UNORD_Q);
  // Exploit the fact that all-ones is a NaN.
  return _mm512_or_ps(min, isnan);
}

template <>
Vectorize<float> inline clamp(const Vectorize<float>& a, const Vectorize<float>& min, const Vectorize<float>& max) {
  return _mm512_min_ps(max, _mm512_max_ps(min, a));
}

template <>
Vectorize<float> inline clamp_max(const Vectorize<float>& a, const Vectorize<float>& max) {
  return _mm512_min_ps(max, a);
}

template <>
Vectorize<float> inline clamp_min(const Vectorize<float>& a, const Vectorize<float>& min) {
  return _mm512_max_ps(min, a);
}

template <>
Vectorize<float> inline operator&(const Vectorize<float>& a, const Vectorize<float>& b) {
  return _mm512_and_ps(a, b);
}

template <>
Vectorize<float> inline operator|(const Vectorize<float>& a, const Vectorize<float>& b) {
  return _mm512_or_ps(a, b);
}

template <>
Vectorize<float> inline operator^(const Vectorize<float>& a, const Vectorize<float>& b) {
  return _mm512_xor_ps(a, b);
}

Vectorize<float> Vectorize<float>::eq(const Vectorize<float>& other) const {
  return (*this == other) & Vectorize<float>(1.0f);
}

Vectorize<float> Vectorize<float>::ne(const Vectorize<float>& other) const {
  return (*this != other) & Vectorize<float>(1.0f);
}

Vectorize<float> Vectorize<float>::gt(const Vectorize<float>& other) const {
  return (*this > other) & Vectorize<float>(1.0f);
}

Vectorize<float> Vectorize<float>::ge(const Vectorize<float>& other) const {
  return (*this >= other) & Vectorize<float>(1.0f);
}

Vectorize<float> Vectorize<float>::lt(const Vectorize<float>& other) const {
  return (*this < other) & Vectorize<float>(1.0f);
}

Vectorize<float> Vectorize<float>::le(const Vectorize<float>& other) const {
  return (*this <= other) & Vectorize<float>(1.0f);
}

template <>
inline void convert(const float* src, float* dst, int64_t n) {
  int64_t i;
#pragma unroll
  for (i = 0; i <= (n - Vectorize<float>::size()); i += Vectorize<float>::size()) {
    _mm512_storeu_ps(dst + i, _mm512_loadu_ps(src + i));
  }
#pragma unroll
  for (; i < n; i++) {
    dst[i] = src[i];
  }
}

#ifdef CPU_CAPABILITY_AVX512
template <>
Vectorize<float> inline fmadd(const Vectorize<float>& a, const Vectorize<float>& b, const Vectorize<float>& c) {
  return _mm512_fmadd_ps(a, b, c);
}
#endif

#endif

}}}
