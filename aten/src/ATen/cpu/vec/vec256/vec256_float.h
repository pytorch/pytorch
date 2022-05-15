#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/util/irange.h>
#if defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)
#include <sleef.h>
#endif

namespace at {
namespace vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)

template <> class Vectorized<float> {
private:
  __m256 values;
public:
  using value_type = float;
  using size_type = int;
  static constexpr size_type size() {
    return 8;
  }
  Vectorized() {}
  Vectorized(__m256 v) : values(v) {}
  Vectorized(float val) {
    values = _mm256_set1_ps(val);
  }
  Vectorized(float val1, float val2, float val3, float val4,
         float val5, float val6, float val7, float val8) {
    values = _mm256_setr_ps(val1, val2, val3, val4, val5, val6, val7, val8);
  }
  operator __m256() const {
    return values;
  }
  template <int64_t mask>
  static Vectorized<float> blend(const Vectorized<float>& a, const Vectorized<float>& b) {
    return _mm256_blend_ps(a.values, b.values, mask);
  }
  static Vectorized<float> blendv(const Vectorized<float>& a, const Vectorized<float>& b,
                              const Vectorized<float>& mask) {
    return _mm256_blendv_ps(a.values, b.values, mask.values);
  }
  template<typename step_t>
  static Vectorized<float> arange(float base = 0.f, step_t step = static_cast<step_t>(1)) {
    return Vectorized<float>(
      base,            base +     step, base + 2 * step, base + 3 * step,
      base + 4 * step, base + 5 * step, base + 6 * step, base + 7 * step);
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
    }
    return b;
  }
  static Vectorized<float> loadu(const void* ptr, int64_t count = size()) {
    if (count == size())
      return _mm256_loadu_ps(reinterpret_cast<const float*>(ptr));
    __at_align__ float tmp_values[size()];
    // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
    // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
    // instructions while a loop would be compiled to one instruction.
    for (const auto i : c10::irange(size())) {
      tmp_values[i] = 0.0;
    }
    std::memcpy(
        tmp_values, reinterpret_cast<const float*>(ptr), count * sizeof(float));
    return _mm256_loadu_ps(tmp_values);
  }
  void store(void* ptr, int64_t count = size()) const {
    if (count == size()) {
      _mm256_storeu_ps(reinterpret_cast<float*>(ptr), values);
    } else if (count > 0) {
      float tmp_values[size()];
      _mm256_storeu_ps(reinterpret_cast<float*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(float));
    }
  }
  const float& operator[](int idx) const  = delete;
  float& operator[](int idx) = delete;
  int zero_mask() const {
    // returns an integer mask where all zero elements are translated to 1-bit and others are translated to 0-bit
    __m256 cmp = _mm256_cmp_ps(values, _mm256_set1_ps(0.0f), _CMP_EQ_OQ);
    return _mm256_movemask_ps(cmp);
  }
  Vectorized<float> isnan() const {
    return _mm256_cmp_ps(values, _mm256_set1_ps(0.0f), _CMP_UNORD_Q);
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
    auto mask = _mm256_set1_ps(-0.f);
    return _mm256_andnot_ps(mask, values);
  }
  Vectorized<float> angle() const {
    const auto zero_vec = _mm256_set1_ps(0.f);
    const auto nan_vec = _mm256_set1_ps(NAN);
    const auto not_nan_mask = _mm256_cmp_ps(values, values, _CMP_EQ_OQ);
    const auto nan_mask = _mm256_cmp_ps(not_nan_mask, zero_vec, _CMP_EQ_OQ);
    const auto pi = _mm256_set1_ps(c10::pi<float>);

    const auto neg_mask = _mm256_cmp_ps(values, zero_vec, _CMP_LT_OQ);
    auto angle = _mm256_blendv_ps(zero_vec, pi, neg_mask);
    angle = _mm256_blendv_ps(angle, nan_vec, nan_mask);
    return angle;
  }
  Vectorized<float> real() const {
    return *this;
  }
  Vectorized<float> imag() const {
    return _mm256_set1_ps(0);
  }
  Vectorized<float> conj() const {
    return *this;
  }
  Vectorized<float> acos() const {
    return Vectorized<float>(Sleef_acosf8_u10(values));
  }
  Vectorized<float> asin() const {
    return Vectorized<float>(Sleef_asinf8_u10(values));
  }
  Vectorized<float> atan() const {
    return Vectorized<float>(Sleef_atanf8_u10(values));
  }
  Vectorized<float> atan2(const Vectorized<float> &b) const {
    return Vectorized<float>(Sleef_atan2f8_u10(values, b));
  }
  Vectorized<float> copysign(const Vectorized<float> &sign) const {
    return Vectorized<float>(Sleef_copysignf8(values, sign));
  }
  Vectorized<float> erf() const {
    return Vectorized<float>(Sleef_erff8_u10(values));
  }
  Vectorized<float> erfc() const {
    return Vectorized<float>(Sleef_erfcf8_u15(values));
  }
  Vectorized<float> erfinv() const {
    return map(calc_erfinv);
  }
  Vectorized<float> exp() const {
    return Vectorized<float>(Sleef_expf8_u10(values));
  }
  Vectorized<float> expm1() const {
    return Vectorized<float>(Sleef_expm1f8_u10(values));
  }
  Vectorized<float> fmod(const Vectorized<float>& q) const {
    return Vectorized<float>(Sleef_fmodf8(values, q));
  }
  Vectorized<float> log() const {
    return Vectorized<float>(Sleef_logf8_u10(values));
  }
  Vectorized<float> log2() const {
    return Vectorized<float>(Sleef_log2f8_u10(values));
  }
  Vectorized<float> log10() const {
    return Vectorized<float>(Sleef_log10f8_u10(values));
  }
  Vectorized<float> log1p() const {
    return Vectorized<float>(Sleef_log1pf8_u10(values));
  }
  Vectorized<float> frac() const;
  Vectorized<float> sin() const {
    return Vectorized<float>(Sleef_sinf8_u10(values));
  }
  Vectorized<float> sinh() const {
    return Vectorized<float>(Sleef_sinhf8_u10(values));
  }
  Vectorized<float> cos() const {
    return Vectorized<float>(Sleef_cosf8_u10(values));
  }
  Vectorized<float> cosh() const {
    return Vectorized<float>(Sleef_coshf8_u10(values));
  }
  Vectorized<float> ceil() const {
    return _mm256_ceil_ps(values);
  }
  Vectorized<float> floor() const {
    return _mm256_floor_ps(values);
  }
  Vectorized<float> hypot(const Vectorized<float> &b) const {
    return Vectorized<float>(Sleef_hypotf8_u05(values, b));
  }
  Vectorized<float> i0() const {
    return map(calc_i0);
  }
  Vectorized<float> i0e() const {
    return map(calc_i0e);
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
    return _mm256_xor_ps(_mm256_set1_ps(-0.f), values);
  }
  Vectorized<float> nextafter(const Vectorized<float> &b) const {
    return Vectorized<float>(Sleef_nextafterf8(values, b));
  }
  Vectorized<float> round() const {
    return _mm256_round_ps(values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
  Vectorized<float> tan() const {
    return Vectorized<float>(Sleef_tanf8_u10(values));
  }
  Vectorized<float> tanh() const {
    return Vectorized<float>(Sleef_tanhf8_u10(values));
  }
  Vectorized<float> trunc() const {
    return _mm256_round_ps(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  }
  Vectorized<float> lgamma() const {
    return Vectorized<float>(Sleef_lgammaf8_u10(values));
  }
  Vectorized<float> sqrt() const {
    return _mm256_sqrt_ps(values);
  }
  Vectorized<float> reciprocal() const {
    return _mm256_div_ps(_mm256_set1_ps(1), values);
  }
  Vectorized<float> rsqrt() const {
    return _mm256_div_ps(_mm256_set1_ps(1), _mm256_sqrt_ps(values));
  }
  Vectorized<float> pow(const Vectorized<float> &b) const {
    return Vectorized<float>(Sleef_powf8_u10(values, b));
  }
  // Comparison using the _CMP_**_OQ predicate.
  //   `O`: get false if an operand is NaN
  //   `Q`: do not raise if an operand is NaN
  Vectorized<float> operator==(const Vectorized<float>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_EQ_OQ);
  }

  Vectorized<float> operator!=(const Vectorized<float>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_NEQ_UQ);
  }

  Vectorized<float> operator<(const Vectorized<float>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_LT_OQ);
  }

  Vectorized<float> operator<=(const Vectorized<float>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_LE_OQ);
  }

  Vectorized<float> operator>(const Vectorized<float>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_GT_OQ);
  }

  Vectorized<float> operator>=(const Vectorized<float>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_GE_OQ);
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
  return _mm256_add_ps(a, b);
}

template <>
Vectorized<float> inline operator-(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm256_sub_ps(a, b);
}

template <>
Vectorized<float> inline operator*(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm256_mul_ps(a, b);
}

template <>
Vectorized<float> inline operator/(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm256_div_ps(a, b);
}

// frac. Implement this here so we can use subtraction
inline Vectorized<float> Vectorized<float>::frac() const {
  return *this - this->trunc();
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<float> inline maximum(const Vectorized<float>& a, const Vectorized<float>& b) {
  Vectorized<float> max = _mm256_max_ps(a, b);
  Vectorized<float> isnan = _mm256_cmp_ps(a, b, _CMP_UNORD_Q);
  // Exploit the fact that all-ones is a NaN.
  return _mm256_or_ps(max, isnan);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<float> inline minimum(const Vectorized<float>& a, const Vectorized<float>& b) {
  Vectorized<float> min = _mm256_min_ps(a, b);
  Vectorized<float> isnan = _mm256_cmp_ps(a, b, _CMP_UNORD_Q);
  // Exploit the fact that all-ones is a NaN.
  return _mm256_or_ps(min, isnan);
}

template <>
Vectorized<float> inline clamp(const Vectorized<float>& a, const Vectorized<float>& min, const Vectorized<float>& max) {
  return _mm256_min_ps(max, _mm256_max_ps(min, a));
}

template <>
Vectorized<float> inline clamp_max(const Vectorized<float>& a, const Vectorized<float>& max) {
  return _mm256_min_ps(max, a);
}

template <>
Vectorized<float> inline clamp_min(const Vectorized<float>& a, const Vectorized<float>& min) {
  return _mm256_max_ps(min, a);
}

template <>
Vectorized<float> inline operator&(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm256_and_ps(a, b);
}

template <>
Vectorized<float> inline operator|(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm256_or_ps(a, b);
}

template <>
Vectorized<float> inline operator^(const Vectorized<float>& a, const Vectorized<float>& b) {
  return _mm256_xor_ps(a, b);
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
    _mm256_storeu_ps(dst + i, _mm256_loadu_ps(src + i));
  }
#pragma unroll
  for (; i < n; i++) {
    dst[i] = src[i];
  }
}


template <>
Vectorized<float> inline fmadd(const Vectorized<float>& a, const Vectorized<float>& b, const Vectorized<float>& c) {
  return _mm256_fmadd_ps(a, b, c);
}

#endif

}}}
