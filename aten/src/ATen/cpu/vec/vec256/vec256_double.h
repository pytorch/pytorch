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

template <> class Vectorized<double> {
private:
  __m256d values;
public:
  using value_type = double;
  using size_type = int;
  static constexpr size_type size() {
    return 4;
  }
  Vectorized() {}
  Vectorized(__m256d v) : values(v) {}
  Vectorized(double val) {
    values = _mm256_set1_pd(val);
  }
  Vectorized(double val1, double val2, double val3, double val4) {
    values = _mm256_setr_pd(val1, val2, val3, val4);
  }
  operator __m256d() const {
    return values;
  }
  template <int64_t mask>
  static Vectorized<double> blend(const Vectorized<double>& a, const Vectorized<double>& b) {
    return _mm256_blend_pd(a.values, b.values, mask);
  }
  static Vectorized<double> blendv(const Vectorized<double>& a, const Vectorized<double>& b,
                               const Vectorized<double>& mask) {
    return _mm256_blendv_pd(a.values, b.values, mask.values);
  }
  template<typename step_t>
  static Vectorized<double> arange(double base = 0., step_t step = static_cast<step_t>(1)) {
    return Vectorized<double>(base, base + step, base + 2 * step, base + 3 * step);
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
    }
    return b;
  }
  static Vectorized<double> loadu(const void* ptr, int64_t count = size()) {
    if (count == size())
      return _mm256_loadu_pd(reinterpret_cast<const double*>(ptr));


    __at_align__ double tmp_values[size()];
    // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
    // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
    // instructions while a loop would be compiled to one instruction.
    for (const auto i : c10::irange(size())) {
      tmp_values[i] = 0.0;
    }
    std::memcpy(
        tmp_values,
        reinterpret_cast<const double*>(ptr),
        count * sizeof(double));
    return _mm256_load_pd(tmp_values);
  }
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      _mm256_storeu_pd(reinterpret_cast<double*>(ptr), values);
    } else if (count > 0) {
      double tmp_values[size()];
      _mm256_storeu_pd(reinterpret_cast<double*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(double));
    }
  }
  const double& operator[](int idx) const  = delete;
  double& operator[](int idx) = delete;
  int zero_mask() const {
    // returns an integer mask where all zero elements are translated to 1-bit and others are translated to 0-bit
    __m256d cmp = _mm256_cmp_pd(values, _mm256_set1_pd(0.0), _CMP_EQ_OQ);
    return _mm256_movemask_pd(cmp);
  }
  Vectorized<double> isnan() const {
    return _mm256_cmp_pd(values, _mm256_set1_pd(0.0), _CMP_UNORD_Q);
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
    auto mask = _mm256_set1_pd(-0.f);
    return _mm256_andnot_pd(mask, values);
  }
  Vectorized<double> angle() const {
    const auto zero_vec = _mm256_set1_pd(0.f);
    const auto nan_vec = _mm256_set1_pd(NAN);
    const auto not_nan_mask = _mm256_cmp_pd(values, values, _CMP_EQ_OQ);
    const auto nan_mask = _mm256_cmp_pd(not_nan_mask, zero_vec, _CMP_EQ_OQ);
    const auto pi = _mm256_set1_pd(c10::pi<double>);

    const auto neg_mask = _mm256_cmp_pd(values, zero_vec, _CMP_LT_OQ);
    auto angle = _mm256_blendv_pd(zero_vec, pi, neg_mask);
    angle = _mm256_blendv_pd(angle, nan_vec, nan_mask);
    return angle;
  }
  Vectorized<double> real() const {
    return *this;
  }
  Vectorized<double> imag() const {
    return _mm256_set1_pd(0);
  }
  Vectorized<double> conj() const {
    return *this;
  }
  Vectorized<double> acos() const {
    return Vectorized<double>(Sleef_acosd4_u10(values));
  }
  Vectorized<double> asin() const {
    return Vectorized<double>(Sleef_asind4_u10(values));
  }
  Vectorized<double> atan() const {
    return Vectorized<double>(Sleef_atand4_u10(values));
  }
  Vectorized<double> atan2(const Vectorized<double> &b) const {
    return Vectorized<double>(Sleef_atan2d4_u10(values, b));
  }
  Vectorized<double> copysign(const Vectorized<double> &sign) const {
    return Vectorized<double>(Sleef_copysignd4(values, sign));
  }
  Vectorized<double> erf() const {
    return Vectorized<double>(Sleef_erfd4_u10(values));
  }
  Vectorized<double> erfc() const {
    return Vectorized<double>(Sleef_erfcd4_u15(values));
  }
  Vectorized<double> erfinv() const {
    return map(calc_erfinv);
  }
  Vectorized<double> exp() const {
    return Vectorized<double>(Sleef_expd4_u10(values));
  }
  Vectorized<double> expm1() const {
    return Vectorized<double>(Sleef_expm1d4_u10(values));
  }
  Vectorized<double> fmod(const Vectorized<double>& q) const {
    return Vectorized<double>(Sleef_fmodd4(values, q));
  }
  Vectorized<double> hypot(const Vectorized<double> &b) const {
    return Vectorized<double>(Sleef_hypotd4_u05(values, b));
  }
  Vectorized<double> i0() const {
    return map(calc_i0);
  }
  Vectorized<double> i0e() const {
    return map(calc_i0e);
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
    return Vectorized<double>(Sleef_logd4_u10(values));
  }
  Vectorized<double> log2() const {
    return Vectorized<double>(Sleef_log2d4_u10(values));
  }
  Vectorized<double> log10() const {
    return Vectorized<double>(Sleef_log10d4_u10(values));
  }
  Vectorized<double> log1p() const {
    return Vectorized<double>(Sleef_log1pd4_u10(values));
  }
  Vectorized<double> sin() const {
    return Vectorized<double>(Sleef_sind4_u10(values));
  }
  Vectorized<double> sinh() const {
    return Vectorized<double>(Sleef_sinhd4_u10(values));
  }
  Vectorized<double> cos() const {
    return Vectorized<double>(Sleef_cosd4_u10(values));
  }
  Vectorized<double> cosh() const {
    return Vectorized<double>(Sleef_coshd4_u10(values));
  }
  Vectorized<double> ceil() const {
    return _mm256_ceil_pd(values);
  }
  Vectorized<double> floor() const {
    return _mm256_floor_pd(values);
  }
  Vectorized<double> frac() const;
  Vectorized<double> neg() const {
    return _mm256_xor_pd(_mm256_set1_pd(-0.), values);
  }
  Vectorized<double> nextafter(const Vectorized<double> &b) const {
    return Vectorized<double>(Sleef_nextafterd4(values, b));
  }
  Vectorized<double> round() const {
    return _mm256_round_pd(values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
  Vectorized<double> tan() const {
    return Vectorized<double>(Sleef_tand4_u10(values));
  }
  Vectorized<double> tanh() const {
    return Vectorized<double>(Sleef_tanhd4_u10(values));
  }
  Vectorized<double> trunc() const {
    return _mm256_round_pd(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  }
  Vectorized<double> lgamma() const {
    return Vectorized<double>(Sleef_lgammad4_u10(values));
  }
  Vectorized<double> sqrt() const {
    return _mm256_sqrt_pd(values);
  }
  Vectorized<double> reciprocal() const {
    return _mm256_div_pd(_mm256_set1_pd(1), values);
  }
  Vectorized<double> rsqrt() const {
    return _mm256_div_pd(_mm256_set1_pd(1), _mm256_sqrt_pd(values));
  }
  Vectorized<double> pow(const Vectorized<double> &b) const {
    return Vectorized<double>(Sleef_powd4_u10(values, b));
  }
  // Comparison using the _CMP_**_OQ predicate.
  //   `O`: get false if an operand is NaN
  //   `Q`: do not raise if an operand is NaN
  Vectorized<double> operator==(const Vectorized<double>& other) const {
    return _mm256_cmp_pd(values, other.values, _CMP_EQ_OQ);
  }

  Vectorized<double> operator!=(const Vectorized<double>& other) const {
    return _mm256_cmp_pd(values, other.values, _CMP_NEQ_UQ);
  }

  Vectorized<double> operator<(const Vectorized<double>& other) const {
    return _mm256_cmp_pd(values, other.values, _CMP_LT_OQ);
  }

  Vectorized<double> operator<=(const Vectorized<double>& other) const {
    return _mm256_cmp_pd(values, other.values, _CMP_LE_OQ);
  }

  Vectorized<double> operator>(const Vectorized<double>& other) const {
    return _mm256_cmp_pd(values, other.values, _CMP_GT_OQ);
  }

  Vectorized<double> operator>=(const Vectorized<double>& other) const {
    return _mm256_cmp_pd(values, other.values, _CMP_GE_OQ);
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
  return _mm256_add_pd(a, b);
}

template <>
Vectorized<double> inline operator-(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm256_sub_pd(a, b);
}

template <>
Vectorized<double> inline operator*(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm256_mul_pd(a, b);
}

template <>
Vectorized<double> inline operator/(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm256_div_pd(a, b);
}

// frac. Implement this here so we can use subtraction.
inline Vectorized<double> Vectorized<double>::frac() const {
  return *this - this->trunc();
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<double> inline maximum(const Vectorized<double>& a, const Vectorized<double>& b) {
  Vectorized<double> max = _mm256_max_pd(a, b);
  Vectorized<double> isnan = _mm256_cmp_pd(a, b, _CMP_UNORD_Q);
  // Exploit the fact that all-ones is a NaN.
  return _mm256_or_pd(max, isnan);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<double> inline minimum(const Vectorized<double>& a, const Vectorized<double>& b) {
  Vectorized<double> min = _mm256_min_pd(a, b);
  Vectorized<double> isnan = _mm256_cmp_pd(a, b, _CMP_UNORD_Q);
  // Exploit the fact that all-ones is a NaN.
  return _mm256_or_pd(min, isnan);
}

template <>
Vectorized<double> inline clamp(const Vectorized<double>& a, const Vectorized<double>& min, const Vectorized<double>& max) {
  return _mm256_min_pd(max, _mm256_max_pd(min, a));
}

template <>
Vectorized<double> inline clamp_min(const Vectorized<double>& a, const Vectorized<double>& min) {
  return _mm256_max_pd(min, a);
}

template <>
Vectorized<double> inline clamp_max(const Vectorized<double>& a, const Vectorized<double>& max) {
  return _mm256_min_pd(max, a);
}

template <>
Vectorized<double> inline operator&(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm256_and_pd(a, b);
}

template <>
Vectorized<double> inline operator|(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm256_or_pd(a, b);
}

template <>
Vectorized<double> inline operator^(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm256_xor_pd(a, b);
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
#pragma unroll
  for (i = 0; i <= (n - Vectorized<double>::size()); i += Vectorized<double>::size()) {
    _mm256_storeu_pd(dst + i, _mm256_loadu_pd(src + i));
  }
#pragma unroll
  for (; i < n; i++) {
    dst[i] = src[i];
  }
}

#ifdef CPU_CAPABILITY_AVX2
template <>
Vectorized<double> inline fmadd(const Vectorized<double>& a, const Vectorized<double>& b, const Vectorized<double>& c) {
  return _mm256_fmadd_pd(a, b, c);
}

template <>
Vectorized<double> inline fmsub(const Vectorized<double>& a, const Vectorized<double>& b, const Vectorized<double>& c) {
  return _mm256_fmsub_pd(a, b, c);
}
#endif

#endif

}}}
