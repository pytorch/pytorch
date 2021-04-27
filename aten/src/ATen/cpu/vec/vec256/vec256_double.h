#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/vec256/intrinsics.h>
#include <ATen/cpu/vec/vec256/vec256_base.h>
#if defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)
#include <sleef.h>
#endif

namespace at {
namespace vec {
// See Note [Acceptable use of anonymous namespace in header]
namespace {

#if defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)

template <> class Vectorize<double> {
private:
  __m256d values;
public:
  using value_type = double;
  using size_type = int;
  static constexpr size_type size() {
    return 4;
  }
  Vectorize() {}
  Vectorize(__m256d v) : values(v) {}
  Vectorize(double val) {
    values = _mm256_set1_pd(val);
  }
  Vectorize(double val1, double val2, double val3, double val4) {
    values = _mm256_setr_pd(val1, val2, val3, val4);
  }
  operator __m256d() const {
    return values;
  }
  template <int64_t mask>
  static Vectorize<double> blend(const Vectorize<double>& a, const Vectorize<double>& b) {
    return _mm256_blend_pd(a.values, b.values, mask);
  }
  static Vectorize<double> blendv(const Vectorize<double>& a, const Vectorize<double>& b,
                               const Vectorize<double>& mask) {
    return _mm256_blendv_pd(a.values, b.values, mask.values);
  }
  template<typename step_t>
  static Vectorize<double> arange(double base = 0., step_t step = static_cast<step_t>(1)) {
    return Vectorize<double>(base, base + step, base + 2 * step, base + 3 * step);
  }
  static Vectorize<double> set(const Vectorize<double>& a, const Vectorize<double>& b,
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
  static Vectorize<double> loadu(const void* ptr, int64_t count = size()) {
    if (count == size())
      return _mm256_loadu_pd(reinterpret_cast<const double*>(ptr));


    __at_align32__ double tmp_values[size()];
    // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
    // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
    // instructions while a loop would be compiled to one instruction.
    for (auto i = 0; i < size(); ++i) {
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
  Vectorize<double> isnan() const {
    return _mm256_cmp_pd(values, _mm256_set1_pd(0.0), _CMP_UNORD_Q);
  }
  Vectorize<double> map(double (*f)(double)) const {
    __at_align32__ double tmp[size()];
    store(tmp);
    for (int64_t i = 0; i < size(); i++) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  Vectorize<double> abs() const {
    auto mask = _mm256_set1_pd(-0.f);
    return _mm256_andnot_pd(mask, values);
  }
  Vectorize<double> angle() const {
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
  Vectorize<double> real() const {
    return *this;
  }
  Vectorize<double> imag() const {
    return _mm256_set1_pd(0);
  }
  Vectorize<double> conj() const {
    return *this;
  }
  Vectorize<double> acos() const {
    return Vectorize<double>(Sleef_acosd4_u10(values));
  }
  Vectorize<double> asin() const {
    return Vectorize<double>(Sleef_asind4_u10(values));
  }
  Vectorize<double> atan() const {
    return Vectorize<double>(Sleef_atand4_u10(values));
  }
  Vectorize<double> atan2(const Vectorize<double> &b) const {
    return Vectorize<double>(Sleef_atan2d4_u10(values, b));
  }
  Vectorize<double> copysign(const Vectorize<double> &sign) const {
    return Vectorize<double>(Sleef_copysignd4(values, sign));
  }
  Vectorize<double> erf() const {
    return Vectorize<double>(Sleef_erfd4_u10(values));
  }
  Vectorize<double> erfc() const {
    return Vectorize<double>(Sleef_erfcd4_u15(values));
  }
  Vectorize<double> erfinv() const {
    return map(calc_erfinv);
  }
  Vectorize<double> exp() const {
    return Vectorize<double>(Sleef_expd4_u10(values));
  }
  Vectorize<double> expm1() const {
    return Vectorize<double>(Sleef_expm1d4_u10(values));
  }
  Vectorize<double> fmod(const Vectorize<double>& q) const {
    return Vectorize<double>(Sleef_fmodd4(values, q));
  }
  Vectorize<double> hypot(const Vectorize<double> &b) const {
    return Vectorize<double>(Sleef_hypotd4_u05(values, b));
  }
  Vectorize<double> i0() const {
    return map(calc_i0);
  }
  Vectorize<double> i0e() const {
    return map(calc_i0e);
  }
  Vectorize<double> igamma(const Vectorize<double> &x) const {
    __at_align32__ double tmp[size()];
    __at_align32__ double tmp_x[size()];
    store(tmp);
    x.store(tmp_x);
    for (int64_t i = 0; i < size(); i++) {
      tmp[i] = calc_igamma(tmp[i], tmp_x[i]);
    }
    return loadu(tmp);
  }
  Vectorize<double> igammac(const Vectorize<double> &x) const {
    __at_align32__ double tmp[size()];
    __at_align32__ double tmp_x[size()];
    store(tmp);
    x.store(tmp_x);
    for (int64_t i = 0; i < size(); i++) {
      tmp[i] = calc_igammac(tmp[i], tmp_x[i]);
    }
    return loadu(tmp);
  }
  Vectorize<double> log() const {
    return Vectorize<double>(Sleef_logd4_u10(values));
  }
  Vectorize<double> log2() const {
    return Vectorize<double>(Sleef_log2d4_u10(values));
  }
  Vectorize<double> log10() const {
    return Vectorize<double>(Sleef_log10d4_u10(values));
  }
  Vectorize<double> log1p() const {
    return Vectorize<double>(Sleef_log1pd4_u10(values));
  }
  Vectorize<double> sin() const {
    return Vectorize<double>(Sleef_sind4_u10(values));
  }
  Vectorize<double> sinh() const {
    return Vectorize<double>(Sleef_sinhd4_u10(values));
  }
  Vectorize<double> cos() const {
    return Vectorize<double>(Sleef_cosd4_u10(values));
  }
  Vectorize<double> cosh() const {
    return Vectorize<double>(Sleef_coshd4_u10(values));
  }
  Vectorize<double> ceil() const {
    return _mm256_ceil_pd(values);
  }
  Vectorize<double> floor() const {
    return _mm256_floor_pd(values);
  }
  Vectorize<double> frac() const;
  Vectorize<double> neg() const {
    return _mm256_xor_pd(_mm256_set1_pd(-0.), values);
  }
  Vectorize<double> nextafter(const Vectorize<double> &b) const {
    return Vectorize<double>(Sleef_nextafterd4(values, b));
  }
  Vectorize<double> round() const {
    return _mm256_round_pd(values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
  Vectorize<double> tan() const {
    return Vectorize<double>(Sleef_tand4_u10(values));
  }
  Vectorize<double> tanh() const {
    return Vectorize<double>(Sleef_tanhd4_u10(values));
  }
  Vectorize<double> trunc() const {
    return _mm256_round_pd(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  }
  Vectorize<double> lgamma() const {
    return Vectorize<double>(Sleef_lgammad4_u10(values));
  }
  Vectorize<double> sqrt() const {
    return _mm256_sqrt_pd(values);
  }
  Vectorize<double> reciprocal() const {
    return _mm256_div_pd(_mm256_set1_pd(1), values);
  }
  Vectorize<double> rsqrt() const {
    return _mm256_div_pd(_mm256_set1_pd(1), _mm256_sqrt_pd(values));
  }
  Vectorize<double> pow(const Vectorize<double> &b) const {
    return Vectorize<double>(Sleef_powd4_u10(values, b));
  }
  // Comparison using the _CMP_**_OQ predicate.
  //   `O`: get false if an operand is NaN
  //   `Q`: do not raise if an operand is NaN
  Vectorize<double> operator==(const Vectorize<double>& other) const {
    return _mm256_cmp_pd(values, other.values, _CMP_EQ_OQ);
  }

  Vectorize<double> operator!=(const Vectorize<double>& other) const {
    return _mm256_cmp_pd(values, other.values, _CMP_NEQ_UQ);
  }

  Vectorize<double> operator<(const Vectorize<double>& other) const {
    return _mm256_cmp_pd(values, other.values, _CMP_LT_OQ);
  }

  Vectorize<double> operator<=(const Vectorize<double>& other) const {
    return _mm256_cmp_pd(values, other.values, _CMP_LE_OQ);
  }

  Vectorize<double> operator>(const Vectorize<double>& other) const {
    return _mm256_cmp_pd(values, other.values, _CMP_GT_OQ);
  }

  Vectorize<double> operator>=(const Vectorize<double>& other) const {
    return _mm256_cmp_pd(values, other.values, _CMP_GE_OQ);
  }

  Vectorize<double> eq(const Vectorize<double>& other) const;
  Vectorize<double> ne(const Vectorize<double>& other) const;
  Vectorize<double> lt(const Vectorize<double>& other) const;
  Vectorize<double> le(const Vectorize<double>& other) const;
  Vectorize<double> gt(const Vectorize<double>& other) const;
  Vectorize<double> ge(const Vectorize<double>& other) const;
};

template <>
Vectorize<double> inline operator+(const Vectorize<double>& a, const Vectorize<double>& b) {
  return _mm256_add_pd(a, b);
}

template <>
Vectorize<double> inline operator-(const Vectorize<double>& a, const Vectorize<double>& b) {
  return _mm256_sub_pd(a, b);
}

template <>
Vectorize<double> inline operator*(const Vectorize<double>& a, const Vectorize<double>& b) {
  return _mm256_mul_pd(a, b);
}

template <>
Vectorize<double> inline operator/(const Vectorize<double>& a, const Vectorize<double>& b) {
  return _mm256_div_pd(a, b);
}

// frac. Implement this here so we can use subtraction.
Vectorize<double> Vectorize<double>::frac() const {
  return *this - this->trunc();
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorize<double> inline maximum(const Vectorize<double>& a, const Vectorize<double>& b) {
  Vectorize<double> max = _mm256_max_pd(a, b);
  Vectorize<double> isnan = _mm256_cmp_pd(a, b, _CMP_UNORD_Q);
  // Exploit the fact that all-ones is a NaN.
  return _mm256_or_pd(max, isnan);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorize<double> inline minimum(const Vectorize<double>& a, const Vectorize<double>& b) {
  Vectorize<double> min = _mm256_min_pd(a, b);
  Vectorize<double> isnan = _mm256_cmp_pd(a, b, _CMP_UNORD_Q);
  // Exploit the fact that all-ones is a NaN.
  return _mm256_or_pd(min, isnan);
}

template <>
Vectorize<double> inline clamp(const Vectorize<double>& a, const Vectorize<double>& min, const Vectorize<double>& max) {
  return _mm256_min_pd(max, _mm256_max_pd(min, a));
}

template <>
Vectorize<double> inline clamp_min(const Vectorize<double>& a, const Vectorize<double>& min) {
  return _mm256_max_pd(min, a);
}

template <>
Vectorize<double> inline clamp_max(const Vectorize<double>& a, const Vectorize<double>& max) {
  return _mm256_min_pd(max, a);
}

template <>
Vectorize<double> inline operator&(const Vectorize<double>& a, const Vectorize<double>& b) {
  return _mm256_and_pd(a, b);
}

template <>
Vectorize<double> inline operator|(const Vectorize<double>& a, const Vectorize<double>& b) {
  return _mm256_or_pd(a, b);
}

template <>
Vectorize<double> inline operator^(const Vectorize<double>& a, const Vectorize<double>& b) {
  return _mm256_xor_pd(a, b);
}

Vectorize<double> Vectorize<double>::eq(const Vectorize<double>& other) const {
  return (*this == other) & Vectorize<double>(1.0);
}

Vectorize<double> Vectorize<double>::ne(const Vectorize<double>& other) const {
  return (*this != other) & Vectorize<double>(1.0);
}

Vectorize<double> Vectorize<double>::gt(const Vectorize<double>& other) const {
  return (*this > other) & Vectorize<double>(1.0);
}

Vectorize<double> Vectorize<double>::ge(const Vectorize<double>& other) const {
  return (*this >= other) & Vectorize<double>(1.0);
}

Vectorize<double> Vectorize<double>::lt(const Vectorize<double>& other) const {
  return (*this < other) & Vectorize<double>(1.0);
}

Vectorize<double> Vectorize<double>::le(const Vectorize<double>& other) const {
  return (*this <= other) & Vectorize<double>(1.0);
}

template <>
inline void convert(const double* src, double* dst, int64_t n) {
  int64_t i;
#pragma unroll
  for (i = 0; i <= (n - Vectorize<double>::size()); i += Vectorize<double>::size()) {
    _mm256_storeu_pd(dst + i, _mm256_loadu_pd(src + i));
  }
#pragma unroll
  for (; i < n; i++) {
    dst[i] = src[i];
  }
}

#ifdef CPU_CAPABILITY_AVX2
template <>
Vectorize<double> inline fmadd(const Vectorize<double>& a, const Vectorize<double>& b, const Vectorize<double>& c) {
  return _mm256_fmadd_pd(a, b, c);
}
#endif

#endif

}}}
