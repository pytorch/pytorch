#pragma once

#include "intrinsics.h"
#include "vec256_base.h"
#include <sleef.h>
#include <iostream>

namespace at {
namespace vec256 {
namespace {

#ifdef __AVX__

template <> class Vec256<double> {
public:
  static constexpr int size = 4;
  __m256d values;
  Vec256() {}
  Vec256(__m256d v) : values(v) {}
  Vec256(double val) {
    values = _mm256_set1_pd(val);
  }
  operator __m256d() const {
    return values;
  }
  template <int64_t mask>
  static Vec256<double> blend(Vec256<double> a, Vec256<double> b) {
    return _mm256_blend_pd(a.values, b.values, mask);
  }
  static Vec256<double> set(Vec256<double> a, Vec256<double> b, int64_t count = size) {
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
  static Vec256<double> loadu(const void* ptr, int64_t count = size) {
    if (count == size)
      return _mm256_loadu_pd(reinterpret_cast<const double*>(ptr));

    __at_align32__ double tmp_values[size];
    std::memcpy(
        tmp_values,
        reinterpret_cast<const double*>(ptr),
        count * sizeof(double));
    return _mm256_load_pd(tmp_values);
  }
  void store(void* ptr, int count = size) const {
    if (count == size) {
      _mm256_storeu_pd(reinterpret_cast<double*>(ptr), values);
    } else {
      double tmp_values[size];
      _mm256_storeu_pd(reinterpret_cast<double*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(double));
    }
  }
  Vec256<double> map(double (*f)(double)) const {
    __at_align32__ double tmp[4];
    store(tmp);
    for (int64_t i = 0; i < 4; i++) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  Vec256<double> abs() const {
    auto mask = _mm256_set1_pd(-0.f);
    return _mm256_andnot_pd(mask, values);
  }
  Vec256<double> acos() const {
    return Vec256<double>(Sleef_acosd4_u10(values));
  }
  Vec256<double> asin() const {
    return Vec256<double>(Sleef_asind4_u10(values));
  }
  Vec256<double> atan() const {
    return Vec256<double>(Sleef_atand4_u10(values));
  }
  Vec256<double> erf() const {
    return Vec256<double>(Sleef_erfd4_u10(values));
  }
  Vec256<double> exp() const {
    return Vec256<double>(Sleef_expd4_u10(values));
  }
  Vec256<double> expm1() const {
    return Vec256<double>(Sleef_expm1d4_u10(values));
  }
  Vec256<double> log() const {
    return Vec256<double>(Sleef_logd4_u10(values));
  }
  Vec256<double> log2() const {
    return Vec256<double>(Sleef_log2d4_u10(values));
  }
  Vec256<double> log10() const {
    return Vec256<double>(Sleef_log10d4_u10(values));
  }
  Vec256<double> log1p() const {
    return Vec256<double>(Sleef_log1pd4_u10(values));
  }
  Vec256<double> sin() const {
    return map(std::sin);
  }
  Vec256<double> cos() const {
    return map(std::cos);
  }
  Vec256<double> ceil() const {
    return _mm256_ceil_pd(values);
  }
  Vec256<double> floor() const {
    return _mm256_floor_pd(values);
  }
  Vec256<double> round() const {
    return _mm256_round_pd(values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
  Vec256<double> tanh() const {
    return Vec256<double>(Sleef_tanhd4_u10(values));
  }
  Vec256<double> trunc() const {
    return _mm256_round_pd(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  }
  Vec256<double> sqrt() const {
    return _mm256_sqrt_pd(values);
  }
};

template <>
Vec256<double> inline operator+(const Vec256<double>& a, const Vec256<double>& b) {
  return _mm256_add_pd(a, b);
}

template <>
Vec256<double> inline operator-(const Vec256<double>& a, const Vec256<double>& b) {
  return _mm256_sub_pd(a, b);
}

template <>
Vec256<double> inline operator*(const Vec256<double>& a, const Vec256<double>& b) {
  return _mm256_mul_pd(a, b);
}

template <>
Vec256<double> inline operator/(const Vec256<double>& a, const Vec256<double>& b) {
  return _mm256_div_pd(a, b);
}

template <>
Vec256<double> inline max(const Vec256<double>& a, const Vec256<double>& b) {
  return _mm256_max_pd(a, b);
}

#endif

}}}
