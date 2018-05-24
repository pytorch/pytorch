#pragma once

#include "intrinsics.h"
#include "vec256_base.h"
#if defined(__AVX__) && !defined(_MSC_VER)
#include <sleef.h>
#endif

namespace at {
namespace vec256 {
namespace {

#if defined(__AVX__) && !defined(_MSC_VER)

template <> class Vec256<double> {
private:
  __m256d values;
public:
  static constexpr int size = 4;
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
    return _mm256_blend_pd(a, b, mask);
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
  static Vec256<double> load(const void* ptr, int64_t count = size) {
    if (count == size)
      return _mm256_loadu_pd(reinterpret_cast<const double*>(ptr));

    __at_align32__ double tmp_values[size];
    std::memcpy(
        tmp_values,
        reinterpret_cast<const double*>(ptr),
        count * sizeof(double));
    return _mm256_loadu_pd(tmp_values);
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
  const double& operator[](int idx) const  = delete;
  double& operator[](int idx) = delete;
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

template <>
Vec256<double> inline min(const Vec256<double>& a, const Vec256<double>& b) {
  return _mm256_min_pd(a, b);
}

template <>
Vec256<double> inline map(double (*f)(double), Vec256<double> x) = delete;

template <>
Vec256<double> inline abs(Vec256<double> x) {
  auto mask = _mm256_set1_pd(-0.f);
  return _mm256_andnot_pd(mask, x);
}

template <>
Vec256<double> inline acos(Vec256<double> x) {
  return Vec256<double>(Sleef_acosd4_u10(x));
}

template <>
Vec256<double> inline asin(Vec256<double> x) {
  return Vec256<double>(Sleef_asind4_u10(x));
}

template <>
Vec256<double> inline atan(Vec256<double> x) {
  return Vec256<double>(Sleef_atand4_u10(x));
}

template <>
Vec256<double> inline erf(Vec256<double> x) {
  return Vec256<double>(Sleef_erfd4_u10(x));
}

template <>
Vec256<double> inline exp(Vec256<double> x) {
  return Vec256<double>(Sleef_expd4_u10(x));
}

template <>
Vec256<double> inline expm1(Vec256<double> x) {
  return Vec256<double>(Sleef_expm1d4_u10(x));
}

template <>
Vec256<double> inline log(Vec256<double> x) {
  return Vec256<double>(Sleef_logd4_u10(x));
}

template <>
Vec256<double> inline log2(Vec256<double> x) {
  return Vec256<double>(Sleef_log2d4_u10(x));
}

template <>
Vec256<double> inline log10(Vec256<double> x) {
  return Vec256<double>(Sleef_log10d4_u10(x));
}

template <>
Vec256<double> inline log1p(Vec256<double> x) {
  return Vec256<double>(Sleef_log1pd4_u10(x));
}

template <>
Vec256<double> inline sin(Vec256<double> x) = delete;

template <>
Vec256<double> inline cos(Vec256<double> x) = delete;

template <>
Vec256<double> inline ceil(Vec256<double> x) {
  return _mm256_ceil_pd(x);
}

template <>
Vec256<double> inline floor(Vec256<double> x) {
  return _mm256_floor_pd(x);
}

template <>
Vec256<double> inline round(Vec256<double> x) {
  return _mm256_round_pd(
      x, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

template <>
Vec256<double> inline tanh(Vec256<double> x) {
  return Vec256<double>(Sleef_tanhd4_u10(x));
}

template <>
Vec256<double> inline trunc(Vec256<double> x) {
  return _mm256_round_pd(x, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
}

template <>
Vec256<double> inline sqrt(Vec256<double> x) {
  return _mm256_sqrt_pd(x);
}

template <>
Vec256<double> inline reciprocal(Vec256<double> x) {
  return _mm256_div_pd(_mm256_set1_pd(1), x);
}

template <>
Vec256<double> inline rsqrt(Vec256<double> x) {
  return reciprocal(sqrt(x));
}

template <>
Vec256<double> inline sigmoid(Vec256<double> x) {
  return _mm256_div_pd(
      _mm256_set1_pd(1),
      _mm256_add_pd(
          _mm256_set1_pd(1),
          exp(Vec256<double>(_mm256_sub_pd(_mm256_set1_pd(0), x)))));
}
#endif
}
} // namespace vec256
} // namespace at
