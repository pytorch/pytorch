#pragma once

#include "intrinsics.h"
#include "vec256_base.h"
#include <sleef.h>

namespace at {
namespace vec256 {
namespace {

#ifdef __AVX__

template <> class Vec256<float> {
public:
  static constexpr int64_t size = 8;
  __m256 values;
  Vec256() {}
  Vec256(__m256 v) : values(v) {}
  Vec256(float val) {
    values = _mm256_set1_ps(val);
  }
  operator __m256() const {
    return values;
  }
  template <int64_t mask>
  static Vec256<float> blend(Vec256<float> a, Vec256<float> b) {
    return _mm256_blend_ps(a.values, b.values, mask);
  }
  static Vec256<float> set(Vec256<float> a, Vec256<float> b, int64_t count = size) {
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
  static Vec256<float> loadu(const void* ptr, int64_t count = size) {
    if (count == size)
      return _mm256_loadu_ps(reinterpret_cast<const float*>(ptr));
    __at_align32__ float tmp_values[size];
    std::memcpy(
        tmp_values, reinterpret_cast<const float*>(ptr), count * sizeof(float));
    return _mm256_loadu_ps(tmp_values);
  }
  void store(void* ptr, int64_t count = size) const {
    if (count == size) {
      _mm256_storeu_ps(reinterpret_cast<float*>(ptr), values);
    } else {
      float tmp_values[size];
      _mm256_storeu_ps(reinterpret_cast<float*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(float));
    }
  }
  Vec256<float> map(float (*f)(float)) const {
    __at_align32__ float tmp[8];
    store(tmp);
    for (int64_t i = 0; i < 8; i++) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  Vec256<float> abs() const {
    auto mask = _mm256_set1_ps(-0.f);
    return _mm256_andnot_ps(mask, values);
  }
  Vec256<float> acos() const {
    return Vec256<float>(Sleef_acosf8_u10(values));
  }
  Vec256<float> asin() const {
    return Vec256<float>(Sleef_asinf8_u10(values));
  }
  Vec256<float> atan() const {
    return Vec256<float>(Sleef_atanf8_u10(values));
  }
  Vec256<float> erf() const {
    return Vec256<float>(Sleef_erff8_u10(values));
  }
  Vec256<float> exp() const {
    return Vec256<float>(Sleef_expf8_u10(values));
  }
  Vec256<float> expm1() const {
    return Vec256<float>(Sleef_expm1f8_u10(values));
  }
  Vec256<float> log() const {
    return Vec256<float>(Sleef_logf8_u10(values));
  }
  Vec256<float> log2() const {
    return Vec256<float>(Sleef_log2f8_u10(values));
  }
  Vec256<float> log10() const {
    return Vec256<float>(Sleef_log10f8_u10(values));
  }
  Vec256<float> log1p() const {
    return Vec256<float>(Sleef_log1pf8_u10(values));
  }
  Vec256<float> sin() const {
    return map(std::sin);
  }
  Vec256<float> cos() const {
    return map(std::cos);
  }
  Vec256<float> ceil() const {
    return _mm256_ceil_ps(values);
  }
  Vec256<float> floor() const {
    return _mm256_floor_ps(values);
  }
  Vec256<float> round() const {
    return _mm256_round_ps(values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
  Vec256<float> tanh() const {
    return Vec256<float>(Sleef_tanhf8_u10(values));
  }
  Vec256<float> trunc() const {
    return _mm256_round_ps(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  }
  Vec256<float> sqrt() const {
    return _mm256_sqrt_ps(values);
  }
};

template <>
Vec256<float> inline operator+(const Vec256<float>& a, const Vec256<float>& b) {
  return _mm256_add_ps(a, b);
}

template <>
Vec256<float> inline operator-(const Vec256<float>& a, const Vec256<float>& b) {
  return _mm256_sub_ps(a, b);
}

template <>
Vec256<float> inline operator*(const Vec256<float>& a, const Vec256<float>& b) {
  return _mm256_mul_ps(a, b);
}

template <>
Vec256<float> inline operator/(const Vec256<float>& a, const Vec256<float>& b) {
  return _mm256_div_ps(a, b);
}

template <>
Vec256<float> inline max(const Vec256<float>& a, const Vec256<float>& b) {
  return _mm256_max_ps(a, b);
}

#endif

}}}
