#pragma once

#include "intrinsics.h"
#include "vec256_base.h"
#include <sleef.h>
#include <iostream>

namespace at {
namespace vec256 {
namespace {

#ifdef __AVX__

template <> class Vec256<float> {
public:
  static constexpr int size = 8;
  __m256 values;
  Vec256() {}
  Vec256(__m256 v) : values(v) {}
  Vec256(float val) {
    values = _mm256_set1_ps(val);
  }
  operator __m256() const {
    return values;
  }
  void load(const void *ptr) {
    values = _mm256_loadu_ps(reinterpret_cast<const float*>(ptr));
  }
  void load_partial(const void *ptr, int count) {
    float tmp_values[size];
    std::memcpy(tmp_values, ptr, count * sizeof(float));
    load(tmp_values);
  }
  static Vec256<float> s_load(const void* ptr) {
    Vec256<float> vec;
    vec.load(ptr);
    return vec;
  }
  void store(void *ptr) const {
    _mm256_storeu_ps(reinterpret_cast<float*>(ptr), values);
  }
  void store_partial(void* ptr, int count) const {
    float tmp_values[size];
    store(tmp_values);
    std::memcpy(ptr, tmp_values, count * sizeof(float));
  }
  Vec256<float> map(float (*f)(float)) const {
    __at_align32__ float tmp[8];
    store(tmp);
    for (int64_t i = 0; i < 8; i++) {
      tmp[i] = f(tmp[i]);
    }
    return s_load(tmp);
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
  Vec256<float> trunc() const {
    return _mm256_round_ps(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  }
  Vec256<float> sqrt() const {
    return _mm256_sqrt_ps(values);
  }

  /*
   * This is an AVX implementation of the cephes tanh function for single
   * precision. http://www.netlib.org/cephes/
   */
  Vec256<float> tanh() const {
    __m256 s, y, z;
    __m256 xmm0, xmm1, xmm2, xmm3;
    __m256 one = _mm256_set1_ps(1.0f);

    // z is absolute value of the loaded values (x)
    z = _mm256_and_ps(
        values, _mm256_castsi256_ps(_mm256_set1_epi32(~0x80000000)));
    __m256 sign_bit = _mm256_and_ps(
        values, _mm256_castsi256_ps(_mm256_set1_epi32((int)0x80000000)));

    // if (z >= 0.625)
    // {
    //  s = expf(z + z);
    //  z =  1.0  - 2.0/(s + 1.0);
    //  if (x < 0)
    //    z = -z;
    //  }
    // }
    xmm1 = _mm256_cmp_ps(z, _mm256_set1_ps(0.625f), _CMP_LT_OS);
    xmm2 = _mm256_add_ps(z, z);

    // using Sleef_expf8_u10 for e^(2x)
    s = Sleef_expf8_u10(xmm2);
    xmm2 = _mm256_sub_ps(
        one, _mm256_div_ps(_mm256_set1_ps(2.0f), _mm256_add_ps(one, s)));
    xmm2 = _mm256_or_ps(sign_bit, xmm2);

    // z = x * x;
    // z =
    //  ((((-5.70498872745E-3 * z
    //  + 2.06390887954E-2) * z
    //  - 5.37397155531E-2) * z
    //  + 1.33314422036E-1) * z
    //  - 3.33332819422E-1) * z * x
    //  + x;
    z = _mm256_mul_ps(values, values);
    y = _mm256_mul_ps(z, _mm256_set1_ps(-5.70498872745E-3f));
    y = _mm256_add_ps(y, _mm256_set1_ps(2.06390887954E-2f));
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, _mm256_set1_ps(-5.37397155531E-2f));
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.33314422036E-1f));
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, _mm256_set1_ps(-3.33332819422E-1f));
    y = _mm256_mul_ps(y, z);
    y = _mm256_mul_ps(y, values);
    xmm3 = _mm256_add_ps(y, values);

    s = _mm256_add_ps(_mm256_and_ps(xmm1, xmm3), _mm256_andnot_ps(xmm1, xmm2));

    // MAXLOGF = 88.02969187150841
    // if (z > 0.5 * MAXLOGF)
    // {
    //  if (x > 0)
    //    return(1.0);
    //  else
    //    return(-1.0);
    // }
    xmm0 = _mm256_cmp_ps(z, _mm256_set1_ps(44.014845935754205f), _CMP_LE_OS);
    xmm2 = _mm256_xor_ps(sign_bit, one);
    return _mm256_add_ps(_mm256_and_ps(xmm0, s), _mm256_andnot_ps(xmm0, xmm2));
  }
};

template <>
Vec256<float> inline operator+(const Vec256<float>& a, const Vec256<float>& b) {
  return _mm256_add_ps(a, b);
}

template <>
Vec256<float> inline operator*(const Vec256<float>& a, const Vec256<float>& b) {
  return _mm256_mul_ps(a, b);
}

template <>
Vec256<float> inline operator/(const Vec256<float>& a, const Vec256<float>& b) {
  return _mm256_div_ps(a, b);
}

#endif

}}}
