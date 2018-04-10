#pragma once

#include "intrinsics.h"
#include "vec256_base.h"

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
  void load(const void* ptr) {
    values = _mm256_loadu_pd(reinterpret_cast<const double*>(ptr));
  }
  void load_partial(const void *ptr, int count) {
    double tmp_values[size];
    std::memcpy(tmp_values, ptr, count * sizeof(double));
    load(tmp_values);
  }
  static Vec256<double> s_load(const void* ptr) {
    Vec256<double> vec;
    vec.load(ptr);
    return vec;
  }
  void store(void* ptr) const {
    _mm256_storeu_pd(reinterpret_cast<double*>(ptr), values);
  }
  void store_partial(void* ptr, int count) const {
    double tmp_values[size];
    store(tmp_values);
    std::memcpy(ptr, tmp_values, count * sizeof(double));
  }
  Vec256<double> map(double (*f)(double)) const {
    __at_align32__ double tmp[4];
    store(tmp);
    for (int64_t i = 0; i < 4; i++) {
      tmp[i] = f(tmp[i]);
    }
    return s_load(tmp);
  }
  Vec256<double> abs() const {
    auto mask = _mm256_set1_pd(-0.f);
    return _mm256_andnot_pd(mask, values);
  }
  Vec256<double> exp() const {
    return map(std::exp);
  }
  Vec256<double> log() const {
    return map(std::log);
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
Vec256<double> inline operator*(const Vec256<double>& a, const Vec256<double>& b) {
  return _mm256_mul_pd(a, b);
}

#endif

}}}
