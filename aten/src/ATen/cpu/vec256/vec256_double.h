#pragma once

#include "intrinsics.h"
#include "vec256_base.h"

namespace at {
namespace vec256 {

#ifdef __AVX__

template <> class Vec256<double> {
public:
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
  static Vec256<double> s_load(const void* ptr) {
    Vec256<double> vec;
    vec.load(ptr);
    return vec;
  }
  void store(void* ptr) const {
    _mm256_storeu_pd(reinterpret_cast<double*>(ptr), values);
  }
  static constexpr int size() {
    return 4;
  }
};

template <>
Vec256<double> inline operator+(const Vec256<double> &a,
                                const Vec256<double> &b) {
  return _mm256_add_pd(a, b);
}

template <>
Vec256<double> inline operator*(const Vec256<double> &a,
                                const Vec256<double> &b) {
  return _mm256_mul_pd(a, b);
}

#endif

}}
