#pragma once

#include "intrinsics.h"
#include "vec256_base.h"

namespace at {
namespace vec256 {

#ifdef __AVX__

template <> class Vec256<float> {
public:
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
  static Vec256<float> s_load(const void* ptr) {
    Vec256<float> vec;
    vec.load(ptr);
    return vec;
  }
  void store(void *ptr) const {
    _mm256_storeu_ps(reinterpret_cast<float*>(ptr), values);
  }
  static constexpr int size() {
    return 8;
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

#endif

}}
