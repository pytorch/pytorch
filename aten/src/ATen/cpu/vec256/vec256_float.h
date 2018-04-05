#pragma once

#include "intrinsics.h"
#include "vec256_base.h"

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
  Vec256<float> exp() const {
    return map(std::exp);
  }
  Vec256<float> log() const {
    return map(std::log);
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

}}}
