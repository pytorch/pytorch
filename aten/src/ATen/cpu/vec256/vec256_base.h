#pragma once

#include <cstring>
#include <functional>
#include <cmath>

#include "ATen/Utils.h"

#if defined(__GNUC__)
#define __at_align32__ __attribute__((aligned(32)))
#elif defined(_WIN32)
#define __at_align32__ __declspec(align(32))
#else
#define __at_align32__
#endif

namespace at {
namespace vec256 {
namespace {

// NOTE: If you specialize on a type, you must define all operations!

// emulates vectorized types
template <class T>
struct Vec256 {
private:
  T values[32 / sizeof(T)] = {0};
public:
  static constexpr int size = 32 / sizeof(T);
  Vec256() {}
  Vec256(T val) {
    for (int i = 0; i != size; i++) {
      values[i] = val;
    }
  }
  template <int64_t mask_>
  static Vec256<T> blend(Vec256<T> a, Vec256<T> b) {
    int64_t mask = mask_;
    Vec256 vec;
    for (int64_t i = 0; i < size; i++) {
      if (mask & 0x01) {
        vec[i] = b[i];
      } else {
        vec[i] = a[i];
      }
      mask = mask >> 1;
    }
    return vec;
  }
  static Vec256<T> set(Vec256<T> a, Vec256<T> b, int64_t count = size) {
    Vec256 vec;
    for (int64_t i = 0; i < size; i++) {
      if (i < count) {
        vec[i] = b[i];
      } else {
        vec[i] = a[i];
      }
    }
    return vec;
  }
  static Vec256<T> loadu(const void* ptr) {
    Vec256 vec;
    std::memcpy(vec.values, ptr, 32);
    return vec;
  }
  static Vec256<T> loadu(const void* ptr, int64_t count) {
    Vec256 vec;
    std::memcpy(vec.values, ptr, count * sizeof(T));
    return vec;
  }
  void store(void* ptr, int count = size) const {
    std::memcpy(ptr, values, count * sizeof(T));
  }
  const T& operator[](int idx) const {
    return values[idx];
  }
  T& operator[](int idx) {
    return values[idx];
  }
  Vec256<T> map(T (*f)(T)) const {
    Vec256<T> ret;
    for (int64_t i = 0; i != size; i++) {
      ret[i] = f(values[i]);
    }
    return ret;
  }
  Vec256<T> abs() const {
    Vec256<T> ret;
    for (int64_t i = 0; i < size; i++) {
      ret[i] = values[i] < 0 ? -values[i] : values[i];
    }
    return ret;
  }
  Vec256<T> acos() const {
    return map(std::acos);
  }
  Vec256<T> asin() const {
    return map(std::asin);
  }
  Vec256<T> atan() const {
    return map(std::atan);
  }
  Vec256<T> erf() const {
    return map(std::erf);
  }
  Vec256<T> erfc() const {
    return map(std::erfc);
  }
  Vec256<T> exp() const {
    return map(std::exp);
  }
  Vec256<T> expm1() const {
    return map(std::expm1);
  }
  Vec256<T> log() const {
    return map(std::log);
  }
  Vec256<T> log10() const {
    return map(std::log10);
  }
  Vec256<T> log1p() const {
    return map(std::log1p);
  }
  Vec256<T> log2() const {
    return map(std::log2);
  }
  Vec256<T> ceil() const {
    return map(std::ceil);
  }
  Vec256<T> cos() const {
    return map(std::cos);
  }
  Vec256<T> cosh() const {
    return map(std::cosh);
  }
  Vec256<T> floor() const {
    return map(std::floor);
  }
  Vec256<T> neg() const {
    return map([](T x) { return -x; });
  }
  Vec256<T> round() const {
    return map(std::round);
  }
  Vec256<T> sin() const {
    return map(std::sin);
  }
  Vec256<T> sinh() const {
    return map(std::sinh);
  }
  Vec256<T> tan() const {
    return map(std::tan);
  }
  Vec256<T> tanh() const {
    return map(std::tanh);
  }
  Vec256<T> trunc() const {
    return map(std::trunc);
  }
  Vec256<T> sqrt() const {
    return map(std::sqrt);
  }
  Vec256<T> reciprocal() const {
    return map([](T x) { return (T)(1) / x; });
  }
  Vec256<T> rsqrt() const {
    return map([](T x) { return 1 / std::sqrt(x); });
  }
};

template <class T> Vec256<T> operator+(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != Vec256<T>::size; i++) {
    c[i] = a[i] + b[i];
  }
  return c;
}

template <class T> Vec256<T> operator-(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != Vec256<T>::size; i++) {
    c[i] = a[i] - b[i];
  }
  return c;
}

template <class T> Vec256<T> operator*(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != Vec256<T>::size; i++) {
    c[i] = a[i] * b[i];
  }
  return c;
}

template <class T> Vec256<T> operator/(const Vec256<T> &a, const Vec256<T> &b) __ubsan_ignore_float_divide_by_zero__ {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != Vec256<T>::size; i++) {
    c[i] = a[i] / b[i];
  }
  return c;
}

template <class T> Vec256<T> max(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != Vec256<T>::size; i++) {
    c[i] = std::max(a[i], b[i]);
  }
  return c;
}

template <typename T>
T fmadd(const T& a, const T& b, const T& c) {
  return a * b + c;
}

}}}
