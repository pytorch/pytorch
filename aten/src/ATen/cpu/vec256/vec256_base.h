#pragma once

#include <cstring>

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
  static constexpr int size = 32 / sizeof(T);
  __at_align32__ T values[32 / sizeof(T)];
  Vec256() {}
  Vec256(T val) {
    for (int i = 0; i != size; i++) {
      values[i] = val;
    }
  }
  void load(const void* ptr) {
    std::memcpy(values, ptr, 32);
  };
  void load_partial(const void* ptr, int count) {
    std::memcpy(values, ptr, count * sizeof(T));
  }
  static Vec256 s_load(const T* ptr) {
    Vec256 vec;
    vec.load(ptr);
    return vec;
  }
  void store(T *ptr) const {
    std::memcpy(ptr, values, 32);
  }
  void store_partial(void* ptr, int count) const {
    std::memcpy(ptr, values, count * sizeof(T));
  }
  Vec256<T> map(T (*f)(T)) const {
    Vec256<T> ret;
    for (int64_t i = 0; i != size; i++) {
      ret.values[i] = f(values[i]);
    }
    return ret;
  }
  Vec256<T> abs() const {
    Vec256<T> ret;
    for (int64_t i = 0; i < size; i++) {
      ret.values[i] = values[i] < 0 ? -values[i] : values[i];
    }
    return ret;
  }
  Vec256<T> exp() const {
    return map(std::exp);
  }
  Vec256<T> log() const {
    return map(std::log);
  }
  Vec256<T> ceil() const {
    return map(std::ceil);
  }
  Vec256<T> cos() const {
    return map(std::cos);
  }
  Vec256<T> floor() const {
    return map(std::floor);
  }
  Vec256<T> round() const {
    return map(std::round);
  }
  Vec256<T> sin() const {
    return map(std::sin);
  }
  Vec256<T> trunc() const {
    return map(std::trunc);
  }
  Vec256<T> sqrt() const {
    return map(std::sqrt);
  }
};

template <class T> Vec256<T> operator+(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != c.size; i++) {
    c.values[i] = a.values[i] + b.values[i];
  }
  return c;
}

template <class T> Vec256<T> operator*(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != c.size; i++) {
    c.values[i] = a.values[i] * b.values[i];
  }
  return c;
}

}}}
