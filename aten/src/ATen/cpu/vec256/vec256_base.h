#pragma once

#include <cstring>

namespace at {
namespace vec256 {

// NOTE: If you specialize on a type, you must define all operations!

// emulates vectorized types
template <class T>
struct Vec256 {
  T values[32 / sizeof(T)];
  Vec256() {}
  Vec256(T val) {
    for (int i = 0; i != size(); i++) {
      values[i] = val;
    }
  }
  inline void load(const T *ptr) {
    std::memcpy(values, ptr, 32);
  };
  static Vec256 s_load(const T* ptr) {
    Vec256 vec;
    vec.load(ptr);
    return vec;
  }
  void store(T *ptr) const {
    std::memcpy(ptr, values, 32);
  }
  static constexpr int size() { return 32 / sizeof(T); }
};

template <class T> Vec256<T> operator+(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != c.size(); i++) {
    c.values[i] = a.values[i] + b.values[i];
  }
  return c;
}

template <class T> Vec256<T> operator*(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != c.size(); i++) {
    c.values[i] = a.values[i] * b.values[i];
  }
  return c;
}

}
}
