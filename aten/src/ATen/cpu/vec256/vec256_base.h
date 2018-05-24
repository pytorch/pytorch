#pragma once

#include <cstring>
#include <functional>
#include <cmath>

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
  T values[32 / sizeof(T)];
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
        vec.values[i] = b[i];
      } else {
        vec.values[i] = a[i];
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
  static Vec256<T> load(
      const void* ptr,
      int64_t count = size,
      int64_t stride = 1) {
    Vec256 vec;
    if (stride == 1) {
      std::memcpy(vec.values, ptr, count * sizeof(T));
    } else {
      for (int64_t i = 0; i < count; i++)
        vec.values[i] = reinterpret_cast<const T*>(ptr)[i * stride];
    }
    return vec;
  }
  void store(void* ptr, int64_t count = size, int64_t stride = 1) const {
    if (stride == 1) {
      std::memcpy(ptr, values, count * sizeof(T));
    } else {
      for (int64_t i = 0; i < count; i++)
        reinterpret_cast<T*>(ptr)[i * stride] = values[i];
    }
  }
  const T& operator [](int idx) const {
    return values[idx];
  }
  T& operator [](int idx) {
    return values[idx];
  }
};
template <class T>Vec256<T> inline map(T (*f)(T), Vec256<T> x) {
  Vec256<T> ret;
  for (int64_t i = 0; i != Vec256<T>::size; i++) {
    ret[i] = f(x[i]);
  }
  return ret;
}

template <class T>Vec256<T> inline abs(Vec256<T> x) {
  Vec256<T> ret;
  for (int64_t i = 0; i < Vec256<T>::size; i++) {
    ret[i] = x[i] < 0 ? -x[i] : x[i];
  }
  return ret;
}

template <class T>
Vec256<T> inline acos(Vec256<T> x) {
  return map(std::acos, x);
}

template <class T>
Vec256<T> inline asin(Vec256<T> x) {
  return map(std::asin, x);
}

template <class T>
Vec256<T> inline atan(Vec256<T> x) {
  return map(std::atan, x);
}

template <class T>
Vec256<T> inline erf(Vec256<T> x) {
  return map(std::erf, x);
}

template <class T>
Vec256<T> inline exp(Vec256<T> x) {
  return map(std::exp, x);
}

template <class T>
Vec256<T> inline expm1(Vec256<T> x) {
  return map(std::expm1, x);
}

template <class T>
Vec256<T> inline log(Vec256<T> x) {
  return map(std::log, x);
}

template <class T>
Vec256<T> inline log10(Vec256<T> x) {
  return map(std::log10, x);
}

template <class T>
Vec256<T> inline log1p(Vec256<T> x) {
  return map(std::log1p, x);
}

template <class T>
Vec256<T> inline log2(Vec256<T> x) {
  return map(std::log2, x);
}

template <class T>
Vec256<T> inline ceil(Vec256<T> x) {
  return map(std::ceil, x);
}

template <class T>
Vec256<T> inline cos(Vec256<T> x) {
  return map(std::cos, x);
}

template <class T>
Vec256<T> inline floor(Vec256<T> x) {
  return map(std::floor, x);
}

template <class T>
Vec256<T> inline round(Vec256<T> x) {
  return map(std::round, x);
}

template <class T>
Vec256<T> inline sin(Vec256<T> x) {
  return map(std::sin, x);
}

template <class T>
Vec256<T> inline sqrt(Vec256<T> x) {
  return map(std::sqrt, x);
}

template <class T>
Vec256<T> inline neg(Vec256<T> x) {
  return Vec256<T>(0) - x;
}

template <class T>
Vec256<T> inline reciprocal(Vec256<T> x) {
  Vec256<T> ret;
  for (int64_t i = 0; i < Vec256<T>::size; i++) {
    ret[i] = 1 / x[i];
  }
  return ret;
}

template <class T>
Vec256<T> inline rsqrt(Vec256<T> x) {
  return reciprocal(sqrt(x));
}

template <class T>
Vec256<T> inline sigmoid(Vec256<T> x) {
  Vec256<T> ret;
  for (int64_t i = 0; i < Vec256<T>::size; i++) {
    ret[i] = ((T)1.0) / (((T)1.0) + std::exp(-x[i]));
  }
  return ret;
}

template <class T>
Vec256<T> inline tanh(Vec256<T> x) {
  return map(std::tanh, x);
}

template <class T>
Vec256<T> inline trunc(Vec256<T> x) {
  return map(std::trunc, x);
}

template <class T>
Vec256<T> inline frac(Vec256<T> x) {
  return x - trunc(x);
}

template <class T> Vec256<T> operator+(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c;
  for (int i = 0; i != Vec256<T>::size; i++) {
    c[i] = a[i] + b[i];
  }
  return c;
}

template <class T> Vec256<T> operator-(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c;
  for (int i = 0; i != Vec256<T>::size; i++) {
    c[i] = a[i] - b[i];
  }
  return c;
}

template <class T> Vec256<T> operator*(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c;
  for (int i = 0; i != Vec256<T>::size; i++) {
    c[i] = a[i] * b[i];
  }
  return c;
}

template <class T> Vec256<T> operator/(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c;
  for (int i = 0; i != Vec256<T>::size; i++) {
    c[i] = a[i] / b[i];
  }
  return c;
}

template <class T> Vec256<T> max(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c;
  for (int i = 0; i != Vec256<T>::size; i++) {
    c[i] = std::max(a[i], b[i]);
  }
  return c;
}

template <class T> Vec256<T> min(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c;
  for (int i = 0; i != Vec256<T>::size; i++) {
    c[i] = std::min(a[i], b[i]);
  }
  return c;
}

}}}
