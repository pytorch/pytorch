#pragma once

#include <ATen/cpu/vec256/intrinsics.h>
#include <ATen/cpu/vec256/vec256_base.h>
#if defined(__AVX__) && !defined(_MSC_VER)
#include <sleef.h>
#endif

namespace at {
namespace vec256 {
// See Note [Acceptable use of anonymous namespace in header]
namespace {

template <> class Vec256<std::complex<double>> {
private:
  std::complex<double> values[2] = {0};
public:
  using value_type = std::complex<double>;
  static constexpr int size() {
    return 2;
  }
  Vec256() {}
  Vec256(std::complex<double> val) {
    for (int i = 0; i != size(); i++) {
      values[i] = val;
    }
  }
  Vec256(std::complex<double> val1, std::complex<double> val2) {
    values[0] = val1;
    values[1] = val2;
  }
  operator __m256d() = delete;
  template <int64_t mask>
  static Vec256<std::complex<double>> blend(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& b) {
    int64_t mask_ = mask;
    Vec256 vec;
    for (int i = 0; i < size(); i++) {
      if (mask & 0x01) {
        vec[i] = b[i];
      } else {
        vec[i] = a[i];
      }
      mask_ = mask_ >> 1;
    }
    return vec;
  }
  static Vec256<std::complex<double>> blendv(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& b,
                               const Vec256<std::complex<double>>& mask) {
    Vec256 vec;
    int64_t buffer[size()];
    mask.store(buffer);
    for (int i = 0; i < size(); i++) {
      if (buffer[i] & 0x01)
       {
        vec[i] = b[i];
      } else {
        vec[i] = a[i];
      }
    }
    return vec;
  }
  static Vec256<std::complex<double>> arange(std::complex<double> base = 0., std::complex<double> step = 1.) {
    return Vec256(base, base + step);
  }
  static Vec256<std::complex<double>> set(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& b,
                            int64_t count = size()) {
    Vec256 vec;
    for (int i = 0; i < size(); i++) {
      if (i < count) {
        vec[i] = b[i];
      } else {
        vec[i] = a[i];
      }
    }
    return vec;
  }
  static Vec256<std::complex<double>> loadu(const void* ptr, int64_t count = size()) {
    Vec256 vec;
    std::memcpy(vec.values, ptr, count * sizeof(std::complex<double>));
    return vec;
  }
  void store(void* ptr, int count = size()) const {
    std::memcpy(ptr, values, count * sizeof(std::complex<double>));
  }
  const std::complex<double>& operator[](int idx) const {
    return values[idx];
  }
  std::complex<double>& operator[](int idx) {
    return values[idx];
  }
  Vec256<std::complex<double>> map(std::complex<double> (*f)(const std::complex<double> &)) const {
    Vec256<std::complex<double>> ret;
    for (int i = 0; i != size(); i++) {
      ret[i] = f(values[i]);
    }
    return ret;
  }
  Vec256<std::complex<double>> abs() const {
    Vec256<std::complex<double>> ret;
    for (int i = 0; i < size(); i++) {
      ret[i] = std::abs(values[i]);
    }
    return ret;
  }
  Vec256<std::complex<double>> real() const {
    Vec256<std::complex<double>> ret;
    for (int i = 0; i < size(); i++) {
      ret[i] = std::real(values[i]);
    }
    return ret;
  }
  Vec256<std::complex<double>> imag() const {
    Vec256<std::complex<double>> ret;
    for (int i = 0; i < size(); i++) {
      ret[i] = std::imag(values[i]);
    }
    return ret;
  }
  #if 0
  Vec256<std::complex<double>> acos() const {
    return map(std::acos);
  }
  Vec256<std::complex<double>> asin() const {
    return map(std::asin);
  }
  Vec256<std::complex<double>> atan() const {
    return map(std::atan);
  }
  Vec256<std::complex<double>> atan2(const Vec256<std::complex<double>> &b) const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<double>> erf() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<double>> erfc() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<double>> exp() const {
    return map(std::exp);
  }
  Vec256<std::complex<double>> expm1() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<double>> log() const {
    return map(std::log);
  }
  Vec256<std::complex<double>> log2() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<double>> log10() const {
    return map(std::log10);
  }
  Vec256<std::complex<double>> log1p() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<double>> sin() const {
    return map(std::sin);
  }
  Vec256<std::complex<double>> sinh() const {
    return map(std::sinh);
  }
  Vec256<std::complex<double>> cos() const {
    return map(std::cos);
  }
  Vec256<std::complex<double>> cosh() const {
    return map(std::cosh);
  }
  Vec256<std::complex<double>> ceil() const {
    return map([](const std::complex<double> &x) {
      return std::complex<double>(std::ceil(std::real(x)), std::ceil(std::imag(x)));
    });
  }
  Vec256<std::complex<double>> floor() const {
    return map([](const std::complex<double> &x) {
      return std::complex<double>(std::floor(std::real(x)), std::floor(std::imag(x)));
    });
  }
  Vec256<std::complex<double>> frac() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<double>> neg() const {
    return map([](const std::complex<double> &x) { return -x; });
  }
  Vec256<std::complex<double>> round() const {
    return map([](const std::complex<double> &x) {
        return std::complex<double>(std::round(std::real(x)), std::round(std::imag(x)));
    });
  }
  Vec256<std::complex<double>> tan() const {
    return map(std::tan);
  }
  Vec256<std::complex<double>> tanh() const {
      return map(std::tanh);
  }
  Vec256<std::complex<double>> trunc() const {
    return map([](const std::complex<double> &x) {
        return std::complex<double>(std::trunc(std::real(x)), std::trunc(std::imag(x)));
    });
  }
  Vec256<std::complex<double>> sqrt() const {
    return map(std::sqrt);
  }
  Vec256<std::complex<double>> reciprocal() const {
    return map([](const std::complex<double> &x) { return std::complex<double>(1) / x; });
  }
  Vec256<std::complex<double>> rsqrt() const {
    return map([](const std::complex<double> &x) { return std::complex<double>(1., 0.) / std::sqrt(x); });
  }
  Vec256<std::complex<double>> pow(const Vec256<std::complex<double>> &b) const {
    Vec256<std::complex<double>> ret;
    for (int i = 0; i < size(); i++) {
      ret[i] = std::pow(values[i], b[i]);
    }
    return ret;
  }

#define DEFINE_COMPLEX_DOUBLE_COMP(binary_pred)                                                                                    \
  Vec256<std::complex<double>> operator binary_pred(const Vec256<std::complex<double>> &other) const {              \
    Vec256<std::complex<double>> vec;                                                                               \
    for (int i = 0; i != size(); i++) {                                                                             \
      if (values[i] binary_pred other.values[i]) {                                                                  \
        std::memset(static_cast<void*>(vec.values + i), 0xFF, sizeof(std::complex<double>));                        \
      } else {                                                                                                      \
        std::memset(static_cast<void*>(vec.values + i), 0, sizeof(std::complex<double>));                           \
      }                                                                                                             \
    }                                                                                                               \
    return vec;                                                                                                     \
  }
  DEFINE_COMPLEX_DOUBLE_COMP(==)
  DEFINE_COMPLEX_DOUBLE_COMP(!=)
  Vec256<std::complex<double>> operator<(const Vec256<std::complex<double>>& other) const {
    AT_ERROR("not supported for complex numbers");
  }

  Vec256<std::complex<double>> operator<=(const Vec256<std::complex<double>>& other) const {
    AT_ERROR("not supported for complex numbers");
  }

  Vec256<std::complex<double>> operator>(const Vec256<std::complex<double>>& other) const {
    AT_ERROR("not supported for complex numbers");
  }

  Vec256<std::complex<double>> operator>=(const Vec256<std::complex<double>>& other) const {
    AT_ERROR("not supported for complex numbers");
  }

  #endif

};

#if 0
template <>
Vec256<std::complex<double>> inline operator+(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& b) {
  Vec256<std::complex<double>> ret;
  for (int i = 0; i < Vec256<std::complex<double>>::size(); i++) {
    ret[i] = a[i] + b[i];
  }
  return ret;
}

template <>
Vec256<std::complex<double>> inline operator-(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& b) {
  Vec256<std::complex<double>> ret;
  for (int i = 0; i < Vec256<std::complex<double>>::size(); i++) {
    ret[i] = a[i] + b[i];
  }
  return ret;
}

template <>
Vec256<std::complex<double>> inline operator*(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& b) {
  Vec256<std::complex<double>> ret;
  for (int i = 0; i < Vec256<std::complex<double>>::size(); i++) {
    ret[i] = a[i] * b[i];
  }
  return ret;
}

template <>
Vec256<std::complex<double>> inline operator/(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& b) {
  Vec256<std::complex<double>> ret;
  for (int i = 0; i < Vec256<std::complex<double>>::size(); i++) {
    ret[i] = a[i] / b[i];
  }
  return ret;
}

template <>
Vec256<std::complex<double>> inline maximum(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& b) {
  AT_ERROR("not supported for complex numbers");
}

template <>
Vec256<std::complex<double>> inline minimum(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& b) {
  AT_ERROR("not supported for complex numbers");
}

template <>
Vec256<std::complex<double>> inline clamp(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& min, const Vec256<std::complex<double>>& max) {
  AT_ERROR("not supported for complex numbers");
}

template <>
Vec256<std::complex<double>> inline clamp_min(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& min) {
  AT_ERROR("not supported for complex numbers");
}

template <>
Vec256<std::complex<double>> inline clamp_max(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& max) {
  AT_ERROR("not supported for complex numbers");
}

template <>
Vec256<std::complex<double>> inline operator&(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& b) {
  AT_ERROR("not supported for complex numbers");
}

template <>
Vec256<std::complex<double>> inline operator|(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& b) {
  AT_ERROR("not supported for complex numbers");
}

template <>
Vec256<std::complex<double>> inline operator^(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& b) {
  AT_ERROR("not supported for complex numbers");
}

template <>
inline void convert(const std::complex<double>* src, std::complex<double>* dst, int64_t n) {
  AT_ERROR("not supported for complex numbers");
}

#endif

}}}
