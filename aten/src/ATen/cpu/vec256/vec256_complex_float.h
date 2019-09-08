#pragma once

#include <ATen/cpu/vec256/intrinsics.h>
#include <ATen/cpu/vec256/vec256_base.h>
#include <complex>

namespace at {
namespace vec256 {
// See Note [Acceptable use of anonymous namespace in header]
namespace {

template <> class Vec256<std::complex<float>> {
private:
  std::complex<float> values[4] = {0};
public:
  using value_type = std::complex<float>;
  static constexpr int size() {
    return 2;
  }
  Vec256() {}
  Vec256(std::complex<float> val) {
    for (int i = 0; i != size(); i++) {
      values[i] = val;
    }
  }
  Vec256(std::complex<float> val1, std::complex<float> val2, std::complex<float> val3, std::complex<float> val4) {
    values[0] = val1;
    values[1] = val2;
    values[2] = val3;
    values[3] = val4;
  }
  operator __m256d() = delete;
  template <int64_t mask>
  static Vec256<std::complex<float>> blend(const Vec256<std::complex<float>>& a, const Vec256<std::complex<float>>& b) {
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
  static Vec256<std::complex<float>> blendv(const Vec256<std::complex<float>>& a, const Vec256<std::complex<float>>& b,
                               const Vec256<std::complex<float>>& mask) {
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
  static Vec256<std::complex<float>> arange(std::complex<float> base = 0., std::complex<float> step = 1.) {
    Vec256 vec;
    for (int i = 0; i < size(); i++) {
      vec.values[i] = base + static_cast<std::complex<float>>(i)*step;
    }
    return vec;
  }
  static Vec256<std::complex<float>> set(const Vec256<std::complex<float>>& a, const Vec256<std::complex<float>>& b,
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
  static Vec256<std::complex<float>> loadu(const void* ptr, int64_t count = size()) {
    Vec256 vec;
    std::memcpy(vec.values, ptr, count * sizeof(std::complex<float>));
    return vec;
  }
  void store(void* ptr, int count = size()) const {
    std::memcpy(ptr, values, count * sizeof(std::complex<float>));
  }
  const std::complex<float>& operator[](int idx) const {
    return values[idx];
  }
  std::complex<float>& operator[](int idx) {
    return values[idx];
  }
  Vec256<std::complex<float>> map(std::complex<float> (*f)(const std::complex<float> &)) const {
    Vec256<std::complex<float>> ret;
    for (int i = 0; i != size(); i++) {
      ret[i] = f(values[i]);
    }
    return ret;
  }
  Vec256<std::complex<float>> abs() const {
    Vec256<std::complex<float>> ret;
    for (int i = 0; i < size(); i++) {
      ret[i] = std::abs(values[i]);
    }
    return ret;
  }
  Vec256<std::complex<float>> real() const {
    Vec256<std::complex<float>> ret;
    for (int i = 0; i < size(); i++) {
      ret[i] = std::real(values[i]);
    }
    return ret;
  }
  Vec256<std::complex<float>> imag() const {
    Vec256<std::complex<float>> ret;
    for (int i = 0; i < size(); i++) {
      ret[i] = std::imag(values[i]);
    }
    return ret;
  }
  Vec256<std::complex<float>> acos() const {
    return map(std::acos);
  }
  Vec256<std::complex<float>> asin() const {
    return map(std::asin);
  }
  Vec256<std::complex<float>> atan() const {
    return map(std::atan);
  }
  Vec256<std::complex<float>> atan2(const Vec256<std::complex<float>> &b) const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<float>> erf() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<float>> erfc() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<float>> exp() const {
    return map(std::exp);
  }
  Vec256<std::complex<float>> expm1() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<float>> log() const {
    return map(std::log);
  }
  Vec256<std::complex<float>> log2() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<float>> log10() const {
    return map(std::log10);
  }
  Vec256<std::complex<float>> log1p() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<float>> sin() const {
    return map(std::sin);
  }
  Vec256<std::complex<float>> sinh() const {
    return map(std::sinh);
  }
  Vec256<std::complex<float>> cos() const {
    return map(std::cos);
  }
  Vec256<std::complex<float>> cosh() const {
    return map(std::cosh);
  }
  Vec256<std::complex<float>> ceil() const {
    return map([](const std::complex<float> &x) {
      return std::complex<float>(std::ceil(std::real(x)), std::ceil(std::imag(x)));
    });
  }
  Vec256<std::complex<float>> floor() const {
    return map([](const std::complex<float> &x) {
      return std::complex<float>(std::floor(std::real(x)), std::floor(std::imag(x)));
    });
  }
  Vec256<std::complex<float>> frac() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<float>> neg() const {
    return map([](const std::complex<float> &x) { return -x; });
  }
  Vec256<std::complex<float>> round() const {
    return map([](const std::complex<float> &x) {
      return std::complex<float>(std::round(std::real(x)), std::round(std::imag(x)));
    });
  }
  Vec256<std::complex<float>> tan() const {
    return map(std::tan);
  }
  Vec256<std::complex<float>> tanh() const {
      return map(std::tanh);
  }
  Vec256<std::complex<float>> trunc() const {
    return map([](const std::complex<float> &x) {
      return std::complex<float>(std::trunc(std::real(x)), std::trunc(std::imag(x)));
    });
  }
  Vec256<std::complex<float>> sqrt() const {
    return map(std::sqrt);
  }
  Vec256<std::complex<float>> reciprocal() const {
    return map([](const std::complex<float> &x) { return std::complex<float>(1) / x; });
  }
  Vec256<std::complex<float>> rsqrt() const {
    return map([](const std::complex<float> &x) { return std::complex<float>(1., 0.) / std::sqrt(x); });
  }
  Vec256<std::complex<float>> pow(const Vec256<std::complex<float>> &b) const {
    Vec256<std::complex<float>> ret;
    for (int i = 0; i < size(); i++) {
      ret[i] = std::pow(values[i], b[i]);
    }
    return ret;
  }

#define DEFINE_COMPLEX_FLOAT_COMP(binary_pred)                                                                                    \
  Vec256<std::complex<float>> operator binary_pred(const Vec256<std::complex<float>> &other) const {              \
    Vec256<std::complex<float>> vec;                                                                               \
    for (int i = 0; i != size(); i++) {                                                                             \
      if (values[i] binary_pred other.values[i]) {                                                                  \
        std::memset(static_cast<void*>(vec.values + i), 0xFF, sizeof(std::complex<float>));                        \
      } else {                                                                                                      \
        std::memset(static_cast<void*>(vec.values + i), 0, sizeof(std::complex<float>));                           \
      }                                                                                                             \
    }                                                                                                               \
    return vec;                                                                                                     \
  }
  DEFINE_COMPLEX_FLOAT_COMP(==)
  DEFINE_COMPLEX_FLOAT_COMP(!=)
  Vec256<std::complex<float>> operator<(const Vec256<std::complex<float>>& other) const {
    AT_ERROR("not supported for complex numbers");
  }

  Vec256<std::complex<float>> operator<=(const Vec256<std::complex<float>>& other) const {
    AT_ERROR("not supported for complex numbers");
  }

  Vec256<std::complex<float>> operator>(const Vec256<std::complex<float>>& other) const {
    AT_ERROR("not supported for complex numbers");
  }

  Vec256<std::complex<float>> operator>=(const Vec256<std::complex<float>>& other) const {
    AT_ERROR("not supported for complex numbers");
  }
};

template <>
Vec256<std::complex<float>> inline operator+(const Vec256<std::complex<float>>& a, const Vec256<std::complex<float>>& b) {
  Vec256<std::complex<float>> ret;
  for (int i = 0; i < Vec256<std::complex<float>>::size(); i++) {
    ret[i] = a[i] + b[i];
  }
  return ret;
}

template <>
Vec256<std::complex<float>> inline operator-(const Vec256<std::complex<float>>& a, const Vec256<std::complex<float>>& b) {
  Vec256<std::complex<float>> ret;
  for (int i = 0; i < Vec256<std::complex<float>>::size(); i++) {
    ret[i] = a[i] + b[i];
  }
  return ret;
}

template <>
Vec256<std::complex<float>> inline operator*(const Vec256<std::complex<float>>& a, const Vec256<std::complex<float>>& b) {
  Vec256<std::complex<float>> ret;
  for (int i = 0; i < Vec256<std::complex<float>>::size(); i++) {
    ret[i] = a[i] * b[i];
  }
  return ret;
}

template <>
Vec256<std::complex<float>> inline operator/(const Vec256<std::complex<float>>& a, const Vec256<std::complex<float>>& b) {
  Vec256<std::complex<float>> ret;
  for (int i = 0; i < Vec256<std::complex<float>>::size(); i++) {
    ret[i] = a[i] / b[i];
  }
  return ret;
}

template <>
Vec256<std::complex<float>> inline maximum(const Vec256<std::complex<float>>& a, const Vec256<std::complex<float>>& b) {
  AT_ERROR("not supported for complex numbers");
}

template <>
Vec256<std::complex<float>> inline minimum(const Vec256<std::complex<float>>& a, const Vec256<std::complex<float>>& b) {
  AT_ERROR("not supported for complex numbers");
}

template <>
Vec256<std::complex<float>> inline clamp(const Vec256<std::complex<float>>& a, const Vec256<std::complex<float>>& min, const Vec256<std::complex<float>>& max) {
  AT_ERROR("not supported for complex numbers");
}

template <>
Vec256<std::complex<float>> inline clamp_min(const Vec256<std::complex<float>>& a, const Vec256<std::complex<float>>& min) {
  AT_ERROR("not supported for complex numbers");
}

template <>
Vec256<std::complex<float>> inline clamp_max(const Vec256<std::complex<float>>& a, const Vec256<std::complex<float>>& max) {
  AT_ERROR("not supported for complex numbers");
}

template <>
Vec256<std::complex<float>> inline operator&(const Vec256<std::complex<float>>& a, const Vec256<std::complex<float>>& b) {
  AT_ERROR("not supported for complex numbers");
}

template <>
Vec256<std::complex<float>> inline operator|(const Vec256<std::complex<float>>& a, const Vec256<std::complex<float>>& b) {
  AT_ERROR("not supported for complex numbers");
}

template <>
Vec256<std::complex<float>> inline operator^(const Vec256<std::complex<float>>& a, const Vec256<std::complex<float>>& b) {
  AT_ERROR("not supported for complex numbers");
}

template <>
inline void convert(const std::complex<float>* src, std::complex<float>* dst, int64_t n) {
  AT_ERROR("not supported for complex numbers");
}

}}}
