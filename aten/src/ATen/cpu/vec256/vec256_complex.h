#pragma once

#include <complex>

namespace at {
namespace vec256 {
namespace {

// the default Vec256 does not work on gcc 4.8
// specialized manually here

// emulates vectorized types
template <>
struct Vec256<std::complex<float> > {
private:
  std::complex<float> values[4] = {0};
public:
  static constexpr int size = 4;
  Vec256() {}
  Vec256(std::complex<float> val) {
    for (int i = 0; i != size; i++) {
      values[i] = val;
    }
  }
  template<typename... Args,
           typename = c10::guts::enable_if_t<(sizeof...(Args) == size)>>
  Vec256(Args... vals) {
    values = { vals... };
  }
  template <int64_t mask_>
  static Vec256<std::complex<float>> blend(const Vec256<std::complex<float>>& a, const Vec256<std::complex<float>>& b) {
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
  static Vec256<std::complex<float>> blendv(const Vec256<std::complex<float>>& a, const Vec256<std::complex<float>>& b,
                          const Vec256<std::complex<float>>& mask) {
    Vec256 vec;
    int64_t buffer[size];
    mask.store(buffer);
    for (int64_t i = 0; i < size; i++) {
      if (buffer[i] & 0x01)
       {
        vec[i] = b[i];
      } else {
        vec[i] = a[i];
      }
    }
    return vec;
  }
  static Vec256<std::complex<float>> arange(std::complex<float> base = static_cast<std::complex<float>>(0), 
    std::complex<float> step = static_cast<std::complex<float>>(1)) {
    Vec256 vec;
    for (int64_t i = 0; i < size; i++) {
      vec.values[i] = base + static_cast<std::complex<float>>(i) * step;
    }
    return vec;
  }
  static Vec256<std::complex<float>> set(const Vec256<std::complex<float>>& a, const Vec256<std::complex<float>>& b, int64_t count = size) {
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
  static Vec256<std::complex<float>> loadu(const void* ptr) {
    Vec256 vec;
    std::memcpy(vec.values, ptr, 32);
    return vec;
  }
  static Vec256<std::complex<float>> loadu(const void* ptr, int64_t count) {
    Vec256 vec;
    std::memcpy(vec.values, ptr, count * sizeof(std::complex<float>));
    return vec;
  }
  void store(void* ptr, int count = size) const {
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
    for (int64_t i = 0; i != size; i++) {
      ret[i] = f(values[i]);
    }
    return ret;
  }
  Vec256<std::complex<float>> abs() const {
    Vec256<std::complex<float>> ret;
    for (int64_t i = 0; i < size; i++) {
      ret[i] = std::abs(values[i]);
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
  Vec256<std::complex<float>> log10() const {
    return map(std::log10);
  }
  Vec256<std::complex<float>> log1p() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<float>> log2() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<float>> ceil() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<float>> cos() const {
    return map(std::cos);
  }
  Vec256<std::complex<float>> cosh() const {
    return map(std::cosh);
  }
  Vec256<std::complex<float>> floor() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<float>> neg() const {
    return map([](const std::complex<float> &x) { return -x; });
  }
  Vec256<std::complex<float>> round() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<float>> sin() const {
    return map(std::sin);
  }
  Vec256<std::complex<float>> sinh() const {
    return map(std::sinh);
  }
  Vec256<std::complex<float>> tan() const {
    return map(std::tan);
  }
  Vec256<std::complex<float>> tanh() const {
    return map(std::tanh);
  }
  Vec256<std::complex<float>> trunc() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<float>> sqrt() const {
    return map(std::sqrt);
  }
  Vec256<std::complex<float>> reciprocal() const {
    return map([](const std::complex<float> &x) { return (std::complex<float>)(1) / x; });
  }
  Vec256<std::complex<float>> rsqrt() const {
    return map([](const std::complex<float> &x) { return static_cast<std::complex<float>>(1) / std::sqrt(x); });
  }
  Vec256<std::complex<float>> pow(const Vec256<std::complex<float>> &exp) const {
    Vec256<std::complex<float>> ret;
    for (int64_t i = 0; i < size; i++) {
      ret[i] = std::pow(values[i], exp[i]);
    }
    return ret;
  }
#define DEFINE_COMP(binary_pred)                                              \
  Vec256<std::complex<float>> operator binary_pred(const Vec256<std::complex<float>> &other) const {              \
    Vec256<std::complex<float>> vec;                                                            \
    for (int64_t i = 0; i != size; i++) {                                     \
      if (values[i] binary_pred other.values[i]) {                            \
        std::memset(static_cast<void*>(vec.values + i), 0xFF, sizeof(std::complex<float>));     \
      } else {                                                                \
        std::memset(static_cast<void*>(vec.values + i), 0, sizeof(std::complex<float>));        \
      }                                                                       \
    }                                                                         \
    return vec;                                                               \
  }
  DEFINE_COMP(==)
  DEFINE_COMP(!=)
#undef DEFINE_COMP
};

template <>
struct Vec256<std::complex<double> > {
private:
  std::complex<double> values[4] = {0};
public:
  static constexpr int size = 4;
  Vec256() {}
  Vec256(std::complex<double> val) {
    for (int i = 0; i != size; i++) {
      values[i] = val;
    }
  }
  template<typename... Args,
           typename = c10::guts::enable_if_t<(sizeof...(Args) == size)>>
  Vec256(Args... vals) {
    values = { vals... };
  }
  template <int64_t mask_>
  static Vec256<std::complex<double>> blend(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& b) {
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
  static Vec256<std::complex<double>> blendv(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& b,
                          const Vec256<std::complex<double>>& mask) {
    Vec256 vec;
    int64_t buffer[size];
    mask.store(buffer);
    for (int64_t i = 0; i < size; i++) {
      if (buffer[i] & 0x01)
       {
        vec[i] = b[i];
      } else {
        vec[i] = a[i];
      }
    }
    return vec;
  }
  static Vec256<std::complex<double>> arange(std::complex<double> base = static_cast<std::complex<double>>(0), 
    std::complex<double> step = static_cast<std::complex<double>>(1)) {
    Vec256 vec;
    for (int64_t i = 0; i < size; i++) {
      vec.values[i] = base + static_cast<std::complex<double>>(i) * step;
    }
    return vec;
  }
  static Vec256<std::complex<double>> set(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& b, int64_t count = size) {
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
  static Vec256<std::complex<double>> loadu(const void* ptr) {
    Vec256 vec;
    std::memcpy(vec.values, ptr, 32);
    return vec;
  }
  static Vec256<std::complex<double>> loadu(const void* ptr, int64_t count) {
    Vec256 vec;
    std::memcpy(vec.values, ptr, count * sizeof(std::complex<double>));
    return vec;
  }
  void store(void* ptr, int count = size) const {
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
    for (int64_t i = 0; i != size; i++) {
      ret[i] = f(values[i]);
    }
    return ret;
  }
  Vec256<std::complex<double>> abs() const {
    Vec256<std::complex<double>> ret;
    for (int64_t i = 0; i < size; i++) {
      ret[i] = std::abs(values[i]);
    }
    return ret;
  }
  Vec256<std::complex<double>> acos() const {
    return map(std::acos);
  }
  Vec256<std::complex<double>> asin() const {
    return map(std::asin);
  }
  Vec256<std::complex<double>> atan() const {
    return map(std::atan);
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
  Vec256<std::complex<double>> log10() const {
    return map(std::log10);
  }
  Vec256<std::complex<double>> log1p() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<double>> log2() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<double>> ceil() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<double>> cos() const {
    return map(std::cos);
  }
  Vec256<std::complex<double>> cosh() const {
    return map(std::cosh);
  }
  Vec256<std::complex<double>> floor() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<double>> neg() const {
    return map([](const std::complex<double> &x) { return -x; });
  }
  Vec256<std::complex<double>> round() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<double>> sin() const {
    return map(std::sin);
  }
  Vec256<std::complex<double>> sinh() const {
    return map(std::sinh);
  }
  Vec256<std::complex<double>> tan() const {
    return map(std::tan);
  }
  Vec256<std::complex<double>> tanh() const {
    return map(std::tanh);
  }
  Vec256<std::complex<double>> trunc() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<double>> sqrt() const {
    return map(std::sqrt);
  }
  Vec256<std::complex<double>> reciprocal() const {
    return map([](const std::complex<double> &x) { return (std::complex<double>)(1) / x; });
  }
  Vec256<std::complex<double>> rsqrt() const {
    return map([](const std::complex<double> &x) { return static_cast<std::complex<double>>(1) / std::sqrt(x); });
  }
  Vec256<std::complex<double>> pow(const Vec256<std::complex<double>> &exp) const {
    Vec256<std::complex<double>> ret;
    for (int64_t i = 0; i < size; i++) {
      ret[i] = std::pow(values[i], exp[i]);
    }
    return ret;
  }
#define DEFINE_COMP(binary_pred)                                              \
  Vec256<std::complex<double>> operator binary_pred(const Vec256<std::complex<double>> &other) const {              \
    Vec256<std::complex<double>> vec;                                                            \
    for (int64_t i = 0; i != size; i++) {                                     \
      if (values[i] binary_pred other.values[i]) {                            \
        std::memset(static_cast<void*>(vec.values + i), 0xFF, sizeof(std::complex<double>));     \
      } else {                                                                \
        std::memset(static_cast<void*>(vec.values + i), 0, sizeof(std::complex<double>));        \
      }                                                                       \
    }                                                                         \
    return vec;                                                               \
  }
  DEFINE_COMP(==)
  DEFINE_COMP(!=)
#undef DEFINE_COMP
};


}}}
