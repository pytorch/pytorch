#pragma once

#include <torch/csrc/jit/code_template.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cpu {

static auto vec256_template = CodeTemplate(R"(

// ********** Vec256 **********
#if defined(__GNUC__)
#define __at_align32__ __attribute__((aligned(32)))
#elif defined(_WIN32)
#define __at_align32__ __declspec(align(32))
#else
#define __at_align32__
#endif

#if defined(__AVX__) && !defined(_MSC_VER)
#include <x86intrin.h>

// Foward declare sleef deps
extern "C" {
const __m256 Sleef_logf8_u10(__m256);
const __m256 Sleef_log2f8_u10(__m256);
const __m256 Sleef_log10f8_u10(__m256);
const __m256 Sleef_log1pf8_u10(__m256);
const __m256 Sleef_expf8_u10(__m256);
const __m256 Sleef_expm1f8_u10(__m256);
const __m256 Sleef_erff8_u10(__m256);
const __m256 Sleef_erfcf8_u15avx(__m256);
const __m256 Sleef_acosf8_u10(__m256);
const __m256 Sleef_asinf8_u10(__m256);
const __m256 Sleef_atanf8_u10(__m256);
const __m256 Sleef_tanhf8_u10(__m256);
const __m256 Sleef_powf8_u10(__m256, __m256);
const __m256 Sleef_erfcf8_u15(__m256);
const __m256 Sleef_tanf8_u10(__m256);
const __m256 Sleef_cosf8_u10(__m256);
const __m256 Sleef_coshf8_u10(__m256);
const __m256 Sleef_sinf8_u10(__m256);
const __m256 Sleef_sinhf8_u10(__m256);
}  // extern "C"

#endif

namespace {
template<size_t n> struct int_of_size;

#define DEFINE_INT_OF_SIZE(int_t) \
template<> struct int_of_size<sizeof(int_t)> { using type = int_t; }

DEFINE_INT_OF_SIZE(int64_t);
DEFINE_INT_OF_SIZE(int32_t);
DEFINE_INT_OF_SIZE(int16_t);
DEFINE_INT_OF_SIZE(int8_t);

#undef DEFINE_INT_OF_SIZE

template <typename T>
using int_same_size_t = typename int_of_size<sizeof(T)>::type;

template <class T>
struct Vec256 {
private:
  T values[32 / sizeof(T)] = {0};
public:
  static constexpr int size() {
    return 32 / sizeof(T);
  }
  Vec256() {}
  Vec256(T val) {
    for (int i = 0; i != size(); i++) {
      values[i] = val;
    }
  }
  template <typename Other>
  Vec256(Vec256<Other> o) {
    for (size_t i = 0; i < size(); ++i) {
      values[i] = o[i];
    }
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
  void store(void* ptr, int count = size()) const {
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
    for (int64_t i = 0; i != size(); i++) {
      ret[i] = f(values[i]);
    }
    return ret;
  }
  Vec256<T> abs() const {
    Vec256<T> ret;
    for (int64_t i = 0; i < size(); i++) {
      ret[i] = values[i] < 0 ? -values[i] : values[i];
    }
    return ret;
  }
  Vec256<T> neg() const {
    return map([](T x) { return -x; });
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
  Vec256<T> exp() const {
    return map(std::exp);
  }
  Vec256<T> expm1() const {
    return map(std::expm1);
  }
  Vec256<T> erf() const {
    return map(std::erf);
  }
  Vec256<T> erfc() const {
    return map(std::erfc);
  }
  Vec256<T> cos() const {
    return map(std::cos);
  }
  Vec256<T> acos() const {
    return map(std::acos);
  }
  Vec256<T> cosh() const {
    return map(std::cosh);
  }
  Vec256<T> sin() const {
    return map(std::sin);
  }
  Vec256<T> asin() const {
    return map(std::asin);
  }
  Vec256<T> sinh() const {
    return map(std::sinh);
  }
  Vec256<T> tan() const {
    return map(std::tan);
  }
  Vec256<T> atan() const {
    return map(std::atan);
  }
  Vec256<T> tanh() const {
    return map(std::tanh);
  }
  Vec256<T> sqrt() const {
    return map(std::sqrt);
  }
  Vec256<T> rsqrt() const {
    return map([](T x) { return 1 / std::sqrt(x); });
  }
  Vec256<T> ceil() const {
    return map(std::ceil);
  }
  Vec256<T> floor() const {
    return map(std::floor);
  }
  Vec256<T> round() const {
    return map(std::nearbyint);
  }
  Vec256<T> trunc() const {
    return map(std::trunc);
  }
  Vec256<T> reciprocal() const {
    return map([](T x) { return (T)(1) / x; });
  }
  Vec256<T> pow(const Vec256<T> &exp) const {
    Vec256<T> ret;
    for (int64_t i = 0; i < size(); i++) {
      ret[i] = std::pow(values[i], exp[i]);
    }
    return ret;
  }
  Vec256<T> where(const Vec256<T> &x, const Vec256<T> &y) {
    Vec256<T> ret;
    for (int64_t i = 0; i < size(); i++) {
      ret[i] = values[i] ? x[i] : y[i];
    }
    return ret;
  }
  #define DEFINE_COMP(binary_pred)                                              \
    Vec256<T> operator binary_pred(const Vec256<T> &other) const {              \
      Vec256<T> vec;                                                            \
      for (int64_t i = 0; i != size(); i++) {                                   \
        if (values[i] binary_pred other.values[i]) {                            \
          std::memset(static_cast<void*>(vec.values + i), 0xFF, sizeof(T));     \
        } else {                                                                \
          std::memset(static_cast<void*>(vec.values + i), 0, sizeof(T));        \
        }                                                                       \
      }                                                                         \
      return vec;                                                               \
    }
    DEFINE_COMP(==)
    DEFINE_COMP(!=)
    DEFINE_COMP(>=)
    DEFINE_COMP(<=)
    DEFINE_COMP(>)
    DEFINE_COMP(<)
  #undef DEFINE_COMP
};

template <class T> Vec256<T> inline operator+(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != Vec256<T>::size(); i++) {
    c[i] = a[i] + b[i];
  }
  return c;
}

template <class T> Vec256<T> inline operator-(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != Vec256<T>::size(); i++) {
    c[i] = a[i] - b[i];
  }
  return c;
}

template <class T> Vec256<T> inline operator*(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != Vec256<T>::size(); i++) {
    c[i] = a[i] * b[i];
  }
  return c;
}

template <class T> Vec256<T> inline operator/(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != Vec256<T>::size(); i++) {
    c[i] = a[i] / b[i];
  }
  return c;
}

template <class T> Vec256<T> inline maximum(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != Vec256<T>::size(); i++) {
    c[i] = (a[i] > b[i]) ? a[i] : b[i];
    if (isnan(a[i])) {
      // If either input is NaN, propagate a NaN.
      // NOTE: The case where b[i] was NaN is handled correctly by the naive
      // ternary operator above.
      c[i] = a[i];
    }
  }
  return c;
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <class T> Vec256<T> inline minimum(const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != Vec256<T>::size(); i++) {
    c[i] = (a[i] < b[i]) ? a[i] : b[i];
    if (isnan(a[i])) {
      // If either input is NaN, propagate a NaN.
      // NOTE: The case where b[i] was NaN is handled correctly by the naive
      // ternary operator above.
      c[i] = a[i];
    }
  }
  return c;
}

template <class T> Vec256<T> inline operator||(
    const Vec256<T> &a, const Vec256<T> &b) {
  Vec256<T> c = Vec256<T>();
  for (int i = 0; i != Vec256<T>::size(); i++) {
    c[i] = a[i] || b[i];
  }
  return c;
}

#define DEFINE_BITWISE_OP(op)                                               \
template <class T>                                                          \
Vec256<T> inline operator op(const Vec256<T> &a, const Vec256<T> &b) {      \
  using iT = int_same_size_t<T>;                                            \
  iT buffer[Vec256<T>::size()];                                             \
  for (int64_t i = 0; i != Vec256<T>::size(); i++) {                        \
    auto a_val = a[i];                                                      \
    auto b_val = b[i];                                                      \
    iT *i_a_ptr = reinterpret_cast<iT*>(&a_val);                            \
    iT *i_b_ptr = reinterpret_cast<iT*>(&b_val);                            \
    buffer[i] = *i_a_ptr op *i_b_ptr;                                       \
  }                                                                         \
  return Vec256<T>::loadu(buffer);                                          \
}
DEFINE_BITWISE_OP(^)
#undef DEFINE_BITWISE_OP


// ********** Vec256<float> **********
#if defined(__AVX__) && !defined(_MSC_VER)

template <> class Vec256<float> {
private:
  __m256 values;
public:
  static constexpr int size() {
    return 8;
  }
  Vec256() {}
  Vec256(__m256 v) : values(v) {}
  Vec256(float val) {
    values = _mm256_set1_ps(val);
  }
  Vec256(float val1, float val2, float val3, float val4,
         float val5, float val6, float val7, float val8) {
    values = _mm256_setr_ps(val1, val2, val3, val4, val5, val6, val7, val8);
  }
  operator __m256() const {
    return values;
  }
  Vec256(__m256i o) {
    values = _mm256_cvtepi32_ps(o);
  }
  static Vec256<float> loadu(const void* ptr, int64_t count = size()) {
    if (count == size())
      return _mm256_loadu_ps(reinterpret_cast<const float*>(ptr));
    __at_align32__ float tmp_values[size()];
    std::memcpy(
        tmp_values, reinterpret_cast<const float*>(ptr), count * sizeof(float));
    return _mm256_loadu_ps(tmp_values);
  }
  void store(void* ptr, int64_t count = size()) const {
    if (count == size()) {
      _mm256_storeu_ps(reinterpret_cast<float*>(ptr), values);
    } else if (count > 0) {
      float tmp_values[size()];
      _mm256_storeu_ps(reinterpret_cast<float*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(float));
    }
  }
  Vec256<float> map(float (*f)(float)) const {
    __at_align32__ float tmp[8];
    store(tmp);
    for (int64_t i = 0; i < 8; i++) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  Vec256<float> abs() const {
    auto mask = _mm256_set1_ps(-0.f);
    return _mm256_andnot_ps(mask, values);
  }
  Vec256<float> neg() const {
    return _mm256_xor_ps(_mm256_set1_ps(-0.f), values);
  }
  Vec256<float> log() const {
    return Vec256<float>(Sleef_logf8_u10(values));
  }
  Vec256<float> log2() const {
    return Vec256<float>(Sleef_log2f8_u10(values));
  }
  Vec256<float> log10() const {
    return Vec256<float>(Sleef_log10f8_u10(values));
  }
  Vec256<float> log1p() const {
    return Vec256<float>(Sleef_log1pf8_u10(values));
  }
  Vec256<float> exp() const {
    return Vec256<float>(Sleef_expf8_u10(values));
  }
  Vec256<float> expm1() const {
    return Vec256<float>(Sleef_expm1f8_u10(values));
  }
  Vec256<float> erf() const {
    return Vec256<float>(Sleef_erff8_u10(values));
  }
  Vec256<float> erfc() const {
    return Vec256<float>(Sleef_erfcf8_u15(values));
  }
  Vec256<float> cos() const {
    return Vec256<float>(Sleef_cosf8_u10(values));
  }
  Vec256<float> acos() const {
    return Vec256<float>(Sleef_acosf8_u10(values));
  }
  Vec256<float> cosh() const {
    return Vec256<float>(Sleef_coshf8_u10(values));
  }
  Vec256<float> sin() const {
    return Vec256<float>(Sleef_sinf8_u10(values));
  }
  Vec256<float> asin() const {
    return Vec256<float>(Sleef_asinf8_u10(values));
  }
  Vec256<float> sinh() const {
    return Vec256<float>(Sleef_sinhf8_u10(values));
  }
  Vec256<float> tan() const {
    return Vec256<float>(Sleef_tanf8_u10(values));
  }
  Vec256<float> atan() const {
    return Vec256<float>(Sleef_atanf8_u10(values));
  }
  Vec256<float> tanh() const {
    return Vec256<float>(Sleef_tanhf8_u10(values));
  }
  Vec256<float> sqrt() const {
    return _mm256_sqrt_ps(values);
  }
  Vec256<float> rsqrt() const {
    return _mm256_div_ps(_mm256_set1_ps(1), _mm256_sqrt_ps(values));
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
  Vec256<float> reciprocal() const {
    return _mm256_div_ps(_mm256_set1_ps(1), values);
  }
  // Comparison using the _CMP_**_OQ predicate.
  //   `O`: get false if an operand is NaN
  //   `Q`: do not raise if an operand is NaN
  Vec256<float> operator==(const Vec256<float>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_EQ_OQ);
  }

  Vec256<float> operator!=(const Vec256<float>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_NEQ_OQ);
  }

  Vec256<float> operator<(const Vec256<float>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_LT_OQ);
  }

  Vec256<float> operator<=(const Vec256<float>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_LE_OQ);
  }

  Vec256<float> operator>(const Vec256<float>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_GT_OQ);
  }

  Vec256<float> operator>=(const Vec256<float>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_GE_OQ);
  }
  Vec256<float> pow(const Vec256<float> &b) const {
    return Vec256<float>(Sleef_powf8_u10(values, b));
  }
  Vec256<float> where(const Vec256<float> &x, const Vec256<float> &y) {
    return _mm256_or_ps(_mm256_and_ps(x.values, values), _mm256_andnot_ps(values, y.values));
  }
};

template <>
Vec256<float> inline operator+(const Vec256<float>& a, const Vec256<float>& b) {
  return _mm256_add_ps(a, b);
}

template <>
Vec256<float> inline operator-(const Vec256<float>& a, const Vec256<float>& b) {
  return _mm256_sub_ps(a, b);
}

template <>
Vec256<float> inline operator*(const Vec256<float>& a, const Vec256<float>& b) {
  return _mm256_mul_ps(a, b);
}

template <>
Vec256<float> inline operator/(const Vec256<float>& a, const Vec256<float>& b) {
  return _mm256_div_ps(a, b);
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vec256<float> inline maximum(const Vec256<float>& a, const Vec256<float>& b) {
  Vec256<float> max = _mm256_max_ps(a, b);
  Vec256<float> isnan = _mm256_cmp_ps(a, b, _CMP_UNORD_Q);
  // Exploit the fact that all-ones is a NaN.
  return _mm256_or_ps(max, isnan);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vec256<float> inline minimum(const Vec256<float>& a, const Vec256<float>& b) {
  Vec256<float> min = _mm256_min_ps(a, b);
  Vec256<float> isnan = _mm256_cmp_ps(a, b, _CMP_UNORD_Q);
  // Exploit the fact that all-ones is a NaN.
  return _mm256_or_ps(min, isnan);
}

// ********** Vec256<int32_t> ***********

struct Vec256i {
protected:
  __m256i values;

  static inline __m256i invert(const __m256i& v) {
    const auto ones = _mm256_set1_epi64x(-1);
    return _mm256_xor_si256(ones, v);
  }
public:
  Vec256i() {}
  Vec256i(__m256i v) : values(v) {}
  operator __m256i() const {
    return values;
  }
};

template <>
struct Vec256<int32_t> : public Vec256i {
  static constexpr int size() {
    return 8;
  }
  using Vec256i::Vec256i;
  Vec256() {}
  Vec256(int32_t v) { values = _mm256_set1_epi32(v); }
  Vec256(int32_t val1, int32_t val2, int32_t val3, int32_t val4,
         int32_t val5, int32_t val6, int32_t val7, int32_t val8) {
    values = _mm256_setr_epi32(val1, val2, val3, val4, val5, val6, val7, val8);
  }
  static Vec256<int32_t> loadu(const void* ptr) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
  }
  static Vec256<int32_t> loadu(const void* ptr, int32_t count) {
    __at_align32__ int32_t tmp_values[size()];
    std::memcpy(tmp_values, ptr, count * sizeof(int32_t));
    return loadu(tmp_values);
  }
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
    } else if (count > 0) {
      __at_align32__ int32_t tmp_values[size()];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(int32_t));
    }
  }
  Vec256<int32_t> abs() const {
    return _mm256_abs_epi32(values);
  }
  Vec256<int32_t> operator==(const Vec256<int32_t>& other) const {
    return _mm256_cmpeq_epi32(values, other.values);
  }
  Vec256<int32_t> operator!=(const Vec256<int32_t>& other) const {
    return invert(_mm256_cmpeq_epi32(values, other.values));
  }
  Vec256<int32_t> operator<(const Vec256<int32_t>& other) const {
    return _mm256_cmpgt_epi32(other.values, values);
  }
  Vec256<int32_t> operator<=(const Vec256<int32_t>& other) const {
    return invert(_mm256_cmpgt_epi32(values, other.values));
  }
  Vec256<int32_t> operator>(const Vec256<int32_t>& other) const {
    return _mm256_cmpgt_epi32(values, other.values);
  }
  Vec256<int32_t> operator>=(const Vec256<int32_t>& other) const {
    return invert(_mm256_cmpgt_epi32(other.values, values));
  }
  Vec256<float> to_float() const {
    return Vec256<float>(_mm256_cvtepi32_ps(values));
  }
  Vec256<int32_t> where(const Vec256<int32_t> &x, const Vec256<int32_t> &y) {
    return _mm256_or_si256(_mm256_and_si256(x.values, values), _mm256_andnot_si256(values, y.values));
  }
};

template <>
Vec256<int32_t> inline maximum(const Vec256<int32_t>& a, const Vec256<int32_t>& b) {
  return _mm256_max_epi32(a, b);
}

template <>
Vec256<int32_t> inline operator+(const Vec256<int32_t>& a, const Vec256<int32_t>& b) {
  return _mm256_add_epi32(a, b);
}

template <>
Vec256<int32_t> inline operator-(const Vec256<int32_t>& a, const Vec256<int32_t>& b) {
  return _mm256_sub_epi32(a, b);
}

template <>
Vec256<int32_t> inline operator*(const Vec256<int32_t>& a, const Vec256<int32_t>& b) {
  return _mm256_mullo_epi32(a, b);
}

template <>
Vec256<int32_t> inline minimum(const Vec256<int32_t>& a, const Vec256<int32_t>& b) {
  return _mm256_min_epi32(a, b);
}

#endif // #if defined(__AVX__) && !defined(_MSC_VER)


}  // namespace

)");

} // namespace cpu
} // namespace fuser
} // namespace jit
} // namespace torch
