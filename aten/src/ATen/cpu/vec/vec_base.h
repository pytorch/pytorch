#pragma once
#if defined(__GNUC__) && __GNUC__ == 10 && __GNUC_MINOR__ <= 2 && \
    defined(__ARM_FEATURE_SVE)
// Workaround for https: //gcc.gnu.org/bugzilla/show_bug.cgi?id=117161
#pragma GCC optimize("no-tree-vectorize")
#endif

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]
//
// Note [Do not compile initializers with AVX]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// If you define a static initializer in this file, the initialization will use
// AVX instructions because these object files are compiled with AVX enabled.
// We need to avoid non-trivial global data in these architecture specific files
// because there's no way to guard the global initializers with CPU capability
// detection.
//
// See https://github.com/pytorch/pytorch/issues/37577 for an instance
// of this bug in the past.

#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstring>
#include <functional>
#include <type_traits>

#include <ATen/NumericUtils.h>
#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/native/Math.h>
#include <ATen/native/cpu/zmath.h>
#include <c10/macros/Macros.h>
#include <c10/util/BFloat16-math.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/Load.h>
#include <c10/util/TypeCast.h>
#include <c10/util/copysign.h>
#include <c10/util/irange.h>

#if defined(__GNUC__)
#define __FORCE_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#define __FORCE_INLINE __forceinline
#endif

#if defined(_MSC_FULL_VER)
/*
https://learn.microsoft.com/en-us/cpp/overview/compiler-versions?view=msvc-170
Use _MSC_FULL_VER to identify current compiler is msvc,
Windows llvm will not have this definition.
*/
#define __msvc_cl__
#endif

// These macros helped us unify vec_base.h
#ifdef CPU_CAPABILITY_AVX512
#if defined(__GNUC__)
#define __at_align__ __attribute__((aligned(64)))
#elif defined(_WIN32)
#define __at_align__ __declspec(align(64))
#else
#define __at_align__
#endif
#define VECTOR_WIDTH 64
#define int_vector __m512i
#elif defined(__aarch64__) && \
    !defined(CPU_CAPABILITY_SVE) // CPU_CAPABILITY_AVX512
// SVE code expects 256-vectors; leave that set for SVE?
#if defined(__GNUC__)
#define __at_align__ __attribute__((aligned(16)))
#elif defined(_WIN32)
#define __at_align__ __declspec(align(16))
#else
#define __at_align__
#endif
#define VECTOR_WIDTH 16
#else // CPU_CAPABILITY_AVX512
#if defined(__GNUC__)
#define __at_align__ __attribute__((aligned(32)))
#elif defined(_WIN32)
#define __at_align__ __declspec(align(32))
#else
#define __at_align__
#endif
#define VECTOR_WIDTH 32
#define int_vector __m256i
#endif // CPU_CAPABILITY_AVX512

namespace at::vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {
// at::Half and at::BFloat16 should be treated as floating point
template <typename T>
struct is_floating_point
    : std::integral_constant<
          bool,
          std::is_floating_point_v<T> || std::is_same_v<T, at::Half> ||
              std::is_same_v<T, at::BFloat16>> {};

template <typename T>
constexpr bool is_floating_point_v = is_floating_point<T>::value;

template <typename T>
struct is_reduced_floating_point
    : std::integral_constant<
          bool,
          std::is_same_v<T, at::Half> || std::is_same_v<T, at::BFloat16>> {};

template <typename T>
constexpr bool is_reduced_floating_point_v =
    is_reduced_floating_point<T>::value;

template <typename T>
struct is_8bit_integer
    : std::integral_constant<
          bool,
          std::is_same_v<T, unsigned char> || std::is_same_v<T, signed char>> {
};

template <typename T>
constexpr bool is_8bit_integer_v = is_8bit_integer<T>::value;

template <size_t n>
struct int_of_size;

#define DEFINE_INT_OF_SIZE(int_t)     \
  template <>                         \
  struct int_of_size<sizeof(int_t)> { \
    using type = int_t;               \
  }

DEFINE_INT_OF_SIZE(int64_t);
DEFINE_INT_OF_SIZE(int32_t);
DEFINE_INT_OF_SIZE(int16_t);
DEFINE_INT_OF_SIZE(int8_t);

#undef DEFINE_INT_OF_SIZE

template <typename T>
using int_same_size_t = typename int_of_size<sizeof(T)>::type;

// NOTE: If you specialize on a type, you must define all operations!

// emulates Vectorized types
#if defined(__s390x__)
template <class T, class TEMP = void>
#else
template <class T>
#endif
struct Vectorized {
 private:
  __at_align__ T values[VECTOR_WIDTH / sizeof(T)];

 public:
  using value_type = T;
  using size_type = int;

  static constexpr size_type kSize = VECTOR_WIDTH / sizeof(T);
  static constexpr size_type size() {
    return kSize;
  }
  Vectorized() : values{static_cast<T>(0)} {}
  Vectorized(T val) {
    for (int i = 0; i != size(); i++) {
      values[i] = val;
    }
  }
  template <
      typename... Args,
      typename = std::enable_if_t<(sizeof...(Args) == size())>>
  Vectorized(Args... vals) : values{vals...} {}
  Vectorized(const T (&arr)[kSize]) {
    std::memcpy(values, arr, sizeof(values));
  }
  // This also implies const T& operator[](int idx) const
  inline operator const T*() const {
    return values;
  }
  // This also implies T& operator[](int idx)
  inline operator T*() {
    return values;
  }
  // Return the values as char* for type punning
  auto as_bytes() const -> const char* {
    return reinterpret_cast<const char*>(values);
  }
  template <int64_t mask_>
  static Vectorized<T> blend(const Vectorized<T>& a, const Vectorized<T>& b) {
    int64_t mask = mask_;
    Vectorized vector;
    for (const auto i : c10::irange(size())) {
      if (mask & 0x01) {
        vector[i] = b[i];
      } else {
        vector[i] = a[i];
      }
      mask = mask >> 1;
    }
    return vector;
  }
// Workaround for https: //gcc.gnu.org/bugzilla/show_bug.cgi?id=117001
#if __GNUC__ <= 12 && !defined(__clang__) && defined(__ARM_FEATURE_SVE)
  static Vectorized<T> __attribute__((optimize("-fno-tree-loop-vectorize")))
  blendv(
      const Vectorized<T>& a,
#else
  static Vectorized<T> blendv(
      const Vectorized<T>& a,
#endif
      const Vectorized<T>& b,
      const Vectorized<T>& mask) {
    Vectorized vector;
    int_same_size_t<T> buffer[size()];
    mask.store(buffer);
#if defined(__clang__) && __ARM_FEATURE_SVE
#pragma clang loop vectorize(disable)
#endif
    for (const auto i : c10::irange(size())) {
      if (buffer[i] & 0x01) {
        vector[i] = b[i];
      } else {
        vector[i] = a[i];
      }
    }
    return vector;
  }
  template <typename step_t> // step sometimes requires a higher precision type
                             // (e.g., T=int, step_t=double)
  static Vectorized<T> arange(
      T base = static_cast<T>(0),
      step_t step = static_cast<step_t>(1)) {
    Vectorized vector;
    for (const auto i : c10::irange(size())) {
      vector.values[i] = base + i * step;
    }
    return vector;
  }
  static Vectorized<T> set(
      const Vectorized<T>& a,
      const Vectorized<T>& b,
      int64_t count = size()) {
    Vectorized vector;
    for (const auto i : c10::irange(size())) {
      if (i < count) {
        vector[i] = b[i];
      } else {
        vector[i] = a[i];
      }
    }
    return vector;
  }
  static Vectorized<T> loadu(const void* ptr) {
    Vectorized vector;
    std::memcpy(vector.values, ptr, VECTOR_WIDTH);
    return vector;
  }
  static Vectorized<T> loadu(const void* ptr, int64_t count) {
    Vectorized vector;
    std::memcpy(vector.values, ptr, count * sizeof(T));
    return vector;
  }
  static Vectorized<T> loadu_one_fourth(const void* ptr) {
    static_assert(
        std::is_same_v<T, signed char> || std::is_same_v<T, unsigned char>,
        "For byte types only");
    return Vectorized::loadu(ptr, 8);
  }

  void store(void* ptr, int count = size()) const {
    std::memcpy(ptr, values, count * sizeof(T));
  }
  int zero_mask() const {
    // returns an integer mask where all zero elements are translated to 1-bit
    // and others are translated to 0-bit
    int mask = 0;
    for (int i = 0; i < size(); ++i) {
      if (values[i] == static_cast<T>(0)) {
        mask |= (1 << i);
      }
    }
    return mask;
  }
  Vectorized<T> isnan() const {
    Vectorized<T> vector;
    for (int64_t i = 0; i != size(); i++) {
      if (_isnan(values[i])) {
        std::memset(static_cast<void*>(vector.values + i), 0xFF, sizeof(T));
      } else {
        std::memset(static_cast<void*>(vector.values + i), 0, sizeof(T));
      }
    }
    return vector;
  }
  bool has_inf_nan() const {
    for (int64_t i = 0; i != size(); i++) {
      if (_isnan(values[i]) || _isinf(values[i])) {
        return true;
      }
    }
    return false;
  }
// MSVC versions between 14.36 and 14.42 has a loop unrolling bug on Windows
// Arm64
//       See
//       https://developercommunity.visualstudio.com/t/MSVC-loop-unrolling-problem-194033813-/10720692
#if defined(_WIN32) && defined(__aarch64__) && \
    ((_MSVC_VER >= 1936) && (_MSVC_VER <= 1942))
  Vectorized<T> map(T (*const f)(T)) const {
    Vectorized<T> ret;
    for (int64_t i = 0; i < size(); i++) {
      ret[i] = f(values[i]);
      if (++i < size())
        ret[i] = f(values[i]);
    }
    return ret;
  }
  T reduce(T (*const f)(T)) const {
    T ret = 0;
    for (int64_t i = 0; i < size(); i++) {
      ret = f(ret, values[i]);
      if (++i < size())
        ret = f(ret, values[i]);
    }
    return ret;
  }
#else
  Vectorized<T> map(T (*const f)(T)) const {
    Vectorized<T> ret;
    for (int64_t i = 0; i != size(); i++) {
      ret[i] = f(values[i]);
    }
    return ret;
  }
  T reduce(T (*const f)(T)) const {
    T ret = 0;
    for (int64_t i = 0; i != size(); i++) {
      ret = f(ret, values[i]);
    }
    return ret;
  }
#endif
  Vectorized<T> map(T (*const f)(const T&)) const {
    Vectorized<T> ret;
    for (int64_t i = 0; i != size(); i++) {
      ret[i] = f(values[i]);
    }
    return ret;
  }
  T reduce(T (*const f)(const T&)) const {
    T ret = 0;
    for (int64_t i = 0; i != size(); i++) {
      ret = f(ret, values[i]);
    }
    return ret;
  }
  template <
      typename other_t_abs = T,
      typename std::enable_if_t<
          !is_floating_point_v<other_t_abs> &&
              !c10::is_complex<other_t_abs>::value,
          int> = 0>
  Vectorized<T> abs() const {
    // other_t_abs is for SFINAE and clarity. Make sure it is not changed.
    static_assert(std::is_same_v<other_t_abs, T>, "other_t_abs must be T");
    return map([](T x) -> T { return x < static_cast<T>(0) ? -x : x; });
  }
  template <
      typename float_t_abs = T,
      typename std::enable_if_t<is_floating_point_v<float_t_abs>, int> = 0>
  Vectorized<T> abs() const {
    // float_t_abs is for SFINAE and clarity. Make sure it is not changed.
    static_assert(std::is_same_v<float_t_abs, T>, "float_t_abs must be T");
    // Specifically deal with floating-point because the generic code above
    // won't handle -0.0 (which should result in 0.0) properly.
    return map([](T x) -> T { return std::abs(x); });
  }
  template <
      typename complex_t_abs = T,
      typename std::enable_if_t<c10::is_complex<complex_t_abs>::value, int> = 0>
  Vectorized<T> abs() const {
    // complex_t_abs is for SFINAE and clarity. Make sure it is not changed.
    static_assert(std::is_same_v<complex_t_abs, T>, "complex_t_abs must be T");
    // Specifically map() does not perform the type conversion needed by abs.
    return map([](T x) { return static_cast<T>(std::abs(x)); });
  }

  template <
      typename other_t_sgn = T,
      typename std::enable_if_t<c10::is_complex<other_t_sgn>::value, int> = 0>
  Vectorized<T> sgn() const {
    return map(at::native::sgn_impl);
  }

  template <
      typename other_t_angle = T,
      typename std::enable_if_t<!c10::is_complex<other_t_angle>::value, int> =
          0>
  Vectorized<T> angle() const {
    // other_t_angle is for SFINAE and clarity. Make sure it is not changed.
    static_assert(std::is_same_v<other_t_angle, T>, "other_t_angle must be T");
    return map(at::native::angle_impl<T>); // compiler is unable to resolve the
                                           // overload without <T>
  }
  template <
      typename complex_t_angle = T,
      typename std::enable_if_t<c10::is_complex<complex_t_angle>::value, int> =
          0>
  Vectorized<T> angle() const {
    // complex_t_angle is for SFINAE and clarity. Make sure it is not changed.
    static_assert(
        std::is_same_v<complex_t_angle, T>, "complex_t_angle must be T");
    return map([](T x) { return static_cast<T>(std::arg(x)); });
  }
  template <
      typename other_t_real = T,
      typename std::enable_if_t<!c10::is_complex<other_t_real>::value, int> = 0>
  Vectorized<T> real() const {
    // other_t_real is for SFINAE and clarity. Make sure it is not changed.
    static_assert(std::is_same_v<other_t_real, T>, "other_t_real must be T");
    return *this;
  }
  template <
      typename complex_t_real = T,
      typename std::enable_if_t<c10::is_complex<complex_t_real>::value, int> =
          0>
  Vectorized<T> real() const {
    // complex_t_real is for SFINAE and clarity. Make sure it is not changed.
    static_assert(
        std::is_same_v<complex_t_real, T>, "complex_t_real must be T");
    return map([](T x) { return static_cast<T>(x.real()); });
  }
  template <
      typename other_t_imag = T,
      typename std::enable_if_t<!c10::is_complex<other_t_imag>::value, int> = 0>
  Vectorized<T> imag() const {
    // other_t_imag is for SFINAE and clarity. Make sure it is not changed.
    static_assert(std::is_same_v<other_t_imag, T>, "other_t_imag must be T");
    return Vectorized(0);
  }
  template <
      typename complex_t_imag = T,
      typename std::enable_if_t<c10::is_complex<complex_t_imag>::value, int> =
          0>
  Vectorized<T> imag() const {
    // complex_t_imag is for SFINAE and clarity. Make sure it is not changed.
    static_assert(
        std::is_same_v<complex_t_imag, T>, "complex_t_imag must be T");
    return map([](T x) { return static_cast<T>(x.imag()); });
  }
  template <
      typename other_t_conj = T,
      typename std::enable_if_t<!c10::is_complex<other_t_conj>::value, int> = 0>
  Vectorized<T> conj() const {
    // other_t_conj is for SFINAE and clarity. Make sure it is not changed.
    static_assert(std::is_same_v<other_t_conj, T>, "other_t_conj must be T");
    return *this;
  }
  template <
      typename complex_t_conj = T,
      typename std::enable_if_t<c10::is_complex<complex_t_conj>::value, int> =
          0>
  Vectorized<T> conj() const {
    // complex_t_conj is for SFINAE and clarity. Make sure it is not changed.
    static_assert(
        std::is_same_v<complex_t_conj, T>, "complex_t_conj must be T");
    return map([](T x) { return static_cast<T>(std::conj(x)); });
  }
  Vectorized<T> acos() const {
    return map(std::acos);
  }
  Vectorized<T> acosh() const {
    return map(std::acosh);
  }
  Vectorized<T> asin() const {
    return map(std::asin);
  }
  Vectorized<T> asinh() const {
    return map(std::asinh);
  }
  Vectorized<T> atan() const {
    return map(std::atan);
  }
  Vectorized<T> atanh() const {
    return map(std::atanh);
  }
  Vectorized<T> atan2(const Vectorized<T>& exp) const {
    Vectorized<T> ret;
    for (const auto i : c10::irange(size())) {
      ret[i] = std::atan2(values[i], exp[i]);
    }
    return ret;
  }
  template <
      typename U = T,
      typename std::enable_if_t<is_floating_point_v<U>, int> = 0>
  Vectorized<T> copysign(const Vectorized<T>& sign) const {
    Vectorized<T> ret;
    for (size_type i = 0; i < size(); i++) {
      ret[i] = c10::copysign(values[i], sign[i]);
    }
    return ret;
  }
  Vectorized<T> erf() const {
    return map(std::erf);
  }
  Vectorized<T> erfc() const {
    return map(std::erfc);
  }
  Vectorized<T> erfinv() const {
    return map(calc_erfinv);
  }
  Vectorized<T> exp() const {
    return map(std::exp);
  }
  Vectorized<T> exp2() const {
    return map(exp2_impl);
  }
  Vectorized<T> expm1() const {
    return map(std::expm1);
  }
  Vectorized<T> exp_u20() const {
    return map(std::exp);
  }
  Vectorized<T> frac() const {
    return *this - this->trunc();
  }
  template <
      typename U = T,
      typename std::enable_if_t<is_floating_point_v<U>, int> = 0>
  Vectorized<T> fmod(const Vectorized<T>& q) const {
    // U is for SFINAE purposes only. Make sure it is not changed.
    static_assert(std::is_same_v<U, T>, "U must be T");
    Vectorized<T> ret;
    for (const auto i : c10::irange(size())) {
      ret[i] = std::fmod(values[i], q[i]);
    }
    return ret;
  }
  Vectorized<T> log() const {
    return map(std::log);
  }
  Vectorized<T> log10() const {
    return map(std::log10);
  }
  Vectorized<T> log1p() const {
    return map(std::log1p);
  }
  template <
      typename other_t_log2 = T,
      typename std::enable_if_t<!c10::is_complex<other_t_log2>::value, int> = 0>
  Vectorized<T> log2() const {
    // other_t_log2 is for SFINAE and clarity. Make sure it is not changed.
    static_assert(std::is_same_v<other_t_log2, T>, "other_t_log2 must be T");
    return map(std::log2);
  }
  template <
      typename complex_t_log2 = T,
      typename std::enable_if_t<c10::is_complex<complex_t_log2>::value, int> =
          0>
  Vectorized<T> log2() const {
    // complex_t_log2 is for SFINAE and clarity. Make sure it is not changed.
    static_assert(
        std::is_same_v<complex_t_log2, T>, "complex_t_log2 must be T");
    const T log_2 = T(std::log(2.0));
    return Vectorized(map(std::log)) / Vectorized(log_2);
  }
  Vectorized<T> ceil() const {
    return map(at::native::ceil_impl);
  }
  Vectorized<T> cos() const {
    return map(std::cos);
  }
  Vectorized<T> cosh() const {
    return map(std::cosh);
  }
  Vectorized<T> floor() const {
    return map(at::native::floor_impl);
  }
  Vectorized<T> hypot(const Vectorized<T>& b) const {
    Vectorized<T> ret;
    for (const auto i : c10::irange(size())) {
      ret[i] = std::hypot(values[i], b[i]);
    }
    return ret;
  }
  Vectorized<T> i0() const {
    return map(calc_i0);
  }
  Vectorized<T> i0e() const {
    return map(calc_i0e);
  }
  Vectorized<T> digamma() const {
    return map(calc_digamma);
  }
  Vectorized<T> igamma(const Vectorized<T>& x) const {
    Vectorized<T> ret;
    for (const auto i : c10::irange(size())) {
      ret[i] = calc_igamma(values[i], x[i]);
    }
    return ret;
  }
  Vectorized<T> igammac(const Vectorized<T>& x) const {
    Vectorized<T> ret;
    for (const auto i : c10::irange(size())) {
      ret[i] = calc_igammac(values[i], x[i]);
    }
    return ret;
  }
  Vectorized<T> neg() const {
    // NB: the trailing return type is needed because we need to coerce the
    // return value back to T in the case of unary operator- incuring a
    // promotion
    return map([](T x) -> T { return -x; });
  }
  Vectorized<T> nextafter(const Vectorized<T>& b) const {
    Vectorized<T> ret;
    for (const auto i : c10::irange(size())) {
      ret[i] = std::nextafter(values[i], b[i]);
    }
    return ret;
  }
  Vectorized<T> round() const {
    // We do not use std::round because we would like to round midway numbers to
    // the nearest even integer.
    return map(at::native::round_impl);
  }
  Vectorized<T> sin() const {
    return map(std::sin);
  }
  Vectorized<T> sinh() const {
    return map(std::sinh);
  }
  Vectorized<T> tan() const {
    return map(std::tan);
  }
  Vectorized<T> tanh() const {
    return map(std::tanh);
  }
  Vectorized<T> trunc() const {
    return map(at::native::trunc_impl);
  }
  Vectorized<T> lgamma() const {
    return map(std::lgamma);
  }
  Vectorized<T> sqrt() const {
    return map(std::sqrt);
  }
  Vectorized<T> reciprocal() const {
    return map([](T x) { return (T)(1) / x; });
  }
  Vectorized<T> rsqrt() const {
    return map([](T x) { return (T)1 / std::sqrt(x); });
  }
  Vectorized<T> pow(const Vectorized<T>& exp) const {
    Vectorized<T> ret;
    for (const auto i : c10::irange(size())) {
      ret[i] = std::pow(values[i], exp[i]);
    }
    return ret;
  }
  T reduce_add() const {
    return reduce([](T x, T y) -> T { return x + y; });
  }
  T reduce_max() const {
    return reduce(std::max);
  }

 private:
  template <typename Op>
  inline Vectorized<T> binary_pred(const Vectorized<T>& other, Op op) const {
    // All bits are set to 1 if the pred is true, otherwise 0.
    Vectorized<T> vector;
    for (int64_t i = 0; i != size(); i++) {
      if (op(values[i], other.values[i])) {
        std::memset(static_cast<void*>(vector.values + i), 0xFF, sizeof(T));
      } else {
        std::memset(static_cast<void*>(vector.values + i), 0, sizeof(T));
      }
    }
    return vector;
  }

 public:
  Vectorized<T> operator==(const Vectorized<T>& other) const {
    return binary_pred(other, std::equal_to<T>());
  }
  Vectorized<T> operator!=(const Vectorized<T>& other) const {
    return binary_pred(other, std::not_equal_to<T>());
  }
  Vectorized<T> operator>=(const Vectorized<T>& other) const {
    return binary_pred(other, std::greater_equal<T>());
  }
  Vectorized<T> operator<=(const Vectorized<T>& other) const {
    return binary_pred(other, std::less_equal<T>());
  }
  Vectorized<T> operator>(const Vectorized<T>& other) const {
    return binary_pred(other, std::greater<T>());
  }
  Vectorized<T> operator<(const Vectorized<T>& other) const {
    return binary_pred(other, std::less<T>());
  }

 private:
  template <typename Op>
  inline Vectorized<T> binary_pred_bool(const Vectorized<T>& other, Op op)
      const {
    // 1 if the pred is true, otherwise 0.
    Vectorized<T> vector;
    for (int i = 0; i != size(); ++i) {
      vector[i] = static_cast<T>(op(values[i], other.values[i]));
    }
    return vector;
  }

 public:
  Vectorized<T> eq(const Vectorized<T>& other) const {
    return binary_pred_bool(other, std::equal_to<T>());
  }
  Vectorized<T> ne(const Vectorized<T>& other) const {
    return binary_pred_bool(other, std::not_equal_to<T>());
  }
  Vectorized<T> gt(const Vectorized<T>& other) const {
    return binary_pred_bool(other, std::greater<T>());
  }
  Vectorized<T> ge(const Vectorized<T>& other) const {
    return binary_pred_bool(other, std::greater_equal<T>());
  }
  Vectorized<T> lt(const Vectorized<T>& other) const {
    return binary_pred_bool(other, std::less<T>());
  }
  Vectorized<T> le(const Vectorized<T>& other) const {
    return binary_pred_bool(other, std::less_equal<T>());
  }
};

template <class T>
Vectorized<T> inline operator-(const Vectorized<T>& a) {
  return a.neg();
}

// There is an implicit conversion that would make this work if
// these operators weren't template functions, but they are template
// functions (and can't be moved to be non-member friends defined in
// the class body as suggested in
// https://stackoverflow.com/questions/9787593/implicit-type-conversion-with-template/9788255#9788255
// because we have a lot of disparate specializations of
// Vectorized). So, just explicitly make scalars work.
#define VECTORIZED_SUPPORT_SCALARS_FOR_BINARY_FUNC(name)   \
  template <class T>                                       \
  Vectorized<T> inline name(const Vectorized<T>& a, T b) { \
    return name(a, Vectorized<T>(b));                      \
  }                                                        \
  template <class T>                                       \
  Vectorized<T> inline name(T a, const Vectorized<T>& b) { \
    return name(Vectorized<T>(a), b);                      \
  }
#define VECTORIZED_SUPPORT_SCALARS_FOR_BINARY_OP(op) \
  VECTORIZED_SUPPORT_SCALARS_FOR_BINARY_FUNC(operator op)

template <class T>
Vectorized<T> inline operator+(const Vectorized<T>& a, const Vectorized<T>& b) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] + b[i];
  }
  return c;
}

VECTORIZED_SUPPORT_SCALARS_FOR_BINARY_OP(+)

template <class T>
Vectorized<T> inline operator-(const Vectorized<T>& a, const Vectorized<T>& b) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] - b[i];
  }
  return c;
}

VECTORIZED_SUPPORT_SCALARS_FOR_BINARY_OP(-)

template <class T>
Vectorized<T> inline operator*(const Vectorized<T>& a, const Vectorized<T>& b) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] * b[i];
  }
  return c;
}

VECTORIZED_SUPPORT_SCALARS_FOR_BINARY_OP(*)

template <class T>
Vectorized<T> inline operator/(const Vectorized<T>& a, const Vectorized<T>& b)
    __ubsan_ignore_float_divide_by_zero__ {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] / b[i];
  }
  return c;
}

VECTORIZED_SUPPORT_SCALARS_FOR_BINARY_OP(/)

template <class T, typename std::enable_if_t<!is_floating_point_v<T>, int> = 0>
Vectorized<T> inline operator%(const Vectorized<T>& a, const Vectorized<T>& b)
    __ubsan_ignore_float_divide_by_zero__ {
  return a - a / b * b;
}

VECTORIZED_SUPPORT_SCALARS_FOR_BINARY_OP(%)

template <class T>
Vectorized<T> inline operator||(
    const Vectorized<T>& a,
    const Vectorized<T>& b) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] || b[i];
  }
  return c;
}

VECTORIZED_SUPPORT_SCALARS_FOR_BINARY_OP(||)

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <
    class T,
    typename std::enable_if_t<!c10::is_complex<T>::value, int> = 0>
Vectorized<T> inline maximum(const Vectorized<T>& a, const Vectorized<T>& b) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = (a[i] > b[i]) ? a[i] : b[i];
    if (_isnan(a[i])) {
      // If either input is NaN, propagate a NaN.
      // NOTE: The case where b[i] was NaN is handled correctly by the naive
      // ternary operator above.
      c[i] = a[i];
    }
  }
  return c;
}

template <
    class T,
    typename std::enable_if_t<c10::is_complex<T>::value, int> = 0>
Vectorized<T> inline maximum(const Vectorized<T>& a, const Vectorized<T>& b) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = (std::abs(a[i]) > std::abs(b[i])) ? a[i] : b[i];
    if (_isnan(a[i])) {
      // If either input is NaN, propagate a NaN.
      // NOTE: The case where b[i] was NaN is handled correctly by the naive
      // ternary operator above.
      c[i] = a[i];
    }
  }
  return c;
}

VECTORIZED_SUPPORT_SCALARS_FOR_BINARY_FUNC(maximum)

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <
    class T,
    typename std::enable_if_t<!c10::is_complex<T>::value, int> = 0>
Vectorized<T> inline minimum(const Vectorized<T>& a, const Vectorized<T>& b) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = (a[i] < b[i]) ? a[i] : b[i];
    if (_isnan(a[i])) {
      // If either input is NaN, propagate a NaN.
      // NOTE: The case where b[i] was NaN is handled correctly by the naive
      // ternary operator above.
      c[i] = a[i];
    }
  }
  return c;
}

template <
    class T,
    typename std::enable_if_t<c10::is_complex<T>::value, int> = 0>
Vectorized<T> inline minimum(const Vectorized<T>& a, const Vectorized<T>& b) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = (std::abs(a[i]) < std::abs(b[i])) ? a[i] : b[i];
    if (_isnan(a[i])) {
      // If either input is NaN, propagate a NaN.
      // NOTE: The case where b[i] was NaN is handled correctly by the naive
      // ternary operator above.
      c[i] = a[i];
    }
  }
  return c;
}

VECTORIZED_SUPPORT_SCALARS_FOR_BINARY_FUNC(minimum)

template <
    class T,
    typename std::enable_if_t<!c10::is_complex<T>::value, int> = 0>
Vectorized<T> inline clamp(
    const Vectorized<T>& a,
    const Vectorized<T>& min_vec,
    const Vectorized<T>& max_vec) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = std::min(std::max(a[i], min_vec[i]), max_vec[i]);
  }
  return c;
}

#define VECTORIZED_SUPPORT_SCALARS_FOR_TERNARY_FUNC(name)       \
  template <class T>                                            \
  Vectorized<T> inline name(                                    \
      const Vectorized<T>& a, const Vectorized<T>& b, T c) {    \
    return name(a, b, Vectorized<T>(c));                        \
  }                                                             \
                                                                \
  template <class T>                                            \
  Vectorized<T> inline name(                                    \
      const Vectorized<T>& a, T b, const Vectorized<T>& c) {    \
    return name(a, Vectorized<T>(b), c);                        \
  }                                                             \
                                                                \
  template <class T>                                            \
  Vectorized<T> inline name(const Vectorized<T>& a, T b, T c) { \
    return name(a, Vectorized<T>(b), Vectorized<T>(c));         \
  }                                                             \
                                                                \
  template <class T>                                            \
  Vectorized<T> inline name(                                    \
      T a, const Vectorized<T>& b, const Vectorized<T>& c) {    \
    return name(Vectorized<T>(a), b, c);                        \
  }                                                             \
                                                                \
  template <class T>                                            \
  Vectorized<T> inline name(T a, const Vectorized<T>& b, T c) { \
    return name(Vectorized<T>(a), b, Vectorized<T>(c));         \
  }                                                             \
                                                                \
  template <class T>                                            \
  Vectorized<T> inline name(T a, T b, const Vectorized<T>& c) { \
    return name(Vectorized<T>(a), Vectorized<T>(b), c);         \
  }

VECTORIZED_SUPPORT_SCALARS_FOR_TERNARY_FUNC(clamp)

template <
    class T,
    typename std::enable_if_t<!c10::is_complex<T>::value, int> = 0>
Vectorized<T> inline clamp_max(
    const Vectorized<T>& a,
    const Vectorized<T>& max_vec) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] > max_vec[i] ? max_vec[i] : a[i];
  }
  return c;
}

VECTORIZED_SUPPORT_SCALARS_FOR_BINARY_FUNC(clamp_max)

template <
    class T,
    typename std::enable_if_t<!c10::is_complex<T>::value, int> = 0>
Vectorized<T> inline clamp_min(
    const Vectorized<T>& a,
    const Vectorized<T>& min_vec) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] < min_vec[i] ? min_vec[i] : a[i];
  }
  return c;
}

VECTORIZED_SUPPORT_SCALARS_FOR_BINARY_FUNC(clamp_min)

struct Vectorizedi;

#if defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)
template <class T, typename Op>
static inline Vectorized<T> bitwise_binary_op(
    const Vectorized<T>& a,
    const Vectorized<T>& b,
    Op op) {
  int_vector buffer;
#if defined(CPU_CAPABILITY_AVX2)
  int_vector a_buffer =
      _mm256_load_si256(reinterpret_cast<const int_vector*>((const T*)a));
  int_vector b_buffer =
      _mm256_load_si256(reinterpret_cast<const int_vector*>((const T*)b));
#elif defined(CPU_CAPABILITY_AVX512)
  int_vector a_buffer =
      _mm512_load_si512(reinterpret_cast<const int_vector*>((const T*)a));
  int_vector b_buffer =
      _mm512_load_si512(reinterpret_cast<const int_vector*>((const T*)b));
#endif
  buffer = op(a_buffer, b_buffer);
  __at_align__ T results[Vectorized<T>::size()];

#if defined(CPU_CAPABILITY_AVX2)
  _mm256_store_si256(reinterpret_cast<int_vector*>(results), buffer);
#elif defined(CPU_CAPABILITY_AVX512)
  _mm512_store_si512(reinterpret_cast<int_vector*>(results), buffer);
#endif
  return Vectorized<T>::loadu(results);
}

template <
    class T,
    typename std::enable_if_t<
        !std::is_base_of<Vectorizedi, Vectorized<T>>::value,
        int> = 0>
inline Vectorized<T> operator&(const Vectorized<T>& a, const Vectorized<T>& b) {
  // We enclose _mm512_and_si512 or _mm256_and_si256 with lambda because it is
  // always_inline
#if defined(CPU_CAPABILITY_AVX2)
  return bitwise_binary_op(
      a, b, [](int_vector a, int_vector b) { return _mm256_and_si256(a, b); });
#elif defined(CPU_CAPABILITY_AVX512)
  return bitwise_binary_op(
      a, b, [](int_vector a, int_vector b) { return _mm512_and_si512(a, b); });
#endif
}
template <
    class T,
    typename std::enable_if_t<
        !std::is_base_of<Vectorizedi, Vectorized<T>>::value,
        int> = 0>
inline Vectorized<T> operator|(const Vectorized<T>& a, const Vectorized<T>& b) {
  // We enclose _mm512_or_si512 or _mm256_or_si256 with lambda because it is
  // always_inline
#if defined(CPU_CAPABILITY_AVX2)
  return bitwise_binary_op(
      a, b, [](int_vector a, int_vector b) { return _mm256_or_si256(a, b); });
#elif defined(CPU_CAPABILITY_AVX512)
  return bitwise_binary_op(
      a, b, [](int_vector a, int_vector b) { return _mm512_or_si512(a, b); });
#endif
}
template <
    class T,
    typename std::enable_if_t<
        !std::is_base_of<Vectorizedi, Vectorized<T>>::value,
        int> = 0>
inline Vectorized<T> operator^(const Vectorized<T>& a, const Vectorized<T>& b) {
  // We enclose _mm512_xor_si512 or _mm256_xor_si256 with lambda because it is
  // always_inline
#if defined(CPU_CAPABILITY_AVX2)
  return bitwise_binary_op(
      a, b, [](int_vector a, int_vector b) { return _mm256_xor_si256(a, b); });
#elif defined(CPU_CAPABILITY_AVX512)
  return bitwise_binary_op(
      a, b, [](int_vector a, int_vector b) { return _mm512_xor_si512(a, b); });
#endif
}

#else

template <typename T>
auto load(char const* data) -> T {
  T ret;
  std::memcpy(&ret, data, sizeof(ret));
  return ret;
}

template <class T, typename Op>
static inline Vectorized<T> bitwise_binary_op(
    const Vectorized<T>& a,
    const Vectorized<T>& b,
    Op op) {
  static constexpr uint32_t element_no = VECTOR_WIDTH / sizeof(intmax_t);
  __at_align__ intmax_t buffer[element_no];
  static_assert(
      VECTOR_WIDTH % sizeof(intmax_t) == 0,
      "VECTOR_WIDTH not a multiple of sizeof(intmax_t)");
  static_assert(
      sizeof(buffer) == sizeof(Vectorized<T>),
      "sizeof(buffer) must match sizeof(Vectorized<T>)");
  // We should be using memcpy in order to respect the strict aliasing rule
  // see: https://github.com/pytorch/pytorch/issues/66119
  // Using char* is defined in the C11 standard 6.5 Expression paragraph 7
  // (http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)
  const auto* a_data = a.as_bytes();
  const auto* b_data = b.as_bytes();
  // load each intmax_t chunk and process; increase pointers by sizeof(intmax_t)
  for (auto& out : buffer) {
    out = op(load<intmax_t>(a_data), load<intmax_t>(b_data));
    a_data += sizeof(intmax_t);
    b_data += sizeof(intmax_t);
  }
  assert(a_data == a.as_bytes() + sizeof(a));
  assert(b_data == b.as_bytes() + sizeof(b));
  return Vectorized<T>::loadu(buffer);
}

template <
    class T,
    typename std::
        enable_if_t<!std::is_base_of_v<Vectorizedi, Vectorized<T>>, int> = 0>
inline Vectorized<T> operator&(const Vectorized<T>& a, const Vectorized<T>& b) {
  return bitwise_binary_op(a, b, std::bit_and<intmax_t>());
}
template <
    class T,
    typename std::
        enable_if_t<!std::is_base_of_v<Vectorizedi, Vectorized<T>>, int> = 0>
inline Vectorized<T> operator|(const Vectorized<T>& a, const Vectorized<T>& b) {
  return bitwise_binary_op(a, b, std::bit_or<intmax_t>());
}
template <
    class T,
    typename std::
        enable_if_t<!std::is_base_of_v<Vectorizedi, Vectorized<T>>, int> = 0>
inline Vectorized<T> operator^(const Vectorized<T>& a, const Vectorized<T>& b) {
  return bitwise_binary_op(a, b, std::bit_xor<intmax_t>());
}

#endif // defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)

VECTORIZED_SUPPORT_SCALARS_FOR_BINARY_OP(&)
VECTORIZED_SUPPORT_SCALARS_FOR_BINARY_OP(|)
VECTORIZED_SUPPORT_SCALARS_FOR_BINARY_OP(^)

template <
    class T,
    typename std::
        enable_if_t<!std::is_base_of_v<Vectorizedi, Vectorized<T>>, int> = 0>
inline Vectorized<T> operator~(const Vectorized<T>& a) {
  using int_t = int_same_size_t<T>;
  Vectorized<T> ones(c10::bit_cast<T>((int_t)(~(int_t)0))); // All bits are 1
  return a ^ ones;
}

template <class T>
Vectorized<T> inline operator<<(
    const Vectorized<T>& a,
    const Vectorized<T>& b) {
  constexpr T max_shift = sizeof(T) * CHAR_BIT;
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    T shift = b[i];
    if ((static_cast<std::make_signed_t<T>>(shift) < 0) ||
        (shift >= max_shift)) {
      c[i] = 0;
    } else {
      c[i] = static_cast<std::make_unsigned_t<T>>(a[i]) << shift;
    }
  }
  return c;
}

template <class T>
Vectorized<T> inline operator>>(
    const Vectorized<T>& a,
    const Vectorized<T>& b) {
  // right shift value to retain sign bit for signed and no bits for unsigned
  constexpr T max_shift = sizeof(T) * CHAR_BIT - std::is_signed_v<T>;
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    T shift = b[i];
    if ((static_cast<std::make_signed_t<T>>(shift) < 0) ||
        (shift >= max_shift)) {
      c[i] = a[i] >> max_shift;
    } else {
      c[i] = a[i] >> shift;
    }
  }
  return c;
}

template <typename T>
inline Vectorized<T>& operator+=(Vectorized<T>& a, const Vectorized<T>& b) {
  a = a + b;
  return a;
}
template <typename T>
inline Vectorized<T>& operator-=(Vectorized<T>& a, const Vectorized<T>& b) {
  a = a - b;
  return a;
}
template <typename T>
inline Vectorized<T>& operator/=(Vectorized<T>& a, const Vectorized<T>& b) {
  a = a / b;
  return a;
}
template <typename T>
inline Vectorized<T>& operator%=(Vectorized<T>& a, const Vectorized<T>& b) {
  a = a % b;
  return a;
}
template <typename T>
inline Vectorized<T>& operator*=(Vectorized<T>& a, const Vectorized<T>& b) {
  a = a * b;
  return a;
}

template <typename T>
inline Vectorized<T>& operator<<=(Vectorized<T>& a, const Vectorized<T>& b) {
  a = a << b;
  return a;
}

template <typename T>
inline Vectorized<T>& operator>>=(Vectorized<T>& a, const Vectorized<T>& b) {
  a = a >> b;
  return a;
}

template <typename T>
inline Vectorized<T> fmadd(
    const Vectorized<T>& a,
    const Vectorized<T>& b,
    const Vectorized<T>& c) {
  return a * b + c;
}

VECTORIZED_SUPPORT_SCALARS_FOR_TERNARY_FUNC(fmadd)

template <typename T>
inline Vectorized<T> fmsub(
    const Vectorized<T>& a,
    const Vectorized<T>& b,
    const Vectorized<T>& c) {
  return a * b - c;
}

VECTORIZED_SUPPORT_SCALARS_FOR_TERNARY_FUNC(fmsub)

template <typename T>
Vectorized<T> inline operator&&(
    const Vectorized<T>& a,
    const Vectorized<T>& b) {
  Vectorized<T> ret;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    ret[i] = a[i] && b[i];
  }
  return ret;
}

VECTORIZED_SUPPORT_SCALARS_FOR_BINARY_OP(&&)

template <int64_t scale = 1, typename T = void>
std::enable_if_t<
    scale == 1 || scale == 2 || scale == 4 || scale == 8,
    Vectorized<
        T>> inline gather(T const* base_addr, const Vectorized<int_same_size_t<T>>& vindex) {
  static constexpr int size = Vectorized<T>::size();
  int_same_size_t<T> index_arr[size];
  vindex.store(static_cast<void*>(index_arr));
  T buffer[size];
  for (const auto i : c10::irange(size)) {
    buffer[i] = base_addr[index_arr[i] * scale / sizeof(T)];
  }
  return Vectorized<T>::loadu(static_cast<void*>(buffer));
}

template <int64_t scale = 1, typename T = void>
std::
    enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<T>> inline mask_gather(
        const Vectorized<T>& src,
        T const* base_addr,
        const Vectorized<int_same_size_t<T>>& vindex,
        Vectorized<T>& mask) {
  static constexpr int size = Vectorized<T>::size();
  T src_arr[size];
  int_same_size_t<T> mask_arr[size]; // use int type so we can logical and
  int_same_size_t<T> index_arr[size];
  src.store(static_cast<void*>(src_arr));
  mask.store(static_cast<void*>(mask_arr));
  vindex.store(static_cast<void*>(index_arr));
  T buffer[size];
  for (const auto i : c10::irange(size)) {
    if (mask_arr[i] & 0x01) { // check highest bit
      buffer[i] = base_addr[index_arr[i] * scale / sizeof(T)];
    } else {
      buffer[i] = src_arr[i];
    }
  }
  mask = Vectorized<T>(static_cast<T>(0)); // "zero out" mask
  return Vectorized<T>::loadu(static_cast<void*>(buffer));
}

// Cast a given vector to another type without changing the bits representation.
// So a Vectorized<double> of 512 bits containing all ones can be cast to a
// Vectorized<int64_t> of 512 bits containing all ones (i.e., eight negative
// 1s). A Vec<double> of 256 bits containing all ones can be cast to a
// Vec<int64_t> of 256 bits containing all ones (i.e., four negative 1s).
// There is a struct here because we don't have static_if and I can't
// partially specialize a templated function.
template <typename dst_t, typename src_t>
struct CastImpl {
  static inline Vectorized<dst_t> apply(const Vectorized<src_t>& src) {
    src_t src_arr[Vectorized<src_t>::size()];
    src.store(static_cast<void*>(src_arr));
    return Vectorized<dst_t>::loadu(static_cast<const void*>(src_arr));
  }
};

template <typename scalar_t>
struct CastImpl<scalar_t, scalar_t> {
  static inline Vectorized<scalar_t> apply(const Vectorized<scalar_t>& src) {
    return src;
  }
};

template <typename dst_t, typename src_t>
inline Vectorized<dst_t> cast(const Vectorized<src_t>& src) {
  return CastImpl<dst_t, src_t>::apply(src);
}

template <typename T, typename IntType = int_same_size_t<T>>
inline Vectorized<IntType> convert_to_int_of_same_size(
    const Vectorized<T>& src) {
  static_assert(sizeof(T) == sizeof(IntType));
  static constexpr int size = Vectorized<T>::size();

  std::array<T, size> src_arr = {};
  src.store(static_cast<void*>(src_arr.data()));
  std::array<IntType, size> buffer;
  std::transform(
      src_arr.cbegin(), src_arr.cend(), buffer.begin(), [](const T& x) {
        return static_cast<IntType>(x);
      });
  return Vectorized<IntType>::loadu(static_cast<const void*>(buffer.data()));
}

template <typename T, typename IntType = int_same_size_t<T>>
inline Vectorized<T> convert_to_fp_of_same_size(
    const Vectorized<IntType>& src) {
  static_assert(sizeof(T) == sizeof(IntType));
  static constexpr int size = Vectorized<T>::size();

  std::array<IntType, size> src_arr;
  src.store(static_cast<void*>(src_arr.data()));
  std::array<T, size> buffer;
  std::transform(
      src_arr.cbegin(), src_arr.cend(), buffer.begin(), [](const IntType& x) {
        return static_cast<T>(x);
      });
  return Vectorized<T>::loadu(static_cast<const void*>(buffer.data()));
}

// clang-format off
// Example inputs for AVX512:
// a   Vectorized<float>   = {a0, b0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, a6, b6, a7, b7}
// b   Vectorized<float>   = {a8, b8, a9, b9, a10, b10, a11, b11, a12, b12, a13, b13, a14, b14, a15, b15}
// returns:
//           Vectorized<float>   = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15}
//           Vectorized<float>   = {b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15}
// Example inputs for AVX2: a           Vectorized<float>   = {a0, b0, a1, b1, a2, b2, a3, b3}
//               b                      Vectorized<float>   = {a4, b4, a5, b5, a6, b6, a7, b7}
//       returns:                       Vectorized<float>   = {a0, a1, a2, a3, a4, a5, a6, a7}
//                                      Vectorized<float>   = {b0, b1, b2, b3, b4, b5, b6, b7}
// clang-format on
template <typename T>
inline std::enable_if_t<
    Vectorized<T>::size() % 2 == 0,
    std::pair<Vectorized<T>, Vectorized<T>>>
deinterleave2(const Vectorized<T>& a, const Vectorized<T>& b) {
  static constexpr int size = Vectorized<T>::size();
  static constexpr int half_size = size / 2;
  T a_arr[size];
  T b_arr[size];
  T buffer1[size];
  T buffer2[size];
  a.store(static_cast<void*>(a_arr));
  b.store(static_cast<void*>(b_arr));
  for (const auto i : c10::irange(half_size)) {
    buffer1[i] = a_arr[i * 2];
    buffer1[half_size + i] = b_arr[i * 2];
    buffer2[i] = a_arr[i * 2 + 1];
    buffer2[half_size + i] = b_arr[i * 2 + 1];
  }
  return std::make_pair(
      Vectorized<T>::loadu(static_cast<void*>(buffer1)),
      Vectorized<T>::loadu(static_cast<void*>(buffer2)));
}

VECTORIZED_SUPPORT_SCALARS_FOR_BINARY_FUNC(deinterleave2)

// clang-format off
// inverse operation of deinterleave2
// Example inputs for AVX512:
//  a       Vectorized<float>   = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15}
//  b       Vectorized<float>   = {b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15}
// returns, for AVX512:
//          Vectorized<float>   = {a0, b0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, a6, b6, a7, b7}
//          Vectorized<float>   = {a8, b8, a9, b9, a10, b10, a11, b11, a12, b12, a13, b13, a14, b14, a15, b15}
// Example inputs for AVX2 : a           Vectorized<float>   = {a0, a1, a2, a3, a4, a5, a6, a7}
//                   b                   Vectorized<float>   = {b0, b1, b2, b3, b4, b5, b6, b7}
//       returns:            Vectorized<float>   = {a0, b0, a1, b1, a2, b2, a3, b3}
//                           Vectorized<float>   = {a4, b4, a5, b5, a6, b6, a7, b7}
// clang-format on
template <typename T>
inline std::enable_if_t<
    Vectorized<T>::size() % 2 == 0,
    std::pair<Vectorized<T>, Vectorized<T>>>
interleave2(const Vectorized<T>& a, const Vectorized<T>& b) {
  static constexpr int size = Vectorized<T>::size();
  static constexpr int half_size = size / 2;
  T a_arr[size];
  T b_arr[size];
  T buffer1[size];
  T buffer2[size];
  a.store(static_cast<void*>(a_arr));
  b.store(static_cast<void*>(b_arr));
  for (const auto i : c10::irange(half_size)) {
    buffer1[i * 2] = a_arr[i];
    buffer1[i * 2 + 1] = b_arr[i];
    buffer2[i * 2] = a_arr[half_size + i];
    buffer2[i * 2 + 1] = b_arr[half_size + i];
  }
  return std::make_pair(
      Vectorized<T>::loadu(static_cast<void*>(buffer1)),
      Vectorized<T>::loadu(static_cast<void*>(buffer2)));
}

VECTORIZED_SUPPORT_SCALARS_FOR_BINARY_FUNC(interleave2)

#undef VECTORIZED_SUPPORT_SCALARS_FOR_BINARY_FUNC
#undef VECTORIZED_SUPPORT_SCALARS_FOR_BINARY_OP
#undef VECTORIZED_SUPPORT_SCALARS_FOR_TERNARY_FUNC

template <typename src_T, typename dst_T>
inline void convert(const src_T* src, dst_T* dst, int64_t n) {
#ifndef _MSC_VER
#pragma unroll
#endif
  for ([[maybe_unused]] const auto i : c10::irange(n)) {
    *dst = c10::convert<dst_T>(c10::load(src));
    src++;
    dst++;
  }
}

template <typename T>
inline Vectorized<T> flip(const Vectorized<T>& data) {
  static constexpr int size = Vectorized<T>::size();
  T output[size];
  T buffer[size];
  data.store(static_cast<void*>(buffer));
  for (const auto i : c10::irange(size)) {
    output[i] = buffer[size - i - 1];
  }
  return Vectorized<T>::loadu(static_cast<void*>(output));
}

// Transpose the `src` buffer of type `T` and size (M,N) into the `dst` buffer.
// `ld_src` is the leading dimension of `src` and `ld_dst` is the leading
// dimension of `dst`.
template <typename T>
inline void transpose_mxn(
    const T* src,
    int64_t ld_src,
    T* dst,
    int64_t ld_dst,
    int M,
    int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      dst[j * ld_dst + i] = src[i * ld_src + j];
    }
  }
}

template <typename T, int M, int N>
inline void transpose_mxn(
    const T* src,
    int64_t ld_src,
    T* dst,
    int64_t ld_dst) {
  transpose_mxn<T>(src, ld_src, dst, ld_dst, M, N);
}

} // namespace CPU_CAPABILITY
} // namespace at::vec

// additional headers for more operations that depend on vec_base
#include <ATen/cpu/vec/vec_convert.h>
#include <ATen/cpu/vec/vec_mask.h>
#include <ATen/cpu/vec/vec_n.h>
