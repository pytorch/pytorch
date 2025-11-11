// Metal helper functions
#pragma once
#include <c10/metal/common.h>
#include <metal_stdlib>

namespace c10 {
namespace metal {

namespace detail {
template <typename T>
struct vectypes {};

template <>
struct vectypes<float> {
  using type4 = float4;
  using type3 = float3;
  using type2 = float2;
};

template <>
struct vectypes<half> {
  using type4 = half4;
  using type3 = half3;
  using type2 = half2;
};

template <>
struct vectypes<bfloat> {
  using type4 = bfloat4;
  using type3 = bfloat3;
  using type2 = bfloat2;
};

template <>
struct vectypes<short> {
  using type4 = short4;
  using type3 = short3;
  using type2 = short2;
};

template <>
struct vectypes<int> {
  using type4 = int4;
  using type3 = int3;
  using type2 = int2;
};

template <>
struct vectypes<long> {
  using type4 = short4;
  using type3 = short3;
  using type2 = short2;
};

template <typename T>
struct OpMathType {
  using type = T;
};

template <>
struct OpMathType<half> {
  using type = float;
};

template <>
struct OpMathType<short> {
  using type = int;
};

template <>
struct OpMathType<char> {
  using type = int;
};

template <>
struct OpMathType<uchar> {
  using type = int;
};

template <>
struct OpMathType<bfloat> {
  using type = float;
};

// Type promotion structure for higher precision accumulation
template <typename T>
struct AccumulationType {
  using type = T;
};

// Specialization for half - promote to float for accumulation
template <>
struct AccumulationType<half> {
  using type = float;
};

// Specialization for bfloat - promote to float for accumulation
template <>
struct AccumulationType<bfloat> {
  using type = float;
};

} // namespace detail

template <typename T>
::metal::enable_if_t<::metal::is_floating_point_v<T>, T> max(T a, T b) {
  return ::metal::isunordered(a, b) ? NAN : ::metal::max(a, b);
}

template <typename T, typename U>
::metal::enable_if_t<::metal::is_integral_v<T>&& ::metal::is_integral_v<U>, T>
max(T a, U b) {
  return ::metal::max(a, static_cast<T>(b));
}

template <typename T>
::metal::enable_if_t<::metal::is_floating_point_v<T>, T> min(T a, T b) {
  return ::metal::isunordered(a, b) ? NAN : ::metal::min(a, b);
}

template <typename T, typename U>
::metal::enable_if_t<::metal::is_integral_v<T>&& ::metal::is_integral_v<U>, T>
min(T a, U b) {
  return ::metal::min(a, static_cast<T>(b));
}

template <>
inline bfloat min(bfloat a, bfloat b) {
  return bfloat(
      ::metal::isunordered(a, b) ? NAN : ::metal::min(float(a), float(b)));
}

template <>
inline bfloat max(bfloat a, bfloat b) {
  return bfloat(
      ::metal::isunordered(a, b) ? NAN : ::metal::max(float(a), float(b)));
}

template <typename T>
using vec2type_t = typename detail::vectypes<T>::type2;

template <typename T>
using vec4type_t = typename detail::vectypes<T>::type4;

template <typename T>
using opmath_t = typename detail::OpMathType<T>::type;

template <typename T>
using accum_t = typename detail::AccumulationType<T>::type;

// TODO: Move it to type_traits header may be
template <typename F, typename... Args>
using result_of = decltype(::metal::declval<F>()(::metal::declval<Args>()...));

template <typename T>
constexpr constant bool is_complex_v =
    ::metal::is_same_v<T, float2> || ::metal::is_same_v<T, half2>;

template <typename T>
constexpr constant bool is_scalar_floating_point_v =
    ::metal::is_floating_point_v<T> && ::metal::is_scalar_v<T>;

template <typename T>
constexpr constant bool is_scalar_integral_v =
    ::metal::is_integral_v<T> && ::metal::is_scalar_v<T>;

template <typename U, typename V>
using common_dtype = decltype(U(0) + V(0));

// floor_divide
template <
    typename T,
    typename U,
    ::metal::enable_if_t<
        is_scalar_integral_v<T> && is_scalar_integral_v<U>,
        bool> = true>
inline common_dtype<T, U> floor_divide(T x, U y) {
  const auto quot = x / y;
  return (x < 0) == (y < 0) ? quot : (x % y != 0) ? quot - 1 : quot;
}

template <
    typename T,
    typename U,
    ::metal::enable_if_t<
        is_scalar_floating_point_v<T> && is_scalar_floating_point_v<U>,
        bool> = true>
inline common_dtype<T, U> floor_divide(T x, U y) {
  return ::metal::floor(x / y);
}

// fmod
template <
    typename T,
    typename U,
    ::metal::enable_if_t<
        is_scalar_integral_v<T> && is_scalar_integral_v<U>,
        bool> = true>
inline common_dtype<T, U> fmod(T x, U y) {
  return x % y;
}

template <
    typename T,
    typename U,
    ::metal::enable_if_t<
        is_scalar_floating_point_v<T> && is_scalar_floating_point_v<U>,
        bool> = true>
inline common_dtype<T, U> fmod(T x, U y) {
  return ::metal::fmod(x, y);
}

// cast_to primitives
//  - No-op if types as the same
template <
    typename T,
    typename U,
    ::metal::enable_if_t<::metal::is_same_v<U, T>, bool> = true>
inline T cast_to(const U from) {
  return from;
}
//  - Simple cast between scalar and complex dtypes
template <
    typename T,
    typename U,
    ::metal::enable_if_t<
        !::metal::is_same_v<U, T> && (is_complex_v<T> == is_complex_v<U>),
        bool> = true>
inline T cast_to(const U from) {
  return static_cast<T>(from);
}

// - Scalar to complex
template <
    typename T,
    typename U,
    ::metal::enable_if_t<is_complex_v<T> && !is_complex_v<U>, bool> = true>
inline T cast_to(const U from) {
  return T(float(from), 0.0);
}
// - Complex to scalar (should not really be used, but exists for compliteness)
template <
    typename T,
    typename U,
    ::metal::enable_if_t<!is_complex_v<T> && is_complex_v<U>, bool> = true>
inline T cast_to(const U from) {
  return static_cast<T>(from.x);
}

// Generalizable math operators (used for both scalar and complex)

template <
    typename T,
    typename U,
    ::metal::enable_if_t<!is_complex_v<T>, bool> = true>
inline common_dtype<T, U> mul(const T x, const U y) {
  return x * y;
}

template <
    typename T,
    typename U,
    ::metal::enable_if_t<is_complex_v<T> && is_complex_v<U>, bool> = true>
inline common_dtype<T, U> mul(const T x, const U y) {
  return T(x.x * y.x - x.y * y.y, x.x * y.y + x.y * y.x);
}

template <
    typename T,
    typename U,
    ::metal::enable_if_t<!is_complex_v<T>, bool> = true>
inline common_dtype<T, U> div(const T x, const U y) {
  return x / y;
}

template <
    typename T,
    typename U,
    ::metal::enable_if_t<is_complex_v<T> && is_complex_v<U>, bool> = true>
inline common_dtype<T, U> div(const T x, const U y) {
  return T(::metal::dot(x, y), x.y * y.x - x.x * y.y) / ::metal::dot(y, y);
}

// Remainder operator
template <
    typename T,
    typename U,
    ::metal::enable_if_t<
        is_scalar_floating_point_v<T> || is_scalar_floating_point_v<U>,
        bool> = true>
inline float remainder(const T x, const U y) {
  const auto x_f = static_cast<float>(x);
  const auto y_f = static_cast<float>(y);
  return x_f - y_f * floor_divide(x_f, y_f);
}

template <
    typename T,
    typename U,
    ::metal::enable_if_t<
        is_scalar_integral_v<T> && is_scalar_integral_v<U>,
        bool> = true>
inline common_dtype<T, U> remainder(const T x, const U y) {
  auto rc = x % y;
  return rc == 0 || (x ^ y) > 0 ? rc : rc + y;
}

// Based on algorithm described in
// https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html#1202
inline float log1p(float x) {
  const auto xp1 = 1.0f + x;
  // First two elements of Taylor series for log(1+x) in Horner's form are:
  // log(1+x) = x * (1 - x * (.5 ...)), but if 1 + x == x, then it's just x
  if (xp1 == 1.0f) {
    return x;
  }
  auto rc = ::metal::precise::log(xp1);
  if (x > -.5 && x < .5) {
    // Order of operations is important here for higher precision
    rc *= x / (xp1 - 1.0f);
  }
  return rc;
}

// The function is ported from mlx
inline float2 log1p(float2 in) {
  float x = in.x;
  float y = in.y;
  float zabs = ::metal::precise::sqrt(x * x + y * y);
  float theta = ::metal::atan2(y, x + 1);
  if (zabs < 0.5f) {
    float r = x * (2 + x) + y * y;
    if (r == 0) { // handle underflow
      return {x, theta};
    }
    return {0.5f * log1p(r), theta};
  } else {
    auto z0 = ::metal::sqrt((x + 1) * (x + 1) + y * y);
    return {::metal::log(z0), theta};
  }
}

template <typename T1, typename T2 = T1>
struct pair {
  T1 first;
  T2 second;
};

template <typename T>
inline T conj(T a) {
  return a;
}

template <>
inline half2 conj(half2 a) {
  return half2(a.x, -a.y);
}

template <>
inline float2 conj(float2 a) {
  return float2(a.x, -a.y);
}

#define INSTANTIATE_FOR_ALL_TYPES(MACRO) \
  MACRO(float);                          \
  MACRO(half);                           \
  MACRO(bfloat);                         \
  MACRO(float2);                         \
  MACRO(long);                           \
  MACRO(char);                           \
  MACRO(uchar);                          \
  MACRO(short);                          \
  MACRO(int);

#define INSTANTIATE_FOR_FLOAT_TYPES(MACRO) \
  MACRO(float);                            \
  MACRO(half);                             \
  MACRO(bfloat);

} // namespace metal
} // namespace c10
