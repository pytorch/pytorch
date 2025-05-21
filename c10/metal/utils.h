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

#if __METAL_VERSION__ >= 310
template <>
struct vectypes<bfloat> {
  using type4 = bfloat4;
  using type3 = bfloat3;
  using type2 = bfloat2;
};
#endif

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

#if __METAL_VERSION__ >= 310
template <>
struct OpMathType<bfloat> {
  using type = float;
};
#endif
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

#if __METAL_VERSION__ >= 310
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
#endif

template <typename T>
using vec2type_t = typename detail::vectypes<T>::type2;

template <typename T>
using vec4type_t = typename detail::vectypes<T>::type4;

template <typename T>
using opmath_t = typename detail::OpMathType<T>::type;

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

// floor_divide
template <
    typename T,
    typename U,
    ::metal::enable_if_t<
        is_scalar_integral_v<T> && is_scalar_integral_v<U>,
        bool> = true>
inline decltype(T(0) + U(0)) floor_divide(T x, U y) {
  const auto quot = x / y;
  return (x < 0) == (y < 0) ? quot : (x % y != 0) ? quot - 1 : quot;
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
template <typename U, typename V>
using common_dtype = decltype(U(0) + V(0));

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

} // namespace metal
} // namespace c10
