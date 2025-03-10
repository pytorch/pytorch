// Metal helper functions
#pragma once
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

template <typename T>
::metal::enable_if_t<::metal::is_integral_v<T>, T> max(T a, T b) {
  return ::metal::max(a, b);
}

template <typename T>
::metal::enable_if_t<::metal::is_floating_point_v<T>, T> min(T a, T b) {
  return ::metal::isunordered(a, b) ? NAN : ::metal::min(a, b);
}

template <typename T>
::metal::enable_if_t<::metal::is_integral_v<T>, T> min(T a, T b) {
  return ::metal::min(a, b);
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

} // namespace metal
} // namespace c10
