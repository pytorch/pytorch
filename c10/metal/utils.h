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

template <typename T>
using vec2type_t = typename detail::vectypes<T>::type2;

template <typename T>
using vec4type_t = typename detail::vectypes<T>::type4;

} // namespace metal
} // namespace c10
