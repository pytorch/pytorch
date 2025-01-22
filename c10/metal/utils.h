#include <metal_stdlib>

namespace c10 {
namespace metal {

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

} // namespace metal
} // namespace c10
