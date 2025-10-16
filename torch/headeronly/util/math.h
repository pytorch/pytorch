#pragma once

#include <torch/headeronly/macros/Macros.h>
#include <type_traits>

namespace torch::headeronly {

/**
 * Integer division with ceiling (rounding up).
 * Computes ceil(x / y) for integer types.
 *
 * @param x dividend
 * @param y divisor
 * @return ceiling of x/y
 */
template <typename T>
inline constexpr T divup(T x, T y) {
  static_assert(std::is_integral_v<T>, "divup requires integral types");
  return (x + y - 1) / y;
}

} // namespace torch::headeronly
