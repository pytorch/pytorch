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
inline constexpr int64_t divup(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

} // namespace torch::headeronly
