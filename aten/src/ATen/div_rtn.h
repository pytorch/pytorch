#pragma once

#include <type_traits>

// Integer division rounding to -Infinity
template <typename T>
static inline T div_rtn(T x, T y) {
  static_assert(
      std::is_integral_v<T>, "div_rtn is only valid for integral types");
  T q = x / y;
  T r = x % y;
  if ((r != 0) && ((r < 0) != (y < 0)))
    --q;
  return q;
}
