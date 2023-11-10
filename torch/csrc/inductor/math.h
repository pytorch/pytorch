#pragma once

#include <c10/macros/Macros.h>
#include <cmath>
#include <cstdint>

// Functions in this file should be header-only as it is used in ABI-compatible
// mode.

namespace torch::inductor {

namespace detail {

inline bool signs_differ(int64_t a, int64_t b) {
  return a < 0L != b < 0L;
}

} // namespace detail

// Copy of aten::native::div_floor_integer but specialized for int64_t and
// uncoupled from c10.
inline int64_t div_floor_int64(int64_t a, int64_t b) {
  if (detail::signs_differ(a, b)) {
    // Subtracts one from the results of truncation division if the
    // divisor and dividend have different sign(bit)s and the remainder of
    // the division is nonzero
    const auto quot = a / b;
    const auto rem = a % b;
    return rem ? quot - 1 : quot;
  }
  return a / b;
}

// Copy of aten::native::div_floor_floating but specialized for double and
// uncoupled from c10 (aside from header-only macros).
inline double div_floor_double(double a, double b)
    __ubsan_ignore_float_divide_by_zero__ {
  if (C10_UNLIKELY(b == 0)) {
    // Divide by zero: return standard IEEE result
    return a / b;
  }

  auto mod = std::fmod(a, b);
  auto div = (a - mod) / b;
  if ((mod != 0) && (b < 0) != (mod < 0)) {
    div -= double(1);
  }

  double floordiv;
  if (div != 0) {
    floordiv = std::floor(div);
    if (div - floordiv > 0.5) {
      floordiv += 1.0;
    }
  } else {
    floordiv = std::copysign(double(0), a / b);
  }
  return floordiv;
}

} // namespace torch::inductor
