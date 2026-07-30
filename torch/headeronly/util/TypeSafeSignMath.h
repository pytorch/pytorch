#pragma once

#include <torch/headeronly/macros/Macros.h>
#include <limits>
#include <type_traits>

C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wstring-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wstring-conversion")
#endif
#if C10_CLANG_HAS_WARNING("-Wimplicit-int-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-int-float-conversion")
#endif

namespace c10 {

/// Returns true if x < 0
/// NOTE: Will fail on an unsigned custom type
///       For the most part it's possible to fix this if
///       the custom type has a constexpr constructor.
///       However, notably, c10::Half does not :-(
template <typename T>
inline constexpr bool is_negative(const T& x) {
  if constexpr (std::is_unsigned_v<T>) {
    // An unsigned value can never be less than zero.
    return false;
  } else {
    return x < T(0);
  }
}

/// Returns the sign of x as -1, 0, 1
/// NOTE: Will fail on an unsigned custom type
///       For the most part it's possible to fix this if
///       the custom type has a constexpr constructor.
///       However, notably, c10::Half does not :-(
template <typename T>
inline constexpr int signum(const T& x) {
  if constexpr (std::is_unsigned_v<T>) {
    return T(0) < x;
  } else {
    return (T(0) < x) - (x < T(0));
  }
}

/// Returns true if a and b are not both negative
template <typename T, typename U>
inline constexpr bool signs_differ(const T& a, const U& b) {
  return is_negative(a) != is_negative(b);
}

// Suppress sign compare warning when compiling with GCC
// as later does not account for short-circuit rule before
// raising the warning, see https://godbolt.org/z/Tr3Msnz99
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#endif

/// Returns true if x is greater than the greatest value of the type Limit
template <typename Limit, typename T>
inline constexpr bool greater_than_max(const T& x) {
  constexpr bool can_overflow =
      std::numeric_limits<T>::digits > std::numeric_limits<Limit>::digits;
  return can_overflow && x > std::numeric_limits<Limit>::max();
}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

/// Returns true if x is less than the lowest value of type Limit
/// NOTE: Will fail on an unsigned custom type
///       For the most part it's possible to fix this if
///       the custom type has a constexpr constructor.
///       However, notably, c10::Half does not :
template <typename Limit, typename T>
inline constexpr bool less_than_lowest(const T& x) {
  if constexpr (std::is_unsigned_v<T>) {
    // x is unsigned, so it can never be below the lowest value of any type.
    return false;
  } else if constexpr (std::is_unsigned_v<Limit>) {
    // Limit is unsigned, so its lowest value is zero.
    return x < T(0);
  } else {
    return x < std::numeric_limits<Limit>::lowest();
  }
}

} // namespace c10

C10_CLANG_DIAGNOSTIC_POP()

HIDDEN_NAMESPACE_BEGIN(torch, headeronly)
using c10::greater_than_max;
using c10::is_negative;
using c10::less_than_lowest;
using c10::signs_differ;
using c10::signum;
HIDDEN_NAMESPACE_END(torch, headeronly)
