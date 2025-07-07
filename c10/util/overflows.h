#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/TypeSafeSignMath.h>
#include <c10/util/complex.h>

#include <cmath>
#include <limits>
#include <type_traits>

namespace c10 {
// In some versions of MSVC, there will be a compiler error when building.
// C4146: unary minus operator applied to unsigned type, result still unsigned
// C4804: unsafe use of type 'bool' in operation
// It can be addressed by disabling the following warning.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4146)
#pragma warning(disable : 4804)
#pragma warning(disable : 4018)
#endif

// The overflow checks may involve float to int conversion which may
// trigger precision loss warning. Re-enable the warning once the code
// is fixed. See T58053069.
C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wimplicit-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-float-conversion")
#endif

// bool can be converted to any type.
// Without specializing on bool, in pytorch_linux_trusty_py2_7_9_build:
// `error: comparison of constant '255' with boolean expression is always false`
// for `f > limit::max()` below
template <typename To, typename From>
std::enable_if_t<std::is_same_v<From, bool>, bool> overflows(
    From /*f*/,
    bool strict_unsigned [[maybe_unused]] = false) {
  return false;
}

// skip isnan and isinf check for integral types
template <typename To, typename From>
std::enable_if_t<std::is_integral_v<From> && !std::is_same_v<From, bool>, bool>
overflows(From f, bool strict_unsigned = false) {
  using limit = std::numeric_limits<typename scalar_value_type<To>::type>;
  if constexpr (!limit::is_signed && std::numeric_limits<From>::is_signed) {
    // allow for negative numbers to wrap using two's complement arithmetic.
    // For example, with uint8, this allows for `a - b` to be treated as
    // `a + 255 * b`.
    if (!strict_unsigned) {
      return greater_than_max<To>(f) ||
          (c10::is_negative(f) &&
           -static_cast<uint64_t>(f) > static_cast<uint64_t>(limit::max()));
    }
  }
  return c10::less_than_lowest<To>(f) || greater_than_max<To>(f);
}

template <typename To, typename From>
std::enable_if_t<std::is_floating_point_v<From>, bool> overflows(
    From f,
    bool strict_unsigned [[maybe_unused]] = false) {
  using limit = std::numeric_limits<typename scalar_value_type<To>::type>;
  if (limit::has_infinity && std::isinf(static_cast<double>(f))) {
    return false;
  }
  if (!limit::has_quiet_NaN && (f != f)) {
    return true;
  }
  return f < limit::lowest() || f > limit::max();
}

C10_CLANG_DIAGNOSTIC_POP()

#ifdef _MSC_VER
#pragma warning(pop)
#endif

template <typename To, typename From>
std::enable_if_t<is_complex<From>::value, bool> overflows(
    From f,
    bool strict_unsigned = false) {
  // casts from complex to real are considered to overflow if the
  // imaginary component is non-zero
  if (!is_complex<To>::value && f.imag() != 0) {
    return true;
  }
  // Check for overflow componentwise
  // (Technically, the imag overflow check is guaranteed to be false
  // when !is_complex<To>, but any optimizer worth its salt will be
  // able to figure it out.)
  return overflows<
             typename scalar_value_type<To>::type,
             typename From::value_type>(f.real(), strict_unsigned) ||
      overflows<
             typename scalar_value_type<To>::type,
             typename From::value_type>(f.imag(), strict_unsigned);
}
} // namespace c10
