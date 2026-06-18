#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include <numbers>

C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wimplicit-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-float-conversion")
#endif

namespace c10 {
namespace detail {
template <typename T>
C10_HOST_DEVICE inline constexpr T e() {
  return static_cast<T>(std::numbers::e);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T euler() {
  return static_cast<T>(std::numbers::egamma);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T frac_1_pi() {
  return static_cast<T>(std::numbers::inv_pi);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T frac_1_sqrt_pi() {
  return static_cast<T>(std::numbers::inv_sqrtpi);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T frac_sqrt_2() {
  // No std::numbers equivalent for 1/sqrt(2).
  return static_cast<T>(0.707106781186547524400844362104849);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T frac_sqrt_3() {
  return static_cast<T>(std::numbers::inv_sqrt3);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T golden_ratio() {
  return static_cast<T>(std::numbers::phi);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T ln_10() {
  return static_cast<T>(std::numbers::ln10);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T ln_2() {
  return static_cast<T>(std::numbers::ln2);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T log_10_e() {
  return static_cast<T>(std::numbers::log10e);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T log_2_e() {
  return static_cast<T>(std::numbers::log2e);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T pi() {
  return static_cast<T>(std::numbers::pi);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T sqrt_2() {
  return static_cast<T>(std::numbers::sqrt2);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T sqrt_3() {
  return static_cast<T>(std::numbers::sqrt3);
}

template <>
C10_HOST_DEVICE inline constexpr BFloat16 pi<BFloat16>() {
  // According to
  // https://en.wikipedia.org/wiki/Bfloat16_floating-point_format#Special_values
  // pi is encoded as 4049
  return BFloat16(0x4049, BFloat16::from_bits());
}

template <>
C10_HOST_DEVICE inline constexpr Half pi<Half>() {
  return Half(0x4248, Half::from_bits());
}
} // namespace detail

template <typename T>
constexpr T e = c10::detail::e<T>();

template <typename T>
constexpr T euler = c10::detail::euler<T>();

template <typename T>
constexpr T frac_1_pi = c10::detail::frac_1_pi<T>();

template <typename T>
constexpr T frac_1_sqrt_pi = c10::detail::frac_1_sqrt_pi<T>();

template <typename T>
constexpr T frac_sqrt_2 = c10::detail::frac_sqrt_2<T>();

template <typename T>
constexpr T frac_sqrt_3 = c10::detail::frac_sqrt_3<T>();

template <typename T>
constexpr T golden_ratio = c10::detail::golden_ratio<T>();

template <typename T>
constexpr T ln_10 = c10::detail::ln_10<T>();

template <typename T>
constexpr T ln_2 = c10::detail::ln_2<T>();

template <typename T>
constexpr T log_10_e = c10::detail::log_10_e<T>();

template <typename T>
constexpr T log_2_e = c10::detail::log_2_e<T>();

template <typename T>
constexpr T pi = c10::detail::pi<T>();

template <typename T>
constexpr T sqrt_2 = c10::detail::sqrt_2<T>();

template <typename T>
constexpr T sqrt_3 = c10::detail::sqrt_3<T>();
} // namespace c10

C10_CLANG_DIAGNOSTIC_POP()
