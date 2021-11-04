#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

namespace c10 {
// TODO: Replace me with inline constexpr variable when C++17 becomes available
namespace detail {
template <typename T>
C10_HOST_DEVICE inline constexpr T pi() {
  return static_cast<T>(3.14159265358979323846L);
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

// TODO: Replace me with std::numbers::pi when C++20 is there
template <typename T>
constexpr T pi = c10::detail::pi<T>();

} // namespace c10
