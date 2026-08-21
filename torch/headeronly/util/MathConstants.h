#pragma once

#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Half.h>

C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wimplicit-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-float-conversion")
#endif

HIDDEN_NAMESPACE_BEGIN(torch, headeronly)
// TODO: Replace me with inline constexpr variable when C++17 becomes available
namespace detail {
template <typename T>
C10_HOST_DEVICE inline constexpr T e() {
  return static_cast<T>(2.718281828459045235360287471352662);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T euler() {
  return static_cast<T>(0.577215664901532860606512090082402);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T frac_1_pi() {
  return static_cast<T>(0.318309886183790671537767526745028);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T frac_1_sqrt_pi() {
  return static_cast<T>(0.564189583547756286948079451560772);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T frac_sqrt_2() {
  return static_cast<T>(0.707106781186547524400844362104849);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T frac_sqrt_3() {
  return static_cast<T>(0.577350269189625764509148780501957);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T golden_ratio() {
  return static_cast<T>(1.618033988749894848204586834365638);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T ln_10() {
  return static_cast<T>(2.302585092994045684017991454684364);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T ln_2() {
  return static_cast<T>(0.693147180559945309417232121458176);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T log_10_e() {
  return static_cast<T>(0.434294481903251827651128918916605);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T log_2_e() {
  return static_cast<T>(1.442695040888963407359924681001892);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T pi() {
  return static_cast<T>(3.141592653589793238462643383279502);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T sqrt_2() {
  return static_cast<T>(1.414213562373095048801688724209698);
}

template <typename T>
C10_HOST_DEVICE inline constexpr T sqrt_3() {
  return static_cast<T>(1.732050807568877293527446341505872);
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
constexpr T e = torch::headeronly::detail::e<T>();

template <typename T>
constexpr T euler = torch::headeronly::detail::euler<T>();

template <typename T>
constexpr T frac_1_pi = torch::headeronly::detail::frac_1_pi<T>();

template <typename T>
constexpr T frac_1_sqrt_pi = torch::headeronly::detail::frac_1_sqrt_pi<T>();

template <typename T>
constexpr T frac_sqrt_2 = torch::headeronly::detail::frac_sqrt_2<T>();

template <typename T>
constexpr T frac_sqrt_3 = torch::headeronly::detail::frac_sqrt_3<T>();

template <typename T>
constexpr T golden_ratio = torch::headeronly::detail::golden_ratio<T>();

template <typename T>
constexpr T ln_10 = torch::headeronly::detail::ln_10<T>();

template <typename T>
constexpr T ln_2 = torch::headeronly::detail::ln_2<T>();

template <typename T>
constexpr T log_10_e = torch::headeronly::detail::log_10_e<T>();

template <typename T>
constexpr T log_2_e = torch::headeronly::detail::log_2_e<T>();

template <typename T>
constexpr T pi = torch::headeronly::detail::pi<T>();

template <typename T>
constexpr T sqrt_2 = torch::headeronly::detail::sqrt_2<T>();

template <typename T>
constexpr T sqrt_3 = torch::headeronly::detail::sqrt_3<T>();
HIDDEN_NAMESPACE_END(torch, headeronly)

namespace c10 {
using torch::headeronly::e;
using torch::headeronly::euler;
using torch::headeronly::frac_1_pi;
using torch::headeronly::frac_1_sqrt_pi;
using torch::headeronly::frac_sqrt_2;
using torch::headeronly::frac_sqrt_3;
using torch::headeronly::golden_ratio;
using torch::headeronly::ln_10;
using torch::headeronly::ln_2;
using torch::headeronly::log_10_e;
using torch::headeronly::log_2_e;
using torch::headeronly::pi;
using torch::headeronly::sqrt_2;
using torch::headeronly::sqrt_3;
namespace detail {
using torch::headeronly::detail::e;
using torch::headeronly::detail::euler;
using torch::headeronly::detail::frac_1_pi;
using torch::headeronly::detail::frac_1_sqrt_pi;
using torch::headeronly::detail::frac_sqrt_2;
using torch::headeronly::detail::frac_sqrt_3;
using torch::headeronly::detail::golden_ratio;
using torch::headeronly::detail::ln_10;
using torch::headeronly::detail::ln_2;
using torch::headeronly::detail::log_10_e;
using torch::headeronly::detail::log_2_e;
using torch::headeronly::detail::pi;
using torch::headeronly::detail::sqrt_2;
using torch::headeronly::detail::sqrt_3;
} // namespace detail
} // namespace c10

C10_CLANG_DIAGNOSTIC_POP()
