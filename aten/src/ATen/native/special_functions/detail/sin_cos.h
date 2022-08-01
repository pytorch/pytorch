#pragma once

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
inline constexpr
sin_cos_t<T1>
sin_cos(T1 x) {
  return sin_cos_t<Tp>{std::sin(x), std::cos(x)};
} // sin_cos_t<T1> sin_cos(T1 x)

inline constexpr
sin_cos_t<double>
sin_cos(double x) {
  return sin_cos<double>(x);
} // sin_cos_t<double> sin_cos(double x)
} // namespace detail
} // namespace special_functions
} // namespace native
} // namespace at
