#pragma once

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
inline constexpr
T1
sin_pi(T1 x) {
  using T2 = numeric_t<T1>;

  if (std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (x < T1(0)) {
    return -sin_pi(-x);
  } else if (x < T1(0.5L)) {
    return std::sin(x * c10::pi<T2>);
  } else if (x < T1(1)) {
    return std::sin((T1(1) - x) * c10::pi<T2>);
  } else {
    const auto floor_x = std::floor(x);

    const auto x_negative_floor_x = x - floor_x;

    if ((int(floor_x) & 1) == 1) {
      if (x_negative_floor_x < T1(0.5L)) {
        return -1 * sin_pi(x_negative_floor_x);
      } else {
        return -1 * sin_pi(T1(1) - x_negative_floor_x);
      }
    } else {
      if (x_negative_floor_x < T1(0.5L)) {
        return +1 * sin_pi(x_negative_floor_x);
      } else {
        return +1 * sin_pi(T1(1) - x_negative_floor_x);
      }
    }
  }
} // T1 sin_pi(T1 x)

template<typename T1>
inline constexpr
std::complex<T1>
sin_pi(std::complex<T1> z) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  const auto pi_v = c10::pi<T3>;

  const auto real_z = std::real(z);
  const auto imag_z = std::imag(z);

  return sin_pi(real_z) * std::cosh(pi_v * imag_z) + std::complex<T1>{0, 1} * cos_pi(real_z) * std::sinh(pi_v * imag_z);
} // std::complex<T1> sin_pi(std::complex<T1> z)
} // namespace detail
} // namespace special_functions
} // namespace native
} // namespace at
