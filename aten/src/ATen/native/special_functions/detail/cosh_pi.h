#pragma once

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
inline constexpr
T1
cosh_pi(T1 x) {
  using T2 = numeric_t<T1>;

  if (std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (x < T1{0}) {
    return cosh_pi(-x);
  } else {
    return std::cosh(c10::pi<T2> * x);
  }
} // T1 cosh_pi(T1 x)

template<typename T1>
inline constexpr
std::complex<T1>
cosh_pi(std::complex<T1> z) {
  using T2 = T1;
  using T3 = numeric_t<T2>;
  return std::cosh(c10::pi<T3> * std::real(z)) * cos_pi(std::imag(z))
      + std::complex<T1>{0, 1} * std::sinh(c10::pi<T3> * std::real(z)) * sin_pi(std::imag(z));
} // std::complex<T1> cosh_pi(std::complex<T1> z)
} // namespace detail
} // namespace special_functions
} // namespace native
} // namespace at
