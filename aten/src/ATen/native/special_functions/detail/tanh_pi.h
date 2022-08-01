#pragma once

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
tanh_pi(T1 x) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  return std::tanh(c10::pi<T3> * x);
} // T1 tanh_pi(T1 x)

template<typename T1>
std::complex<T1>
tanh_pi(std::complex<T1> z) {
  using T2 = T1;
  using T3 = numeric_t<T2>;
  
  return (std::tanh(c10::pi<T3> * std::real(z)) + std::complex<T1>{0, 1} * tan_pi(std::imag(z))) / (T2{1} + std::complex<T1>{0, 1} * std::tanh(c10::pi<T3> * std::real(z)) * tan_pi(std::imag(z)));
} // std::complex<T1> tanh_pi(std::complex<T1> z)
} // namespace detail
} // namespace special_functions
} // namespace native
} // namespace at
