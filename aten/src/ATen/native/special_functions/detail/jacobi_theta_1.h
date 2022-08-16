#pragma once

#include <c10/util/numbers.h>
#include <ATen/native/special_functions/detail/theta.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
std::complex<T1>
jacobi_theta_1(std::complex<T1> q, std::complex<T1> x) {
  using T2 = numeric_t<T1>;
  using T3 = std::complex<T2>;

  if (std::isnan(q) || std::isnan(x))
    return T1{std::numeric_limits<T1>::quiet_NaN()};
  else if (std::abs(q) >= T2(1))
    throw std::domain_error("jacobi_theta_1: nome q out of range");
  else if (std::abs(q) < T2{0.001L} || std::abs(q) > T2{0.95e-1L})
    return jacobi_theta_1_prod(q, x);
  else if (std::abs(x) < std::numeric_limits<T1>::epsilon())
    return std::complex<T1>{0, 0};
  else {
    const auto lattice = jacobi_lattice_t<T3, T3>(q);
    auto tau = lattice.tau().val;

    const auto x_red = lattice.reduce(x);
    auto fact = std::complex<T1>{1, 0};
    if (x_red.m != 0)
      fact *= parity<T1>(x_red.m);
    if (x_red.n != 0)
      fact *= parity<T1>(x_red.n)
          * std::exp(std::complex<T2>{0, 1} * T2(-2 * x_red.n) * x_red.z)
          * std::pow(q, -x_red.n * x_red.n);
    x = x_red.z;

    // theta_1(tau+1, z) = exp(i tau/4) elliptic_theta_1(tau, z)
    const auto itau = std::floor(std::real(tau));
    tau -= itau;
    fact *= polar_pi(T2(1), itau / T2(4));

    if (std::imag(tau) < 0.5) {
      tau = T2(-1) / tau;
      fact *= std::exp(std::complex<T2>{0, 1} * tau * x * x / c10::numbers::pi_v<T2>)
          / (std::complex<T2>{0, 1} * std::sqrt(-std::complex<T2>{0, 1} * tau));
      q = std::exp(std::complex<T2>{0, 1} * c10::numbers::pi_v<T2> * tau);
      x *= tau;
    }

    return fact * jacobi_theta_1_sum(q, x);
  }
}

template<typename T1>
T1
jacobi_theta_1(T1 q, const T1 x) {
  using T2 = std::complex<T1>;

  if (std::abs(jacobi_theta_1(T2(q), T2(x))) > std::numeric_limits<T1>::epsilon()
      && std::abs(std::imag(jacobi_theta_1(T2(q), T2(x))))
          > std::numeric_limits<T1>::epsilon() * std::abs(jacobi_theta_1(T2(q), T2(x)))) {
    throw std::runtime_error("jacobi_theta_1: Unexpected large imaginary part");
  } else {
    return std::real(jacobi_theta_1(T2(q), T2(x)));
  }
}
}
}
}
}
