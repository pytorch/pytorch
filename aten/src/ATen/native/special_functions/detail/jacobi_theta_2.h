#pragma once

#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename Tp>
std::complex<Tp>
jacobi_theta_2(std::complex<Tp> q, std::complex<Tp> x) {
  using Real = numeric_t<Tp>;
  using Cmplx = std::complex<Real>;
  const auto s_NaN = std::numeric_limits<Tp>::quiet_NaN();
  const auto s_eps = std::numeric_limits<Tp>::epsilon();
  const auto s_pi = c10::numbers::pi_v<Real>;
  const auto s_i = std::complex<Real>{0, 1};
  constexpr auto s_q_min = Real{0.001L};
  constexpr auto s_q_max = Real{0.95e-1L};

  if (std::isnan(q) || std::isnan(x))
    return Tp{s_NaN};
  else if (std::abs(q) >= Real{1})
    throw std::domain_error("jacobi_theta_2: nome q out of range");
  else if (std::abs(q) < s_q_min || std::abs(q) > s_q_max)
    return jacobi_theta_2_prod(q, x);
  else if (std::abs(x) < s_eps)
    return jacobi_theta_0_t<Cmplx, Cmplx>(
        jacobi_lattice_t<Cmplx, Cmplx>(q)).th2;
  else {
    const auto lattice = jacobi_lattice_t<Cmplx, Cmplx>(q);
    auto tau = lattice.tau().val;

    const auto x_red = lattice.reduce(x);
    auto fact = std::complex<Tp>{1, 0};
    if (x_red.m != 0)
      fact *= parity<Tp>(x_red.m);
    if (x_red.n != 0)
      fact *= std::exp(s_i * Real(-2 * x_red.n) * x_red.z)
          * std::pow(q, -x_red.n * x_red.n);
    x = x_red.z;

    // theta_2(tau+1, z) = elliptic_theta_2(tau, z)
    const auto itau = std::floor(std::real(tau));
    tau -= itau;
    fact *= polar_pi(Real{1}, itau / Real{4});

    if (std::imag(tau) < 0.5) {
      const auto fact2 = std::sqrt(-s_i * tau);
      tau = Real{-1} / tau;
      const auto phase = std::exp(s_i * tau * x * x / s_pi);
      fact *= phase / fact2;
      q = std::exp(s_i * s_pi * tau);
      x *= tau;
      return fact * jacobi_theta_4_sum(q, x);
    } else
      return fact * jacobi_theta_2_sum(q, x);
  }
}

template<typename T1>
T1
jacobi_theta_2(T1 q, const T1 x) {
  using T2 = std::complex<T1>;

  if (std::abs(jacobi_theta_2(T2(q), T2(x))) > std::numeric_limits<T1>::epsilon()
      && std::abs(std::imag(jacobi_theta_2(T2(q), T2(x))))
          > std::numeric_limits<T1>::epsilon() * std::abs(jacobi_theta_2(T2(q), T2(x)))) {
    throw std::runtime_error("jacobi_theta_2: Unexpected large imaginary part");
  } else {
    return std::real(jacobi_theta_2(T2(q), T2(x)));
  }
}
}
}
}
}
