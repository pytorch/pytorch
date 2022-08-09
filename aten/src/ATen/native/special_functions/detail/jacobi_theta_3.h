#pragma once

#include <c10/util/numbers.h>

namespace at::native::special_functions::detail {
template<typename T1>
std::complex<T1>
jacobi_theta_3(std::complex<T1> q, std::complex<T1> x) {
  using Real = numeric_t<T1>;
  using Cmplx = std::complex<Real>;
  const auto s_NaN = std::numeric_limits<T1>::quiet_NaN();
  const auto s_eps = std::numeric_limits<T1>::epsilon();
  const auto s_pi = c10::numbers::pi_v<Real>;
  const auto s_i = std::complex<Real>{0, 1};
  constexpr auto s_q_min = Real{0.001L};
  constexpr auto s_q_max = Real{0.95e-1L};

  if (std::isnan(q) || std::isnan(x))
    return T1{s_NaN};
  else if (std::abs(q) >= Real{1})
    throw std::domain_error("jacobi_theta_3: nome q out of range");
  else if (std::abs(q) < s_q_min || std::abs(q) > s_q_max)
    return jacobi_theta_3_prod(q, x);
  else if (std::abs(x) < s_eps)
    return jacobi_theta_0_t<Cmplx, Cmplx>(
        jacobi_lattice_t<Cmplx, Cmplx>(q)).th3;
  else {
    const auto lattice = jacobi_lattice_t<Cmplx, Cmplx>(q);
    auto tau = lattice.tau().val;

    const auto x_red = lattice.reduce(x);
    auto fact = std::complex<T1>{1, 0};
    if (x_red.n != 0)
      fact *= std::exp(s_i * Real(-2 * x_red.n) * x_red.z)
          * std::pow(q, -x_red.n * x_red.n);
    x = x_red.z;

    // theta_3(tau+1, z) = elliptic_theta_3(tau, z)
    const auto itau = std::floor(std::real(tau));
    tau -= itau;

    if (std::imag(tau) < 0.5) {
      const auto fact2 = std::sqrt(-s_i * tau);
      tau = Real{-1} / tau;
      const auto phase = std::exp(s_i * tau * x * x / s_pi);
      fact *= phase / fact2;
      q = std::exp(s_i * s_pi * tau);
      x *= tau;
    }

    return fact * jacobi_theta_3_sum(q, x);
  }
}

template<typename Tp>
Tp
jacobi_theta_3(Tp q, const Tp x) {
  using Cmplx = std::complex<Tp>;
  const auto s_eps = std::numeric_limits<Tp>::epsilon();

  const auto ret = jacobi_theta_3(Cmplx(q), Cmplx(x));

  if (std::abs(ret) > s_eps
      && std::abs(std::imag(ret)) > s_eps * std::abs(ret))
    throw std::runtime_error("jacobi_theta_3: "
                             "Unexpected large imaginary part");
  else
    return std::real(ret);
}
}
