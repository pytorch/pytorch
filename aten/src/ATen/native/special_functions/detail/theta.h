#pragma once

#include <c10/util/numbers.h>

namespace at::native::special_functions {
template<typename T1>
inline constexpr detail::promote_t<T1>
complete_elliptic_integral_k(T1 k);
namespace detail {
template<typename T1>
struct jacobi_elliptic_t {
  T1 cn;
  T1 dn;
  T1 sn;

  constexpr T1 am() const noexcept { return std::asin(sn); }

  constexpr T1 nc() const noexcept { return T1(1) / cn; }
  constexpr T1 nd() const noexcept { return T1(1) / dn; }
  constexpr T1 ns() const noexcept { return T1(1) / sn; }

  constexpr T1 cd() const noexcept { return cn / dn; }
  constexpr T1 cs() const noexcept { return cn / sn; }
  constexpr T1 dc() const noexcept { return dn / cn; }
  constexpr T1 ds() const noexcept { return dn / sn; }
  constexpr T1 sc() const noexcept { return sn / cn; }
  constexpr T1 sd() const noexcept { return sn / dn; }

  constexpr T1 sn_derivative() const noexcept { return cn * dn; }
  constexpr T1 cn_derivative() const noexcept { return -sn * dn; }
};

template<typename Tp_Omega1, typename Tp_Omega3 = std::complex<Tp_Omega1>>
struct jacobi_lattice_t {
  static_assert(is_complex_v < Tp_Omega1 > || is_complex_v < Tp_Omega3 > , "One frequecy type must be complex.");
  using _Real_Omega1 = numeric_t<Tp_Omega1>;
  using _Real_Omega3 = numeric_t<Tp_Omega3>;
  using Real = promote_t<_Real_Omega1, _Real_Omega3>;
  using Cmplx = std::complex<Real>;
  using _Tp_Nome = std::conditional_t<is_complex_v < Tp_Omega1>
  && is_complex_v<Tp_Omega3>,
  Cmplx, Real>;

  struct tau_t {
    Cmplx val;

    explicit tau_t(Cmplx tau)
        : val(tau) {}
  };

  struct arg_t {
    int m;
    int n;
    Cmplx z;
  };

  /// Construct the lattice from two complex lattice frequencies.
  jacobi_lattice_t(const Tp_Omega1 &omega1,
                   const Tp_Omega3 &omega3)
      : m_omega_1(omega1),
        m_omega_3(omega3) {
    if (std::isnan(m_omega_1) || std::isnan(m_omega_3))
      throw std::domain_error("Invalid input");
    else if (std::imag(this->tau().val) <= Real{0})
      throw std::domain_error("jacobi_lattice_t: "
                              "Lattice parameter must have positive imaginary part.");
    else {
      auto det = std::real(m_omega_3) * std::imag(m_omega_1)
          - std::imag(m_omega_3) * std::real(m_omega_1);
      if (std::abs(det) == 0)
        throw std::domain_error("jacobi_lattice_t: "
                                "Lattice frequencies must be linearly independent.");
    }
  }

  /// Construct the lattice from a single complex lattice parameter
  /// or half period ratio.
  explicit jacobi_lattice_t(const tau_t &tau)
      : m_omega_1(2 * s_pi),
        m_omega_3(2 * s_pi) {
    if (std::isnan(tau.val))
      throw std::domain_error("Invalid input");
    else if (std::imag(tau.val) <= Real{0})
      throw std::domain_error("jacobi_lattice_t: Lattice parameter must have positive imaginary part.");
    else {
      if constexpr (is_complex_v < Tp_Omega3 >)
        m_omega_3 *= tau.val;
      else
        m_omega_1 *= tau.val;
    }
  }

  /// Construct the lattice from a single scalar elliptic nome.
  explicit jacobi_lattice_t(_Tp_Nome q)
      : jacobi_lattice_t(tau_t(Cmplx{0, -1} * std::log(q) / s_pi)) {
    if (std::abs(q) == Real{0})
      throw std::domain_error("jacobi_lattice_t: Nome must be nonzero.");
  }

  /// Return the acalar lattice parameter or half period ratio.
  tau_t
  tau() const { return tau_t(this->m_omega_3 / this->m_omega_1); }

  /// Return the first lattice frequency.
  Tp_Omega1
  omega_1() const { return this->m_omega_1; }

  /// Return the second lattice frequency.
  Cmplx
  omega_2() const { return -(Cmplx(this->m_omega_1) + Cmplx(this->m_omega_3)); }

  /// Return the third lattice frequency.
  Tp_Omega3
  omega_3() const { return this->m_omega_3; }

  _Tp_Nome
  ellnome() const;

  arg_t
  reduce(const Cmplx &z) const;

  static constexpr auto s_pi = c10::numbers::pi_v<Real>;
  Tp_Omega1 m_omega_1;
  Tp_Omega3 m_omega_3;
};

template<typename Tp_Omega1, typename Tp_Omega3>
typename jacobi_lattice_t<Tp_Omega1, Tp_Omega3>::_Tp_Nome
jacobi_lattice_t<Tp_Omega1, Tp_Omega3>::ellnome() const {
  const auto s_i = Cmplx{0, 1};
  const auto s_pi = c10::numbers::pi_v<Real>;
  if constexpr (is_complex_v < _Tp_Nome >)
    return std::exp(s_i * s_pi * this->tau().val);
  else
    return std::real(std::exp(s_i * s_pi * this->tau().val));
}

template<typename _Tp1, typename _Tp3>
typename jacobi_lattice_t<_Tp1, _Tp3>::arg_t
jacobi_lattice_t<_Tp1, _Tp3>::
reduce(const typename jacobi_lattice_t<_Tp1, _Tp3>::Cmplx &z) const {
  const auto s_pi = c10::numbers::pi_v<Real>;

  const auto tau = this->tau().val;
  const auto tau_r = std::real(tau);
  const auto tau_i = std::imag(tau);
  const auto z_r = std::real(z);
  const auto z_i = std::imag(z);

  // Solve z = (z_r, z_i) = pi a (1, 0) + pi b (tau_r, tau_i).
  const auto b = z_i / tau_i / s_pi;
  const int n = std::floor(b);
  const auto nu = b - n;
  const auto a = (z_r - b * tau_r * s_pi) / s_pi;
  const int m = std::floor(a);
  const auto mu = a - m;

  return {m, n,
          s_pi * Cmplx(mu + nu * tau_r, nu * tau_i)};
}

template<typename _Tp1, typename _Tp3 = std::complex<_Tp1>>
struct jacobi_theta_0_t {
  jacobi_theta_0_t(const jacobi_lattice_t<_Tp1, _Tp3> &lattice);

  using _Type = typename jacobi_lattice_t<_Tp1, _Tp3>::_Tp_Nome;
  using Real = numeric_t<_Type>;
  using Cmplx = std::complex<Real>;

  _Type th1p;
  _Type th1ppp;
  _Type th2;
  _Type th2pp;
  _Type th3;
  _Type th3pp;
  _Type th4;
  _Type th4pp;
  _Type eta_1;
  Cmplx eta_2;
  Cmplx eta_3;

  _Type
  dedekind_eta() const { return std::cbrt(th2 * th3 * th4 / _Type{2}); }
};

template<typename _Tp1, typename _Tp3>
jacobi_theta_0_t<_Tp1, _Tp3>::
jacobi_theta_0_t(const jacobi_lattice_t<_Tp1, _Tp3> &lattice) {
  constexpr std::size_t s_max_iter = 50;
  const auto q = lattice.ellnome();

  const auto s_eps = std::numeric_limits<_Tp1>::epsilon();

  const auto fact = Real{2} * std::pow(q, Real{0.25L});
  this->th1p = fact;
  this->th2 = fact;
  this->th3 = Real{1};
  this->th4 = Real{1};
  this->th1ppp = Real{0};
  this->th2pp = Real{0};
  this->th3pp = Real{0};
  this->th4pp = Real{0};
  auto q2n = _Type{1};
  for (std::size_t n = 1; n < s_max_iter; ++n) {
    q2n *= q;
    const auto tp = Real{1} + q2n;
    this->th3 *= tp * tp;
    const auto tm = Real{1} - q2n;
    this->th4 *= tm * tm;

    this->th3pp += q2n / tp / tp;
    this->th4pp += q2n / tm / tm;

    q2n *= q;
    const auto tm2 = Real{1} - q2n;
    this->th3 *= tm2;
    this->th4 *= tm2;
    this->th2 *= tm2;
    this->th1p *= tm2 * tm2 * tm2;
    const auto tp2 = Real{1} + q2n;
    this->th2 *= tp2 * tp2;

    this->th1ppp += q2n / tm2 / tm2;
    this->th2pp += q2n / tp2 / tp2;

    if (std::abs(q2n) < s_eps)
      break;
  }
  // Could check th1p =? th2pp * th3pp * th4pp at this point.
  // Could check th1ppp =? th2pp + th3pp + th4pp at this point.
  this->th1ppp = (Real{-1} + Real{24} * this->th1ppp) * this->th1p;
  this->th2pp = (Real{-1} - Real{8} * this->th2pp) * this->th2;
  this->th3pp = Real{-8} * this->th3;
  this->th4pp = Real{8} * this->th4;

  const auto s_pi = c10::numbers::pi_v<Real>;
  this->eta_1 = -s_pi * s_pi * this->th1ppp
      / _Type{12} / lattice.omega_1() / this->th1p;
  const auto s_i = Cmplx{0, 1};
  this->eta_2 = (lattice.omega_2() * this->eta_1
      + s_i * s_pi / Real{2}) / lattice.omega_1();
  this->eta_3 = (lattice.omega_3() * this->eta_1
      - s_i * s_pi / Real{2}) / lattice.omega_1();
}

template<typename _Tp1, typename _Tp3 = std::complex<_Tp1>>
struct weierstrass_roots_t {
  using _Type = typename jacobi_lattice_t<_Tp1, _Tp3>::_Tp_Nome;
  using Real = numeric_t<_Type>;
  using Cmplx = std::complex<Real>;

  _Type e1, e2, e3;

  explicit
  weierstrass_roots_t(const jacobi_lattice_t<_Tp1, _Tp3> &lattice);

  weierstrass_roots_t(const jacobi_theta_0_t<_Tp1, _Tp3> &theta0,
                      _Tp1 omega1);

  /// Return the discriminant
  /// @f$ \Delta = 16(e_2 - e_3)^2(e_3 - e_1)^2(e_1 - e_2)^2 @f$.
  _Type
  delta() const {
    const auto del1 = e2 - e3;
    const auto del2 = e3 - e1;
    const auto del3 = e1 - e2;
    const auto del = del1 * del2 * del3;
    return _Type{16} * del * del;
  }
};

template<typename _Tp1, typename _Tp3>
weierstrass_roots_t<_Tp1, _Tp3>::
weierstrass_roots_t(const jacobi_lattice_t<_Tp1, _Tp3> &lattice)
#if cplusplus > 201403L
: weierstrass_roots_t(jacobi_theta_0_t(lattice),
                lattice.omega_1())
#else
    : weierstrass_roots_t(jacobi_theta_0_t<_Tp1, _Tp3>(lattice),
                          lattice.omega_1())
#endif
{}

template<typename _Tp1, typename _Tp3>
weierstrass_roots_t<_Tp1, _Tp3>::
weierstrass_roots_t(const jacobi_theta_0_t<_Tp1, _Tp3> &theta0,
                    _Tp1 omega_1) {
  const auto s_pi = c10::numbers::pi_v<Real>;

  const auto th22 = theta0.th2 * theta0.th2;
  const auto th24 = th22 * th22;
  const auto th42 = theta0.th4 * theta0.th4;
  const auto th44 = th42 * th42;
  const auto fr = s_pi / omega_1;
  const auto fc = fr * fr / Real{12};

  e1 = fc * (th24 + Real{2} * th44);
  e2 = fc * (th24 - th44);
  e3 = fc * (Real{-2} * th24 - th44);
}

template<typename _Tp1, typename _Tp3>
struct weierstrass_invariants_t {
  using _Type = typename jacobi_lattice_t<_Tp1, _Tp3>::_Tp_Nome;
  using Real = numeric_t<_Type>;
  using Cmplx = std::complex<Real>;

  _Type g_2, g_3;

  weierstrass_invariants_t(const jacobi_lattice_t<_Tp1, _Tp3> &);

  /// Return the discriminant @f$ \Delta = g_2^3 - 27 g_3^2 @f$.
  _Type
  delta() const {
    const auto g_2p3 = g_2 * g_2 * g_2;
    return g_2p3 - _Type{27} * g_3 * g_3;
  }

  /// Return Klein's invariant @f$ J = 1738 g_2^3 / (g_2^3 - 27 g_3^2) @f$.
  _Type
  klein_j() const {
    const auto g_2p3 = g_2 * g_2 * g_2;
    return _Type{1738} * g_2p3 / (g_2p3 - _Type{27} * g_3 * g_3);
  }
};

template<typename _Tp1, typename _Tp3>
weierstrass_invariants_t<_Tp1, _Tp3>::
weierstrass_invariants_t(const jacobi_lattice_t<_Tp1, _Tp3> &lattice) {
  const auto roots = weierstrass_roots_t<_Tp1, _Tp3>(lattice);
  g_2 = Real{2} * (roots.e1 * roots.e1
      + roots.e2 * roots.e2
      + roots.e3 * roots.e3);
  g_3 = Real{4} * roots.e1 * roots.e2 * roots.e3;
}

template<typename Tp>
Tp
jacobi_theta_1_sum(Tp q, Tp x) {
  using Real = numeric_t<Tp>;
  const auto s_eps = std::numeric_limits<Tp>::epsilon();
  constexpr std::size_t s_max_iter = 50;

  Tp sum{};
  Real sign{-1};
  for (std::size_t n = 0; n < s_max_iter; ++n) {
    sign *= -1;
    const auto term = sign
        * std::pow(q, Real((n + 0.5L) * (n + 0.5L)))
        * std::sin(Real(2 * n + 1) * x);
    sum += term;
    if (std::abs(term) < s_eps * std::abs(sum))
      break;
  }
  return Real{2} * sum;
}

template<typename Tp>
Tp
jacobi_theta_1_prod(Tp q, Tp x) {
  using Real = numeric_t<Tp>;
  const auto s_eps = std::numeric_limits<Tp>::epsilon();
  constexpr std::size_t s_max_iter = 50;
  const auto q2 = q * q;
  const auto q4 = q2 * q2;
  const auto cos2x = std::cos(Real{2} * x);

  auto q2n = Tp{1};
  auto q4n = Tp{1};
  auto prod = Tp{1};
  for (std::size_t n = 1; n < s_max_iter; ++n) {
    q2n *= q2;
    q4n *= q4;
    const auto fact = (Real{1} - q2n)
        * (Real{1} - Real{2} * q2n * cos2x + q4n);
    prod *= fact;
    if (std::abs(fact) < s_eps)
      break;
  }

  return Real{2} * std::pow(q, Tp{0.25L}) * std::sin(x) * prod;
}

template<typename Tp>
Tp
jacobi_theta_2_sum(Tp q, Tp x) {
  using Real = numeric_t<Tp>;

  Tp sum{};
  for (std::size_t n = 0; n < 50; ++n) {
    sum += std::pow(q, Real((n + 0.5L) * (n + 0.5L)))
        * std::cos(Real(2 * n + 1) * x);
    if (std::abs(std::pow(q, Real((n + 0.5L) * (n + 0.5L)))
                     * std::cos(Real(2 * n + 1) * x)) < std::numeric_limits<Tp>::epsilon() * std::abs(sum))
      break;
  }
  return Real{2} * sum;
}

template<typename Tp>
Tp
jacobi_theta_2_prod(Tp q, Tp x) {
  using Real = numeric_t<Tp>;

  auto q2n = Tp{1};
  auto q4n = Tp{1};
  auto prod = Tp{1};
  for (std::size_t n = 1; n < 50; ++n) {
    q2n *= q * q;
    q4n *= q * q * (q * q);
    const auto fact = (Real{1} - q2n)
        * (Real{1} + Real{2} * q2n * std::cos(Real{2} * x) + q4n);
    prod *= fact;
    if (std::abs(fact) < std::numeric_limits<Tp>::epsilon())
      break;
  }

  return Real{2} * std::pow(q, Tp{0.25L}) * std::cos(x) * prod;
}

// Pre-declare Jacobi elliptic_theta_4 sum ...
template<typename Tp>
Tp
jacobi_theta_4_sum(Tp q, Tp x);

// ... and product.
template<typename Tp>
Tp
jacobi_theta_4_prod(Tp q, Tp x);

template<typename Tp>
Tp
jacobi_theta_3_sum(Tp q, Tp x) {
  using Real = numeric_t<Tp>;
  const auto s_eps = std::numeric_limits<Tp>::epsilon();
  constexpr std::size_t s_max_iter = 50;

  Tp sum{};
  for (std::size_t n = 1; n < s_max_iter; ++n) {
    const auto term = std::pow(q, Real(n * n))
        * std::cos(Real(2 * n) * x);
    sum += term;
    if (std::abs(term) < s_eps * std::abs(sum))
      break;
  }
  return Real{1} + Real{2} * sum;
}

template<typename Tp>
Tp
jacobi_theta_3_prod(Tp q, Tp x) {
  using Real = numeric_t<Tp>;
  const auto s_eps = std::numeric_limits<Tp>::epsilon();
  constexpr std::size_t s_max_iter = 50;
  const auto q2 = q * q;
  const auto q4 = q2 * q2;
  const auto cos2x = std::cos(Real{2} * x);

  auto q2nm1 = q;
  auto q4nm2 = q2;
  auto prod = Tp{1};
  for (std::size_t n = 1; n < s_max_iter; ++n) {
    const auto fact = (Real{1} - q2nm1 * q)
        * (Real{1} + Real{2} * q2nm1 * cos2x + q4nm2);
    prod *= fact;
    if (std::abs(fact) < s_eps)
      break;
    q2nm1 *= q2;
    q4nm2 *= q4;
  }

  return prod;
}

template<typename Tp>
Tp
jacobi_theta_4_sum(Tp q, Tp x) {
  using Real = numeric_t<Tp>;
  const auto s_eps = std::numeric_limits<Tp>::epsilon();
  constexpr std::size_t s_max_iter = 50;

  Tp sum{};
  Real sign{1};
  for (std::size_t n = 1; n < s_max_iter; ++n) {
    sign *= -1;
    const auto term = sign * std::pow(q, Real(n * n))
        * std::cos(Real(2 * n) * x);
    sum += term;
    if (std::abs(term) < s_eps * std::abs(sum))
      break;
  }
  return Real{1} + Real{2} * sum;
}

template<typename Tp>
Tp
jacobi_theta_4_prod(Tp q, Tp x) {
  using Real = numeric_t<Tp>;
  const auto s_eps = std::numeric_limits<Tp>::epsilon();
  constexpr std::size_t s_max_iter = 50;
  const auto q2 = q * q;
  const auto q4 = q2 * q2;
  const auto cos2x = std::cos(Real{2} * x);

  auto q2nm1 = q;
  auto q4nm2 = q2;
  auto prod = Tp{1};
  for (std::size_t n = 1; n < s_max_iter; ++n) {
    const auto fact = (Real{1} - q2nm1 * q)
        * (Real{1} - Real{2} * q2nm1 * cos2x + q4nm2);
    prod *= fact;
    if (std::abs(fact) < s_eps)
      break;
    q2nm1 *= q2;
    q4nm2 *= q4;
  }

  return prod;
}

template<typename Tp>
jacobi_elliptic_t<Tp>
jacobi_ellint(Tp k, Tp u) {
  const auto s_eps = std::numeric_limits<Tp>::epsilon();
  const auto s_NaN = std::numeric_limits<Tp>::quiet_NaN();

  if (std::isnan(k) || std::isnan(u))
    return jacobi_elliptic_t<Tp>{s_NaN, s_NaN, s_NaN};
  else if (std::abs(k) > Tp{1})
    throw std::domain_error("jacobi_ellint: argument k out of range");
  else if (std::abs(Tp{1} - k) < Tp{2} * s_eps) {
    auto sn = std::tanh(u);
    auto cn = Tp{1} / std::cosh(u);
    auto dn = cn;
    return jacobi_elliptic_t<Tp>{sn, cn, dn};
  } else if (std::abs(k) < Tp{2} * s_eps) {
    auto sn = std::sin(u);
    auto cn = std::cos(u);
    auto dn = Tp{1};
    return jacobi_elliptic_t<Tp>{sn, cn, dn};
  } else {
    const auto s_CA = std::sqrt(s_eps * Tp{0.01});
    const auto s_N = 100;
    std::vector<Tp> m;
    std::vector<Tp> n;
    m.reserve(20);
    n.reserve(20);
    Tp c, d = Tp{1};
    auto mc = Tp{1} - k * k;
    const bool bo = (mc < Tp{0});
    if (bo) {
      mc /= -k * k;
      d = k;
      u *= d;
    }
    auto a = Tp{1};
    auto dn = Tp{1};
    auto l = s_N;
    for (auto i = 0; i < s_N; ++i) {
      l = i;
      m.push_back(a);
      n.push_back(mc = std::sqrt(mc));
      c = (a + mc) / Tp{2};
      if (std::abs(a - mc) < s_CA * a)
        break;
      mc *= a;
      a = c;
    }
    u *= c;
    auto sn = std::sin(u);
    auto cn = std::cos(u);
    if (sn != Tp{0}) {
      a = cn / sn;
      c *= a;
      for (auto ii = l; ii >= 0; --ii) {
        const auto b = m[ii];
        a *= c;
        c *= dn;
        dn = (n[ii] + a) / (b + a);
        a = c / b;
      }
      a = Tp{1} / std::hypot(Tp{1}, c);
      sn = std::copysign(a, sn);
      cn = c * sn;
    }
    if (bo) {
      std::swap(dn, cn);
      sn /= d;
    }
    return jacobi_elliptic_t<Tp>{sn, cn, dn};
  }
}
}
}
