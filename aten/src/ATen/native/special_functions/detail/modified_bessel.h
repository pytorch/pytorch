#pragma once

#include <ATen/native/special_functions/detail/cyl_bessel_asymp_sums.h>
#include <ATen/native/special_functions/detail/numeric_t.h>
#include <ATen/native/special_functions/detail/promote_t.h>
#include <ATen/native/special_functions/sin_pi.h>
#include <c10/util/numbers.h>
#include <ATen/native/special_functions/detail/bessel.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1, typename T2, typename T3>
struct modified_bessel_t {
  T1 n;
  T2 x;

  T3 i;
  T3 i_derivative;

  T3 k;
  T3 k_derivative;
};

template<typename T1, typename T2>
constexpr modified_bessel_t<T1, T2, T2>
cyl_bessel_ik_scaled_asymp(T1 n, T2 x) {
  using T3 = promote_t<T1, T2>;
  using T4 = numeric_t<T3>;

  return {n, x, std::sqrt(T4(1) / (T4(2) * c10::numbers::pi_v<T4> * x))
      * (cyl_bessel_asymp_sums(n, x, +1).Psum - cyl_bessel_asymp_sums(n, x, +1).Qsum),
          std::sqrt(T4(1) / (T4(2) * c10::numbers::pi_v<T4> * x))
              * (cyl_bessel_asymp_sums(n, x, +1).Rsum - cyl_bessel_asymp_sums(n, x, +1).Ssum),
          std::sqrt(T4(1) / (T4(2) * c10::numbers::pi_v<T4> * x)) * c10::numbers::pi_v<T4>
              * (cyl_bessel_asymp_sums(n, x, +1).Psum + cyl_bessel_asymp_sums(n, x, +1).Qsum),
          -std::sqrt(T4(1) / (T4(2) * c10::numbers::pi_v<T4> * x)) * c10::numbers::pi_v<T4>
              * (cyl_bessel_asymp_sums(n, x, +1).Rsum + cyl_bessel_asymp_sums(n, x, +1).Ssum)};
}

template<typename T1, typename T2>
constexpr modified_bessel_t<T1, T2, T2>
cyl_bessel_ik_asymp(T1 n, T2 x, bool do_scaled = false) {
  if (do_scaled) {
    return cyl_bessel_ik_scaled_asymp(n, x);
  } else {
    return {cyl_bessel_ik_scaled_asymp(n, x).n, cyl_bessel_ik_scaled_asymp(n, x).x,
            std::exp(x) * cyl_bessel_ik_scaled_asymp(n, x).i,
            std::exp(x) * cyl_bessel_ik_scaled_asymp(n, x).i_derivative,
            T2(1) / std::exp(x) * cyl_bessel_ik_scaled_asymp(n, x).k,
            T2(1) / std::exp(x) * cyl_bessel_ik_scaled_asymp(n, x).k_derivative};
  }
}

template<typename T1>
modified_bessel_t<T1, T1, T1>
cyl_bessel_ik_steed(T1 nu, T1 x, bool do_scaled = false) {
  using T2 = modified_bessel_t<T1, T1, T1>;

  auto h = std::max(T1(10) * std::numeric_limits<T1>::epsilon(), nu * (T1(1) / x));
  auto b = T1(2) * (T1(1) / x) * nu;
  auto d = T1(0);
  auto c = h;
  int i;

  for (i = 1; i <= 15000; ++i) {
    b += T1(2) * (T1(1) / x);
    d = T1(1) / (b + d);
    c = b + T1(1) / c;
    const auto del = c * d;
    h *= del;
    if (std::abs(del - T1(1)) < std::numeric_limits<T1>::epsilon())
      break;
  }

  if (i > 15000) {
    return cyl_bessel_ik_asymp(nu, x, do_scaled);
  }

  auto Inul = T1(10) * std::numeric_limits<T1>::epsilon();
  auto Ipnul = h * Inul;
  auto Inul1 = Inul;
  auto Ipnu1 = Ipnul;
  auto fact = nu * (T1(1) / x);

  for (int l = std::nearbyint(nu); l >= 1; --l) {
    const auto Inutemp = fact * Inul + Ipnul;
    fact -= T1(1) / x;
    Ipnul = fact * Inutemp + Inul;
    Inul = Inutemp;
  }

  const auto f = Ipnul / Inul;
  bool scaled = false;
  T1 Kmu, Knu1;

  if (x < T1(2)) {
    auto d = -std::log(x / T1(2));
    auto e = (nu - T1(std::nearbyint(nu))) * d;
    auto ff =
        (std::abs(c10::numbers::pi_v<T1> * (nu - T1(std::nearbyint(nu)))) < std::numeric_limits<T1>::epsilon() ? T1(1) :
         c10::numbers::pi_v<T1> * (nu - T1(std::nearbyint(nu)))
             / std::sin(c10::numbers::pi_v<T1> * (nu - T1(std::nearbyint(nu)))))
            * (gamma_temme(nu - T1(std::nearbyint(nu))).gamma_1 * std::cosh(e)
                + gamma_temme(nu - T1(std::nearbyint(nu))).gamma_2
                    * (std::abs(e) < std::numeric_limits<T1>::epsilon() ? T1(1) : std::sinh(e) / e) * d);
    auto sum = ff;
    e = std::exp(e);
    auto p = e / (T1(2) * gamma_temme(nu - T1(std::nearbyint(nu))).positive);
    auto q = T1(1) / (T1(2) * e * gamma_temme(nu - T1(std::nearbyint(nu))).negative);
    auto c = T1(1);
    d = x / T1(2) * (x / T1(2));
    auto sum1 = p;
    int i;

    for (i = 1; i <= 15000; i++) {
      ff = (i * ff + p + q) / (i * i - (nu - T1(std::nearbyint(nu))) * (nu - T1(std::nearbyint(nu))));
      c *= d / T1(i);
      p /= T1(i) - (nu - T1(std::nearbyint(nu)));
      q /= T1(i) + (nu - T1(std::nearbyint(nu)));
      const auto del = c * ff;
      sum += del;
      const auto del1 = c * (p - T1(i) * ff);
      sum1 += del1;
      if (std::abs(del) < std::numeric_limits<T1>::epsilon() * std::abs(sum)) { break; }
    }

    if (i > 15000)
      throw std::runtime_error("cyl_bessel_ik_steed: K-series failed to converge");

    Kmu = sum;
    Knu1 = sum1 * (T1(2) * (T1(1) / x));
  } else {
    scaled = true;
    auto b = T1(2) * (T1(1) + x);
    auto d = T1(1) / b;
    auto delh = d;
    auto h = delh;
    auto q1 = T1(0);
    auto q2 = T1(1);
    const auto a1 = T1{0.25L} - (nu - T1(std::nearbyint(nu))) * (nu - T1(std::nearbyint(nu)));
    auto q = c = a1;
    auto a = -a1;
    auto s = T1(1) + q * delh;
    int i;
    for (i = 2; i <= 15000; ++i) {
      a -= T1(2 * (i - 1));
      c = -a * c / i;
      const auto qnew = (q1 - b * q2) / a;
      q1 = q2;
      q2 = qnew;
      q += c * qnew;
      b += T1(2);
      d = T1(1) / (b + a * d);
      delh = (b * d - T1(1)) * delh;
      h += delh;
      const auto dels = q * delh;
      s += dels;
      if (std::abs(dels / s) < std::numeric_limits<T1>::epsilon())
        break;
    }
    if (i > 15000)
      throw std::runtime_error("cyl_bessel_ik_steed: Steed's method failed");
    h = a1 * h;
    Kmu = std::sqrt(c10::numbers::pi_v<T1> / (T1(2) * x)) / s;
    Knu1 = Kmu * (nu - T1(std::nearbyint(nu)) + x + T1{0.5L} - h) * (T1(1) / x);
  }

  auto Kpmu = (nu - T1(std::nearbyint(nu))) * (T1(1) / x) * Kmu - Knu1;
  auto Inumu = T1(1) / x / (f * Kmu - Kpmu);
  auto Inu = Inumu * Inul1 / Inul;
  auto i_derivative = Inumu * Ipnu1 / Inul;
  for (int i = 1; i <= std::nearbyint(nu); ++i) {
    Kmu = std::exchange(Knu1, (nu - T1(std::nearbyint(nu)) + T1(i)) * (T1(2) * (T1(1) / x)) * Knu1 + Kmu);
  }
  auto Knu = Kmu;
  auto k_derivative = nu * (T1(1) / x) * Kmu - Knu1;

  if (do_scaled && !scaled) {
    Inu *= T1(1) / std::exp(x);
    i_derivative *= T1(1) / std::exp(x);
    Knu *= std::exp(x);
    k_derivative *= std::exp(x);
  } else if (!do_scaled && scaled) {
    Inu *= std::exp(x);
    i_derivative *= std::exp(x);
    Knu *= T1(1) / std::exp(x);
    k_derivative *= T1(1) / std::exp(x);
  }

  return {
      nu,
      x,
      Inu,
      i_derivative,
      Knu,
      k_derivative,
  };
}

template<typename T1>
modified_bessel_t<T1, T1, T1>
modified_bessel(T1 n, T1 x, bool exp = false) {
  using T2 = modified_bessel_t<T1, T1, T1>;

  if (n < T1(0)) {
    if (std::abs(at::native::special_functions::sin_pi(-n)) < std::numeric_limits<T1>::epsilon()) {
      return {n, x, modified_bessel(-n, x, exp).i, modified_bessel(-n, x, exp).i_derivative,
              modified_bessel(-n, x, exp).k, modified_bessel(-n, x, exp).k_derivative};
    } else {
      return {n, x, modified_bessel(-n, x, exp).i
          + at::native::special_functions::sin_pi(-n) * T1(2) * modified_bessel(-n, x, exp).k
              / c10::numbers::pi_v<T1>, modified_bessel(-n, x, exp).i_derivative
                  + at::native::special_functions::sin_pi(-n) * T1(2) * modified_bessel(-n, x, exp).k_derivative
                      / c10::numbers::pi_v<T1>, modified_bessel(-n, x, exp).k,
              modified_bessel(-n, x, exp).k_derivative};
    }
  } else if (x == T1(0)) {
    if (n == T1(0)) {
      return {n, x, T1(1), T1(0), std::numeric_limits<T1>::infinity(), -std::numeric_limits<T1>::infinity()};
    } else if (n == T1(1)) {
      return {n, x, T1(0), T1{0.5L}, std::numeric_limits<T1>::infinity(), -std::numeric_limits<T1>::infinity()};
    } else {
      return {n, x, T1(0), T1(0), std::numeric_limits<T1>::infinity(), -std::numeric_limits<T1>::infinity()};
    }
  } else if (x > T1(1000)) {
    return cyl_bessel_ik_asymp(n, x, exp);
  } else {
    return cyl_bessel_ik_steed(n, x, exp);
  }
}
}
}
}
}
