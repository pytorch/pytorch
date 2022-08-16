#pragma once

#include <ATen/native/special_functions/cos_pi.h>
#include <ATen/native/special_functions/detail/cyl_bessel_asymp_sums.h>
#include <ATen/native/special_functions/detail/gamma.h>
#include <ATen/native/special_functions/detail/numeric_t.h>
#include <ATen/native/special_functions/detail/promote_t.h>
#include <ATen/native/special_functions/polar_pi.h>
#include <ATen/native/special_functions/sin_pi.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1, typename T2, typename T3>
struct bessel_t {
  T1 n;
  T2 x;

  T3 j;
  T3 j_derivative;

  T3 y;
  T3 y_derivative;
};

template<typename T1, typename T2, typename T3>
struct hankel_t {
  T1 n;
  T2 x;

  T3 h_1;
  T3 h_1_derivative;

  T3 h_2;
  T3 h_2_derivative;
};

template<typename T1>
struct gamma_temme_t {
  T1 m;

  T1 positive;
  T1 negative;

  T1 gamma_1;
  T1 gamma_2;
};

template<typename T1, typename T2, typename T3>
struct spherical_hankel_t {
  T1 n;
  T2 x;

  T3 h_1;
  T3 h_1_derivative;

  T3 h_2;
  T3 h_2_derivative;
};

template<typename T1, typename T2>
constexpr T2
regular_bessel_series_expansion(T1 n, T2 x, int sign, unsigned int max_iter) {
  using T3 = promote_t<T1, T2>;
  using T4 = numeric_t<T3>;

  const auto s_eps = std::numeric_limits<T4>::epsilon();

  if (std::abs(x) < s_eps) {
    if (n == T1(0)) {
      return T2(1);
    } else {
      return T2(0);
    }
  } else {
    auto p = T3(1);
    auto q = T3(1);

    for (unsigned int j = 1; j < max_iter; j++) {
      q = q * (T3(sign) * (x / T4(2)) * (x / T4(2)) / (T3(j) * (T3(n) + T3(j))));
      p = p + q;

      if (std::abs(q / p) < s_eps) {
        break;
      }
    }

    return std::exp(T3(n) * std::log(x / T4(2)) - ln_gamma(T4(1) + n)) * p;
  }
}

template<typename T1>
gamma_temme_t<T1>
gamma_temme(T1 m) {
  using T2 = gamma_temme_t<T1>;

  if (std::abs(m) < std::numeric_limits<T1>::epsilon()) {
    return {m, T1(1), T1(1), -c10::numbers::egamma_v<T1>, T1(1)};
  } else if (std::real(m) <= T1(0)) {
    return {m, T1(+1) * gamma_reciprocal_series(T1(+1) + m), T1(-1) * gamma_reciprocal_series(T1(-1) * m) / m,
            (T1(-1) * gamma_reciprocal_series(T1(-1) * m) / m - T1(+1) * gamma_reciprocal_series(T1(+1) + m))
                / (T1(2) * m),
            (T1(-1) * gamma_reciprocal_series(T1(-1) * m) / m + T1(+1) * gamma_reciprocal_series(T1(+1) + m)) / T1(2)};
  } else {
    return {m, T1(+1) * gamma_reciprocal_series(T1(+1) * m) / m, T1(+1) * gamma_reciprocal_series(T1(+1) - m),
            (T1(+1) * gamma_reciprocal_series(T1(+1) - m) - T1(+1) * gamma_reciprocal_series(T1(+1) * m) / m)
                / (T1(2) * m),
            (T1(+1) * gamma_reciprocal_series(T1(+1) - m) + T1(+1) * gamma_reciprocal_series(T1(+1) * m) / m) / T1(2)};
  }
}

template<typename T1>
bessel_t<T1, T1, T1>
bessel(T1 x, T1 n) {
  const auto nearby_integer_n = std::nearbyint(n);

  const auto negative_n = -n;

  const auto sin_pi_negative_n = at::native::special_functions::sin_pi(negative_n);
  const auto cos_pi_negative_n = at::native::special_functions::cos_pi(negative_n);

  if (n < T1(0)) {
    if (std::abs(sin_pi_negative_n) < std::numeric_limits<T1>::epsilon()) {
      return {n, x, std::copysign(T1(1), cos_pi_negative_n) * bessel(x, negative_n).j,
              std::copysign(T1(1), cos_pi_negative_n) * bessel(
                  x,
                  negative_n).j_derivative, std::copysign(
              T1(1),
              cos_pi_negative_n) * bessel(x, negative_n).y,
              std::copysign(T1(1), cos_pi_negative_n) * bessel(x, negative_n).y_derivative};
    } else if (std::abs(cos_pi_negative_n) < std::numeric_limits<T1>::epsilon()) {
      return {n, x, -std::copysign(T1(1), sin_pi_negative_n) * bessel(x, negative_n).y,
              -std::copysign(T1(1), sin_pi_negative_n) * bessel(
                  x,
                  negative_n).y_derivative, std::copysign(
              T1(1),
              sin_pi_negative_n) * bessel(x, negative_n).j,
              std::copysign(T1(1), sin_pi_negative_n) * bessel(x, negative_n).j_derivative};
    } else {
      return {n, x, cos_pi_negative_n * bessel(x, negative_n).j - sin_pi_negative_n * bessel(x,
                                                                                             negative_n).y,
              cos_pi_negative_n * bessel(
                  x,
                  negative_n).j_derivative - sin_pi_negative_n * bessel(x, negative_n).y_derivative, sin_pi_negative_n
                  * bessel(x, negative_n).j + cos_pi_negative_n * bessel(x, negative_n).y,
              sin_pi_negative_n * bessel(x, negative_n).j_derivative + cos_pi_negative_n * bessel(x,
                                                                                                  negative_n).y_derivative};
    }
  } else if (x == T1(0)) {
    if (n == T1(0)) {
      return {n, x, T1(1), T1(0), -std::numeric_limits<T1>::infinity(), std::numeric_limits<T1>::infinity()};
    } else if (n == T1(1)) {
      return {n, x, T1(0), T1{0.5L}, -std::numeric_limits<T1>::infinity(), std::numeric_limits<T1>::infinity()};
    } else {
      return {n, x, T1(0), T1(0), -std::numeric_limits<T1>::infinity(), std::numeric_limits<T1>::infinity()};
    }
  } else if (x > T1(1000)) {
    using T2 = numeric_t<T1>;
    return {n, x, std::sqrt(T2(2) / (c10::numbers::pi_v<T2> * x))
        * (std::cos(x - (n + T2{0.5L}) * (c10::numbers::pi_v<T2> / T2(2))) * cyl_bessel_asymp_sums(n, x, -1).Psum
            - std::sin(x - (n + T2{0.5L}) * (c10::numbers::pi_v<T2> / T2(2))) * cyl_bessel_asymp_sums(n, x, -1).Qsum),
            -std::sqrt(T2(2) / (c10::numbers::pi_v<T2> * x))
                * (std::sin(x - (n + T2{0.5L}) * (c10::numbers::pi_v<T2> / T2(2)))
                    * cyl_bessel_asymp_sums(n, x, -1).Rsum
                    + std::cos(x - (n + T2{0.5L}) * (c10::numbers::pi_v<T2> / T2(2)))
                        * cyl_bessel_asymp_sums(n, x, -1).Ssum), std::sqrt(T2(2) / (c10::numbers::pi_v<T2> * x))
                * (std::sin(x - (n + T2{0.5L}) * (c10::numbers::pi_v<T2> / T2(2)))
                    * cyl_bessel_asymp_sums(n, x, -1).Psum
                    + std::cos(x - (n + T2{0.5L}) * (c10::numbers::pi_v<T2> / T2(2)))
                        * cyl_bessel_asymp_sums(n, x, -1).Qsum), std::sqrt(T2(2) / (c10::numbers::pi_v<T2> * x))
                * (std::cos(x - (n + T2{0.5L}) * (c10::numbers::pi_v<T2> / T2(2)))
                    * cyl_bessel_asymp_sums(n, x, -1).Rsum
                    - std::sin(x - (n + T2{0.5L}) * (c10::numbers::pi_v<T2> / T2(2)))
                        * cyl_bessel_asymp_sums(n, x, -1).Ssum)};
  } else {
    const auto s_fp_min = std::sqrt(std::numeric_limits<T1>::min());

    int isign = 1;
    auto h = std::max(s_fp_min, n * (T1(1) / x));
    auto b = T1(2) * (T1(1) / x) * n;
    auto d = T1(0);
    auto c = h;
    int i;

    for (i = 1; i <= 15000; ++i) {
      b += T1(2) * (T1(1) / x);
      d = b - d;
      if (std::abs(d) < s_fp_min)
        d = s_fp_min;
      d = T1(1) / d;
      c = b - T1(1) / c;
      if (std::abs(c) < s_fp_min)
        c = s_fp_min;
      const auto del = c * d;
      h *= del;
      if (d < T1(0))
        isign = -isign;
      if (std::abs(del - T1(1)) < std::numeric_limits<T1>::epsilon())
        break;
    }

    if (i > 15000) {
      using T4 = numeric_t<T1>;
      return {n, x, std::sqrt(T4(2) / (c10::numbers::pi_v<T4> * x))
          * (std::cos(x - (n + T4{0.5L}) * (c10::numbers::pi_v<T4> / T4(2))) * cyl_bessel_asymp_sums(n, x, -1).Psum
              - std::sin(x - (n + T4{0.5L}) * (c10::numbers::pi_v<T4> / T4(2))) * cyl_bessel_asymp_sums(n, x, -1).Qsum),
              -std::sqrt(T4(2) / (c10::numbers::pi_v<T4> * x))
                  * (std::sin(x - (n + T4{0.5L}) * (c10::numbers::pi_v<T4> / T4(2)))
                      * cyl_bessel_asymp_sums(n, x, -1).Rsum
                      + std::cos(x - (n + T4{0.5L}) * (c10::numbers::pi_v<T4> / T4(2)))
                          * cyl_bessel_asymp_sums(n, x, -1).Ssum), std::sqrt(T4(2) / (c10::numbers::pi_v<T4> * x))
                  * (std::sin(x - (n + T4{0.5L}) * (c10::numbers::pi_v<T4> / T4(2)))
                      * cyl_bessel_asymp_sums(n, x, -1).Psum
                      + std::cos(x - (n + T4{0.5L}) * (c10::numbers::pi_v<T4> / T4(2)))
                          * cyl_bessel_asymp_sums(n, x, -1).Qsum), std::sqrt(T4(2) / (c10::numbers::pi_v<T4> * x))
                  * (std::cos(x - (n + T4{0.5L}) * (c10::numbers::pi_v<T4> / T4(2)))
                      * cyl_bessel_asymp_sums(n, x, -1).Rsum
                      - std::sin(x - (n + T4{0.5L}) * (c10::numbers::pi_v<T4> / T4(2)))
                          * cyl_bessel_asymp_sums(n, x, -1).Ssum)};
    }

    auto Jnul = isign * s_fp_min;
    auto Jpnul = h * Jnul;
    auto Jnul1 = Jnul;
    auto Jpnu1 = Jpnul;
    auto fact = n * (T1(1) / x);

    for (int l = (x < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(n - x + T1{1.5L}))); l >= 1; --l) {
      const auto Jnutemp = fact * Jnul + Jpnul;
      fact -= T1(1) / x;
      Jpnul = fact * Jnutemp - Jnul;
      Jnul = Jnutemp;
    }

    if (Jnul == T1(0))
      Jnul = std::numeric_limits<T1>::epsilon();

    T1 Nmu, Nnu1, Npmu, Jmu;

    if (x < T1(2)) {
      auto d = -std::log(x / T1(2));
      auto e = (n - T1((x < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(n - x + T1{1.5L}))))) * d;
      auto ff = (T1(2) / c10::numbers::pi_v<T1>) * (std::abs(c10::numbers::pi_v<T1> * (n
          - T1((x < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(n - x + T1{1.5L}))))))
                                                        < std::numeric_limits<T1>::epsilon() ? T1(1) :
                                                    c10::numbers::pi_v<T1> * (n
                                                        - T1((x < T1(2) ? nearby_integer_n : std::max(0,
                                                                                                      static_cast<int>(n
                                                                                                          - x + T1{
                                                                                                          1.5L})))))
                                                        / std::sin(c10::numbers::pi_v<T1> * (n
                                                            - T1((x < T1(2) ? nearby_integer_n : std::max(0,
                                                                                                          static_cast<int>(
                                                                                                              n - x
                                                                                                                  + T1{
                                                                                                                      1.5L})))))))
          * (gamma_temme(
              n - T1((x < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(n - x + T1{1.5L}))))).gamma_1
              * std::cosh(e) + gamma_temme(
              n - T1((x < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(n - x + T1{1.5L}))))).gamma_2
              * (std::abs(e) < std::numeric_limits<T1>::epsilon() ? T1(1) : std::sinh(e) / e) * d);
      e = std::exp(e);
      auto p = e / (c10::numbers::pi_v<T1> * gamma_temme(
          n - T1((x < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>( n - x + T1{1.5L}))))).positive);
      auto q = T1(1) / (e * c10::numbers::pi_v<T1> * gamma_temme(
          n - T1((x < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>( n - x + T1{1.5L}))))).negative);

      auto c = T1(1);
      d = -(x / T1(2)) * (x / T1(2));
      auto sum = ff + c10::numbers::pi_v<T1> * (c10::numbers::pi_v<T1>
          * (n - T1((x < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(n - x + T1{1.5L}))))) / T1(2)) * (
          std::abs(c10::numbers::pi_v<T1>
                       * (n - T1((x < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>( n - x + T1{1.5L})))))
                       / T1(2)) < std::numeric_limits<T1>::epsilon() ? T1(1) : std::sin(c10::numbers::pi_v<T1> * (n
              - T1((x < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(n - x + T1{1.5L}))))) / T1(2))
              / (c10::numbers::pi_v<T1>
                  * (n - T1((x < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(n - x + T1{1.5L}))))) / T1(2)))
          * (std::abs(c10::numbers::pi_v<T1>
                          * (n - T1((x < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>( n - x + T1{1.5L})))))
                          / T1(2)) < std::numeric_limits<T1>::epsilon() ? T1(1) : std::sin(c10::numbers::pi_v<T1> * (n
              - T1((x < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(n - x + T1{1.5L}))))) / T1(2))
                 / (c10::numbers::pi_v<T1>
                     * (n - T1((x < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(n - x + T1{1.5L})))))
                     / T1(2))) * q;
      auto sum1 = p;
      int i;

      for (i = 1; i <= 15000; ++i) {
        ff = (i * ff + p + q) / (i * i
            - (n - T1((x < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(n - x + T1{1.5L})))))
                * (n - T1((x < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(n - x + T1{1.5L}))))));
        c *= d / T1(i);
        p /= T1(i) - (n - T1((x < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(n - x + T1{1.5L})))));
        q /= T1(i) + (n - T1((x < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(n - x + T1{1.5L})))));
        const auto del = c * (ff + c10::numbers::pi_v<T1> * (c10::numbers::pi_v<T1>
            * (n - T1((x < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(n - x + T1{1.5L}))))) / T1(2)) * (
            std::abs(c10::numbers::pi_v<T1>
                         * (n - T1((x < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>( n - x + T1{1.5L})))))
                         / T1(2)) < std::numeric_limits<T1>::epsilon() ? T1(1) : std::sin(c10::numbers::pi_v<T1> * (n
                - T1((x < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(n - x + T1{1.5L}))))) / T1(2))
                / (c10::numbers::pi_v<T1>
                    * (n - T1((x < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(n - x + T1{1.5L})))))
                    / T1(2)))
            * (std::abs(c10::numbers::pi_v<T1>
                            * (n
                                - T1((x < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>( n - x + T1{1.5L})))))
                            / T1(2)) < std::numeric_limits<T1>::epsilon() ? T1(1) : std::sin(c10::numbers::pi_v<T1> * (n
                - T1((x < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(n - x + T1{1.5L}))))) / T1(2))
                   / (c10::numbers::pi_v<T1>
                       * (n - T1((x < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(n - x + T1{1.5L})))))
                       / T1(2))) * q);
        sum += del;
        const auto del1 = c * p - T1(i) * del;
        sum1 += del1;
        if (std::abs(del) < std::numeric_limits<T1>::epsilon() * (T1(1) + std::abs(sum)))
          break;
      }
      if (i > 15000)
        throw std::runtime_error("cyl_bessel_jn_steed: Y-series failed to converge");
      Nmu = -sum;
      Nnu1 = -sum1 * (T1(2) * (T1(1) / x));
      Npmu = (n - T1((x < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(n - x + T1{1.5L}))))) * (T1(1) / x)
          * Nmu - Nnu1;
      Jmu = T1(2) * (T1(1) / x) / c10::numbers::pi_v<T1> / (Npmu - Jpnul / Jnul * Nmu);
    } else {
      auto a = T1{0.25L}
          - (n - T1((x < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(n - x + T1{1.5L}))))) * (n - T1(
              (x < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(n - x + T1{1.5L})))));
      auto pq = std::complex<T1>(-(T1(1) / x) / T1(2), T1(1));
      auto b = std::complex<T1>(T1(2) * x, T1(2));
      auto fact = a * (T1(1) / x) / std::norm(pq);
      auto c = b + std::complex<T1>{0, 1} * fact * std::conj(pq);
      auto d = std::conj(b) / std::norm(b);
      auto dl = c * d;
      pq *= dl;
      int i;
      for (i = 2; i <= 15000; ++i) {
        a += T1(2 * (i - 1));
        b += std::complex<T1>{0, 1} * T1(2);
        d = a * d + b;
        if (std::abs(d) < s_fp_min)
          d = s_fp_min;
        fact = a / std::norm(c);
        c = b + fact * std::conj(c);
        if (std::abs(c) < s_fp_min)
          c = s_fp_min;
        d = std::conj(d) / std::norm(d);
        dl = c * d;
        pq *= dl;
        if (std::abs(dl - T1(1)) < std::numeric_limits<T1>::epsilon())
          break;
      }

      if (i > 15000) { throw std::runtime_error("cyl_bessel_jn_steed: Lentz's method failed"); }

      const auto foo = reinterpret_cast<T1(&)[2]>(pq);

      const auto p = foo[0];
      const auto q = foo[1];

      Jmu = std::sqrt(T1(2) * (T1(1) / x) / c10::numbers::pi_v<T1> / ((p - Jpnul / Jnul) * ((p - Jpnul / Jnul) / q) + q));
      Jmu = std::copysign(Jmu, Jnul);
      Nmu = (p - Jpnul / Jnul) / q * Jmu;
      Npmu = (p + q / ((p - Jpnul / Jnul) / q)) * Nmu;
      Nnu1 = (n - T1((x < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(n - x + T1{1.5L}))))) * (T1(1) / x) * Nmu - Npmu;
    }

    fact = Jmu / Jnul;

    if (std::abs(c10::numbers::pi_v<T1> * x * (fact * Jnul1) / T1(2)) > std::numeric_limits<T1>::min()) {
      for (int j = 1; j <= (x < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(n - x + T1{1.5L}))); j++) {
        Nmu = std::exchange(Nnu1,
                            (n - T1((x < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(n - x + T1{1.5L}))))
                                + j) * (T1(2) * (T1(1) / x)) * Nnu1 - Nmu);
      }

      return {n, x, fact * Jnul1, fact * Jpnu1, Nmu, n * (T1(1) / x) * Nmu - Nnu1};
    } else {
      return {n, x, fact * Jnul1, fact * Jpnu1, -std::numeric_limits<T1>::infinity(),
              std::numeric_limits<T1>::infinity()};
    }
  }
}

template<typename T1>
bessel_t<T1, T1, std::complex<T1>>
bessel_negative_x(T1 n, T1 x) {
  using T2 = std::complex<T1>;
  using T3 = bessel_t<T1, T1, T2>;

  if (x >= T1(0)) {
    throw std::domain_error("non-negative `x`");
  } else {
    return {n, x, at::native::special_functions::polar_pi(T1(1), +n) * bessel(-x, n).j,
            -at::native::special_functions::polar_pi(T1(1), +n) * bessel(-x, n).j_derivative,
            at::native::special_functions::polar_pi(T1(1), -n) * bessel(-x, n).y
                + T2{0, 1} * T1(2) * at::native::special_functions::cos_pi(n) * bessel(-x, n).j,
            -at::native::special_functions::polar_pi(T1(1), -n) * bessel(-x, n).y_derivative
                - T2{0, 1} * T1(2) * at::native::special_functions::cos_pi(n) * bessel(-x, n).j_derivative};
  }
}
}
}
}
}
