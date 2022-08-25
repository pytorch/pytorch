#pragma once

#include <ATen/native/special/cos_pi.h>
#include <ATen/native/special/detail/cyl_bessel_asymp_sums.h>
#include <ATen/native/special/detail/gamma_temme.h>
#include <ATen/native/special/detail/bessel_t.h>
#include <ATen/native/special/sin_pi.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
bessel_t<T1, T1, T1>
bessel(T1 v, T1 z) {
  const auto nearby_integer_n = std::nearbyint(v);

  const auto negative_n = -v;

  const auto sin_pi_negative_n = at::native::special::sin_pi(negative_n);
  const auto cos_pi_negative_n = at::native::special::cos_pi(negative_n);

  if (v < T1(0)) {
    if (std::abs(sin_pi_negative_n) < std::numeric_limits<T1>::epsilon()) {
      return {v, z, std::copysign(T1(1), cos_pi_negative_n) * bessel(negative_n, z).j,
              std::copysign(T1(1), cos_pi_negative_n) * bessel(
                  negative_n,
                  z).j_derivative, std::copysign(
              T1(1),
              cos_pi_negative_n) * bessel(negative_n, z).y,
              std::copysign(T1(1), cos_pi_negative_n) * bessel(negative_n, z).y_derivative};
    } else if (std::abs(cos_pi_negative_n) < std::numeric_limits<T1>::epsilon()) {
      return {v, z, -std::copysign(T1(1), sin_pi_negative_n) * bessel(negative_n, z).y,
              -std::copysign(T1(1), sin_pi_negative_n) * bessel(
                  negative_n,
                  z).y_derivative, std::copysign(
              T1(1),
              sin_pi_negative_n) * bessel(negative_n, z).j,
              std::copysign(T1(1), sin_pi_negative_n) * bessel(negative_n, z).j_derivative};
    } else {
      return {v, z, cos_pi_negative_n * bessel(negative_n, z).j - sin_pi_negative_n * bessel(
          negative_n, z).y,
              cos_pi_negative_n * bessel(
                  negative_n,
                  z).j_derivative - sin_pi_negative_n * bessel(negative_n, z).y_derivative, sin_pi_negative_n
                  * bessel(negative_n, z).j + cos_pi_negative_n * bessel(negative_n, z).y,
              sin_pi_negative_n * bessel(negative_n, z).j_derivative + cos_pi_negative_n * bessel(
                  negative_n, z).y_derivative};
    }
  } else if (z == T1(0)) {
    if (v == T1(0)) {
      return {v, z, T1(1), T1(0), -std::numeric_limits<T1>::infinity(), std::numeric_limits<T1>::infinity()};
    } else if (v == T1(1)) {
      return {v, z, T1(0), T1{0.5L}, -std::numeric_limits<T1>::infinity(), std::numeric_limits<T1>::infinity()};
    } else {
      return {v, z, T1(0), T1(0), -std::numeric_limits<T1>::infinity(), std::numeric_limits<T1>::infinity()};
    }
  } else if (z > T1(1000)) {
    using T2 = numeric_t<T1>;
    return {v, z, std::sqrt(T2(2) / (c10::numbers::pi_v<T2> * z))
        * (std::cos(z - (v + T2{0.5L}) * (c10::numbers::pi_v<T2> / T2(2))) * cyl_bessel_asymp_sums(v, z, -1).Psum
            - std::sin(z - (v + T2{0.5L}) * (c10::numbers::pi_v<T2> / T2(2))) * cyl_bessel_asymp_sums(v, z, -1).Qsum),
            -std::sqrt(T2(2) / (c10::numbers::pi_v<T2> * z))
                * (std::sin(z - (v + T2{0.5L}) * (c10::numbers::pi_v<T2> / T2(2)))
                    * cyl_bessel_asymp_sums(v, z, -1).Rsum
                    + std::cos(z - (v + T2{0.5L}) * (c10::numbers::pi_v<T2> / T2(2)))
                        * cyl_bessel_asymp_sums(v, z, -1).Ssum), std::sqrt(T2(2) / (c10::numbers::pi_v<T2> * z))
                * (std::sin(z - (v + T2{0.5L}) * (c10::numbers::pi_v<T2> / T2(2)))
                    * cyl_bessel_asymp_sums(v, z, -1).Psum
                    + std::cos(z - (v + T2{0.5L}) * (c10::numbers::pi_v<T2> / T2(2)))
                        * cyl_bessel_asymp_sums(v, z, -1).Qsum), std::sqrt(T2(2) / (c10::numbers::pi_v<T2> * z))
                * (std::cos(z - (v + T2{0.5L}) * (c10::numbers::pi_v<T2> / T2(2)))
                    * cyl_bessel_asymp_sums(v, z, -1).Rsum
                    - std::sin(z - (v + T2{0.5L}) * (c10::numbers::pi_v<T2> / T2(2)))
                        * cyl_bessel_asymp_sums(v, z, -1).Ssum)};
  } else {
    const auto s_fp_min = std::sqrt(std::numeric_limits<T1>::min());

    int isign = 1;
    auto h = std::max(s_fp_min, v * (T1(1) / z));
    auto b = T1(2) * (T1(1) / z) * v;
    auto d = T1(0);
    auto c = h;
    int i;

    for (i = 1; i <= 15000; ++i) {
      b += T1(2) * (T1(1) / z);
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
      return {v, z, std::sqrt(T4(2) / (c10::numbers::pi_v<T4> * z))
          * (std::cos(z - (v + T4{0.5L}) * (c10::numbers::pi_v<T4> / T4(2))) * cyl_bessel_asymp_sums(v, z, -1).Psum
              - std::sin(z - (v + T4{0.5L}) * (c10::numbers::pi_v<T4> / T4(2))) * cyl_bessel_asymp_sums(v, z, -1).Qsum),
              -std::sqrt(T4(2) / (c10::numbers::pi_v<T4> * z))
                  * (std::sin(z - (v + T4{0.5L}) * (c10::numbers::pi_v<T4> / T4(2)))
                      * cyl_bessel_asymp_sums(v, z, -1).Rsum
                      + std::cos(z - (v + T4{0.5L}) * (c10::numbers::pi_v<T4> / T4(2)))
                          * cyl_bessel_asymp_sums(v, z, -1).Ssum), std::sqrt(T4(2) / (c10::numbers::pi_v<T4> * z))
                  * (std::sin(z - (v + T4{0.5L}) * (c10::numbers::pi_v<T4> / T4(2)))
                      * cyl_bessel_asymp_sums(v, z, -1).Psum
                      + std::cos(z - (v + T4{0.5L}) * (c10::numbers::pi_v<T4> / T4(2)))
                          * cyl_bessel_asymp_sums(v, z, -1).Qsum), std::sqrt(T4(2) / (c10::numbers::pi_v<T4> * z))
                  * (std::cos(z - (v + T4{0.5L}) * (c10::numbers::pi_v<T4> / T4(2)))
                      * cyl_bessel_asymp_sums(v, z, -1).Rsum
                      - std::sin(z - (v + T4{0.5L}) * (c10::numbers::pi_v<T4> / T4(2)))
                          * cyl_bessel_asymp_sums(v, z, -1).Ssum)};
    }

    auto Jnul = isign * s_fp_min;
    auto Jpnul = h * Jnul;
    auto Jnul1 = Jnul;
    auto Jpnu1 = Jpnul;
    auto fact = v * (T1(1) / z);

    for (int l = (z < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(v - z + T1{1.5L}))); l >= 1; --l) {
      const auto Jnutemp = fact * Jnul + Jpnul;
      fact -= T1(1) / z;
      Jpnul = fact * Jnutemp - Jnul;
      Jnul = Jnutemp;
    }

    if (Jnul == T1(0))
      Jnul = std::numeric_limits<T1>::epsilon();

    T1 Nmu, Nnu1, Npmu, Jmu;

    if (z < T1(2)) {
      auto d = -std::log(z / T1(2));
      auto e = (v - T1((z < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(v - z + T1{1.5L}))))) * d;
      auto ff = (T1(2) / c10::numbers::pi_v<T1>) * (std::abs(c10::numbers::pi_v<T1> * (v
          - T1((z < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(v - z + T1{1.5L}))))))
                                                        < std::numeric_limits<T1>::epsilon() ? T1(1) :
                                                    c10::numbers::pi_v<T1> * (v
                                                        - T1((z < T1(2) ? nearby_integer_n : std::max(0,
                                                                                                      static_cast<int>(v
                                                                                                          - z + T1{
                                                                                                          1.5L})))))
                                                        / std::sin(c10::numbers::pi_v<T1> * (v
                                                            - T1((z < T1(2) ? nearby_integer_n : std::max(0,
                                                                                                          static_cast<int>(
                                                                                                              v - z
                                                                                                                  + T1{
                                                                                                                      1.5L})))))))
          * (gamma_temme(
              v - T1((z < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(v - z + T1{1.5L}))))).gamma_1
              * std::cosh(e) + gamma_temme(
              v - T1((z < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(v - z + T1{1.5L}))))).gamma_2
              * (std::abs(e) < std::numeric_limits<T1>::epsilon() ? T1(1) : std::sinh(e) / e) * d);
      e = std::exp(e);
      auto p = e / (c10::numbers::pi_v<T1> * gamma_temme(
          v - T1((z < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>( v - z + T1{1.5L}))))).positive);
      auto q = T1(1) / (e * c10::numbers::pi_v<T1> * gamma_temme(
          v - T1((z < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>( v - z + T1{1.5L}))))).negative);

      auto c = T1(1);
      d = -(z / T1(2)) * (z / T1(2));
      auto sum = ff + c10::numbers::pi_v<T1> * (c10::numbers::pi_v<T1>
          * (v - T1((z < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(v - z + T1{1.5L}))))) / T1(2)) * (
          std::abs(c10::numbers::pi_v<T1>
                       * (v - T1((z < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>( v - z + T1{1.5L})))))
                       / T1(2)) < std::numeric_limits<T1>::epsilon() ? T1(1) : std::sin(c10::numbers::pi_v<T1> * (v
              - T1((z < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(v - z + T1{1.5L}))))) / T1(2))
              / (c10::numbers::pi_v<T1>
                  * (v - T1((z < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(v - z + T1{1.5L}))))) / T1(2)))
          * (std::abs(c10::numbers::pi_v<T1>
                          * (v - T1((z < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>( v - z + T1{1.5L})))))
                          / T1(2)) < std::numeric_limits<T1>::epsilon() ? T1(1) : std::sin(c10::numbers::pi_v<T1> * (v
              - T1((z < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(v - z + T1{1.5L}))))) / T1(2))
                 / (c10::numbers::pi_v<T1>
                     * (v - T1((z < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(v - z + T1{1.5L})))))
                     / T1(2))) * q;
      auto sum1 = p;
      int i;

      for (i = 1; i <= 15000; ++i) {
        ff = (i * ff + p + q) / (i * i
            - (v - T1((z < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(v - z + T1{1.5L})))))
                * (v - T1((z < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(v - z + T1{1.5L}))))));
        c *= d / T1(i);
        p /= T1(i) - (v - T1((z < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(v - z + T1{1.5L})))));
        q /= T1(i) + (v - T1((z < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(v - z + T1{1.5L})))));
        const auto del = c * (ff + c10::numbers::pi_v<T1> * (c10::numbers::pi_v<T1>
            * (v - T1((z < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(v - z + T1{1.5L}))))) / T1(2)) * (
            std::abs(c10::numbers::pi_v<T1>
                         * (v - T1((z < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>( v - z + T1{1.5L})))))
                         / T1(2)) < std::numeric_limits<T1>::epsilon() ? T1(1) : std::sin(c10::numbers::pi_v<T1> * (v
                - T1((z < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(v - z + T1{1.5L}))))) / T1(2))
                / (c10::numbers::pi_v<T1>
                    * (v - T1((z < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(v - z + T1{1.5L})))))
                    / T1(2)))
            * (std::abs(c10::numbers::pi_v<T1>
                            * (v
                                - T1((z < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>( v - z + T1{1.5L})))))
                            / T1(2)) < std::numeric_limits<T1>::epsilon() ? T1(1) : std::sin(c10::numbers::pi_v<T1> * (v
                - T1((z < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(v - z + T1{1.5L}))))) / T1(2))
                   / (c10::numbers::pi_v<T1>
                       * (v - T1((z < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(v - z + T1{1.5L})))))
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
      Nnu1 = -sum1 * (T1(2) * (T1(1) / z));
      Npmu = (v - T1((z < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(v - z + T1{1.5L}))))) * (T1(1) / z)
          * Nmu - Nnu1;
      Jmu = T1(2) * (T1(1) / z) / c10::numbers::pi_v<T1> / (Npmu - Jpnul / Jnul * Nmu);
    } else {
      auto a = T1{0.25L}
          - (v - T1((z < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(v - z + T1{1.5L}))))) * (v - T1(
              (z < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(v - z + T1{1.5L})))));
      auto pq = c10::complex<T1>(-(T1(1) / z) / T1(2), T1(1));
      auto b = c10::complex<T1>(T1(2) * z, T1(2));
      auto fact = a * (T1(1) / z) / std::norm(pq);
      auto c = b + c10::complex<T1>{0, 1} * fact * std::conj(pq);
      auto d = std::conj(b) / std::norm(b);
      auto dl = c * d;
      pq *= dl;
      int i;
      for (i = 2; i <= 15000; ++i) {
        a += T1(2 * (i - 1));
        b += c10::complex<T1>{0, 1} * T1(2);
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

      const auto[p, q] = reinterpret_cast<T1(&)[2]>(pq);
      Jmu =
          std::sqrt(T1(2) * (T1(1) / z) / c10::numbers::pi_v<T1> / ((p - Jpnul / Jnul) * ((p - Jpnul / Jnul) / q) + q));
      Jmu = std::copysign(Jmu, Jnul);
      Nmu = (p - Jpnul / Jnul) / q * Jmu;
      Npmu = (p + q / ((p - Jpnul / Jnul) / q)) * Nmu;
      Nnu1 = (v - T1((z < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(v - z + T1{1.5L}))))) * (T1(1) / z)
          * Nmu - Npmu;
    }

    fact = Jmu / Jnul;

    if (std::abs(c10::numbers::pi_v<T1> * z * (fact * Jnul1) / T1(2)) > std::numeric_limits<T1>::min()) {
      for (int j = 1; j <= (z < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(v - z + T1{1.5L}))); j++) {
        Nmu = std::exchange(Nnu1,
                            (v - T1((z < T1(2) ? nearby_integer_n : std::max(0, static_cast<int>(v - z + T1{1.5L}))))
                                + j) * (T1(2) * (T1(1) / z)) * Nnu1 - Nmu);
      }

      return {v, z, fact * Jnul1, fact * Jpnu1, Nmu, v * (T1(1) / z) * Nmu - Nnu1};
    } else {
      return {v, z, fact * Jnul1, fact * Jpnu1, -std::numeric_limits<T1>::infinity(),
              std::numeric_limits<T1>::infinity()};
    }
  }
}
}
}
}
}
