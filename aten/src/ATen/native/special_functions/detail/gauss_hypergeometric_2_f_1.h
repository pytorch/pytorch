#pragma once

#include <ATen/native/special_functions/digamma.h>
#include <ATen/native/special_functions/detail/is_integer.h>
#include <ATen/native/special_functions/detail/ln_gamma_sign.h>
#include <ATen/native/special_functions/detail/ln_gamma.h>
#include <ATen/native/special_functions/detail/numeric_t.h>
#include <ATen/native/special_functions/detail/promote_t.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
gauss_hypergeometric_2_f_1(T1 a, T1 b, T1 c, T1 x) {
  using T2 = numeric_t<T1>;

  const auto d = c - a;
  const auto e = c - b;
  const auto f = d - b;
  const auto g = e - a;

  const auto is_integer_a = is_integer(a, T2(1000));
  const auto is_integer_b = is_integer(b, T2(1000));
  const auto is_integer_c = is_integer(c, T2(1000));

  const auto integer_a = is_integer_a();
  const auto integer_b = is_integer_b();
  const auto integer_c = is_integer_c();

  const auto real_a = std::real(a);
  const auto real_b = std::real(b);
  const auto real_c = std::real(c);
  const auto real_x = std::real(x);

  const auto abs_a = std::abs(a);
  const auto abs_b = std::abs(b);
  const auto abs_c = std::abs(c);
  const auto abs_d = std::abs(d);
  const auto abs_e = std::abs(e);
  const auto abs_g = std::abs(g);
  const auto abs_x = std::abs(x);

  const auto epsilon = std::numeric_limits<T2>::epsilon();

  const auto t1_maximum = std::numeric_limits<T1>::max();
  const auto t2_maximum = std::numeric_limits<T2>::max();

  const auto log_gamma_sign_c = log_gamma_sign(c);
  const auto log_gamma_sign_d = log_gamma_sign(d);
  const auto log_gamma_sign_e = log_gamma_sign(e);

  const auto log_gamma_c = ln_gamma(c);
  const auto log_gamma_f = ln_gamma(f);
  const auto log_gamma_d = ln_gamma(d);
  const auto log_gamma_e = ln_gamma(e);

  if (std::abs(x - T1(1)) < T2(1000) * epsilon && abs_g > T2(0) && !(is_integer_c && integer_c <= 0)) {
    const auto p = log_gamma_sign_c * log_gamma_sign_d * log_gamma_sign_e;
    const auto q = log_gamma_c + log_gamma_f - log_gamma_d - log_gamma_e;

    if (p == T2(0)) {
      return std::numeric_limits<T1>::quiet_NaN();
    } else if (std::abs(q) < std::log(t2_maximum)) {
      return p * std::exp(q);
    } else {
      throw std::domain_error("overflow of gamma functions");
    }
  } else if (abs_x >= T2(1)) {
    throw std::domain_error("argument outside unit circle");
  } else if (std::isnan(a) || std::isnan(b) || std::isnan(c) || std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (is_integer_c && integer_c <= 0) {
    return std::numeric_limits<T1>::infinity();
  } else if (abs_e < T2(1000) * epsilon || abs_d < T2(1000) * epsilon) {
    return std::pow(T1(1) - x, f);
  } else if (real_a >= T2(0) && real_b >= T2(0) && real_c >= T2(0) && real_x >= T2(0) && abs_x < T2{0.995L}) {
    auto p = T1(1);
    auto q = T1(1);

    for (unsigned int j = 0; j < 100000; j++) {
      p = p * ((a + T1(j)) * (b + T1(j)) * x / ((c + T1(j)) * T1(1 + j)));
      q = q + p;

      if (std::abs(p) < epsilon) {
        break;
      }
    }

    return q;
  } else if (abs_a < T2(10) && abs_b < T2(10)) {
    if (is_integer_a && integer_a < 0) {
      auto p = T1(1);
      auto q = T1(1);

      for (unsigned int j = 0; j < 100000; j++) {
        p = p * ((T1(integer_a) + T1(j)) * (b + T1(j)) * x / ((c + T1(j)) * T1(1 + j)));
        q = q + p;

        if (std::abs(p) < epsilon) {
          break;
        }
      }

      return q;
    } else if (is_integer_b && integer_b < 0) {
      auto p = T1(1);
      auto q = T1(1);

      for (unsigned int j = 0; j < 100000; j++) {
        p = p * ((T1(integer_b) + T1(j)) * (a + T1(j)) * x / ((c + T1(j)) * T1(1 + j)));
        q = q + p;

        if (std::abs(p) < epsilon) {
          break;
        }
      }

      return q;
    } else if (real_x < -T2{0.25L} || abs_x < T2{0.5L}) {
      auto f = T1(1);

      auto p = T1(1);
      auto q = T1(1) + (a + T1(1)) * (b + T1(1)) / (T1(2) * c) * -x;
      auto r = T1(1) + (a + T1(2)) * (b + T1(2)) / (T1(2) * (c + T1(1))) * -x
          * (T1(1) + (a + T1(1)) * (b + T1(1)) / (T1(2) * c) / T1(3) * -x);
      auto s = T1(1);
      auto t = q - a * b / c * -x;
      auto u = r - a * b / c * (T1(1) + (a + T1(2)) * (b + T1(2)) / (T1(2) * (c + T1(1))) * -x) * -x
          + a * b / c * ((a + T1(1)) * (b + T1(1)) / (T1(2) * c)) * (c / (c + T1(1))) * -x * -x;

      int n = 3;
      while (true) {
        auto v = (T2(1) + (T2(3 * (n * n)) + (a + b - T2(6)) * T2(n) + T2(2) - a * b - T2(2) * (a + b))
            / (T2(2 * (T2(2 * n - 1) - T2(2))) * (T2(n - 1) + c)) * -x) * u
            + (-(T2(n - 1) + a) * (T2(n - 1) + b) * (T2(n - 1) - c)
                / (T2(2 * (T2(2 * n - 1) - T2(2))) * (T2(n - 1) + c - T2(1)) * (T2(n - 1) + c))
                + -(T2(3 * (n * n)) - (a + b + T2(6)) * T2(n) + T2(2) - a * b) * (T2(n - 1) + a) * (T2(n - 1) + b)
                    / (T2(4 * T2(2 * n - 1) * (T2(2 * n - 1) - T2(2))) * (T2(n - 1) + c - T2(1)) * (T2(n - 1) + c))
                    * -x) * -x * t
            + ((T2(n - 1) + a - T2(1)) * (T2(n - 1) + a) * (T2(n - 1) + b - T2(1)) * (T2(n - 1) + b) * (T2(n - 2) - a)
                * (T2(n - 2) - b))
                / (T2(8 * (T2(2 * n - 1) - T2(2)) * (T2(2 * n - 1) - T2(2)) * (T2(2 * n - 1) - T2(2) - T2(2)))
                    * (T2(n - 3) + c) * (T2(n - 1) + c - T2(1)) * (T2(n - 1) + c)) * (-x * -x * -x) * s;
        auto w = (T2(1) + (T2(3 * (n * n)) + (a + b - T2(6)) * T2(n) + T2(2) - a * b - T2(2) * (a + b))
            / (T2(2 * (T2(2 * n - 1) - T2(2))) * (T2(n - 1) + c)) * -x) * r
            + (-(T2(n - 1) + a) * (T2(n - 1) + b) * (T2(n - 1) - c)
                / (T2(2 * (T2(2 * n - 1) - T2(2))) * (T2(n - 1) + c - T2(1)) * (T2(n - 1) + c))
                + -(T2(3 * (n * n)) - (a + b + T2(6)) * T2(n) + T2(2) - a * b) * (T2(n - 1) + a) * (T2(n - 1) + b)
                    / (T2(4 * T2(2 * n - 1) * (T2(2 * n - 1) - T2(2))) * (T2(n - 1) + c - T2(1)) * (T2(n - 1) + c))
                    * -x) * -x * q
            + ((T2(n - 1) + a - T2(1)) * (T2(n - 1) + a) * (T2(n - 1) + b - T2(1)) * (T2(n - 1) + b) * (T2(n - 2) - a)
                * (T2(n - 2) - b))
                / (T2(8 * (T2(2 * n - 1) - T2(2)) * (T2(2 * n - 1) - T2(2)) * (T2(2 * n - 1) - T2(2) - T2(2)))
                    * (T2(n - 3) + c) * (T2(n - 1) + c - T2(1)) * (T2(n - 1) + c)) * (-x * -x * -x) * p;

        f = v / w;

        if (std::abs((f - v / w) / f) < epsilon || n > 20000) {
          break;
        }

        const auto abs_v = std::abs(v);
        const auto abs_w = std::abs(w);

        if (abs_v > std::pow(t1_maximum, 1 / T2(6)) || abs_w > std::pow(t1_maximum, 1 / T2(6))) {
          p /= std::pow(t1_maximum, 1 / T2(6));
          q /= std::pow(t1_maximum, 1 / T2(6));
          r /= std::pow(t1_maximum, 1 / T2(6));
          s /= std::pow(t1_maximum, 1 / T2(6));
          t /= std::pow(t1_maximum, 1 / T2(6));
          u /= std::pow(t1_maximum, 1 / T2(6));
          v /= std::pow(t1_maximum, 1 / T2(6));
          w /= std::pow(t1_maximum, 1 / T2(6));
        } else if (abs_v < T2(1) / std::pow(t1_maximum, 1 / T2(6)) || abs_w < T2(1) / std::pow(t1_maximum, 1 / T2(6))) {
          p *= std::pow(t1_maximum, 1 / T2(6));
          q *= std::pow(t1_maximum, 1 / T2(6));
          r *= std::pow(t1_maximum, 1 / T2(6));
          s *= std::pow(t1_maximum, 1 / T2(6));
          t *= std::pow(t1_maximum, 1 / T2(6));
          u *= std::pow(t1_maximum, 1 / T2(6));
          v *= std::pow(t1_maximum, 1 / T2(6));
          w *= std::pow(t1_maximum, 1 / T2(6));
        }

        n++;

        p = q;
        q = r;
        r = w;
        s = t;
        t = u;
        u = v;
      }

      if (n >= 20000) {
        throw std::runtime_error("iteration failed to converge");
      }

      return f;
    } else if (abs_c > T2(10)) {
      auto p = T1(1);
      auto q = T1(1);

      for (unsigned int j = 0; j < 100000; j++) {
        p = p * ((a + T1(j)) * (b + T1(j)) * x / ((c + T1(j)) * T1(1 + j)));
        q = q + p;

        if (std::abs(p) < epsilon) {
          break;
        }
      }

      return q;
    } else if (is_integer(f, T2(1000))) {
      T1 F2;
      T1 F1;
      T1 d2;
      T1 d1;

      if (std::real(f) >= T2(0)) {
        d1 = f;
        d2 = T1(0);
      } else {
        d1 = T1(0);
        d2 = f;
      }

      if (std::abs(f) < epsilon) {
        F1 = T1(0);
      } else {
        bool ok_d1 = true;
        T1 lng_ad;
        T1 lng_ad1;
        T1 lng_bd1;
        T1 sgn_ad;
        T1 sgn_ad1;
        T1 sgn_bd1;
        try {
          sgn_ad = log_gamma_sign(std::abs(f));
          lng_ad = ln_gamma(std::abs(f));
          sgn_ad1 = log_gamma_sign(a + d1);
          lng_ad1 = ln_gamma(a + d1);
          sgn_bd1 = log_gamma_sign(b + d1);
          lng_bd1 = ln_gamma(b + d1);
        } catch (...) {
          ok_d1 = false;
        }

        if (ok_d1) {
          auto s = T1(1);
          auto t = T1(1);
          auto ln_pre1 = lng_ad + log_gamma_c + d2 * std::log1p(-x) - lng_ad1 - lng_bd1;

          if (std::abs(ln_pre1) > std::log(t2_maximum)) {
            throw std::runtime_error("hyperg_reflect: overflow of gamma functions");
          }

          for (int j = 1; j < std::abs(f); j++) {
            t = t * ((a + d2 + T2(j - 1)) * (b + d2 + T2(j - 1)) / (T2(1) + d2 + T2(j - 1)) / T2(j) * (T2(1) - x));
            s = s + t;
          }

          if (std::abs(ln_pre1) > std::log(t2_maximum))
            throw std::runtime_error("hyperg_reflect: overflow of gamma functions");
          else
            F1 = sgn_ad * sgn_ad1 * sgn_bd1
                * std::exp(ln_pre1) * s;
        } else {
          // Gamma functions in the denominator were not ok (they diverged).
          // So the F1 term is zero.
          F1 = T1(0);
        }
      }

      bool ok_d2 = true;
      T1 lng_ad2;
      T1 lng_bd2;
      T1 sgn_ad2;
      T1 sgn_bd2;

      try {
        sgn_ad2 = log_gamma_sign(a + d2);
        lng_ad2 = ln_gamma(a + d2);
        sgn_bd2 = log_gamma_sign(b + d2);
        lng_bd2 = ln_gamma(b + d2);
      } catch (...) {
        ok_d2 = false;
      }

      if (ok_d2) {
        auto psi_term =
            -c10::numbers::egamma_v<T2> + at::native::special_functions::digamma(T2(1) + std::abs(f))
                - at::native::special_functions::digamma(a + d1)
                - at::native::special_functions::digamma(b + d1) - std::log1p(-x);
        auto fact = T1(1);
        auto sum2 = psi_term;
        auto ln_pre2 = log_gamma_c + d1 * std::log1p(-x) - lng_ad2 - lng_bd2;

        if (std::abs(ln_pre2) > std::log(t2_maximum))
          throw std::runtime_error("hyperg_reflect: overflow of gamma functions");

        int j;
        for (j = 1; j < 2000; j++) {
          psi_term += T1(1) / T1(j) + T1(1) / (std::abs(f) + j)
              - (T1(1) / (a + d1 + T1(j - 1)) + T1(1) / (b + d1 + T1(j - 1)));
          fact *= (a + d1 + T1(j - 1)) * (b + d1 + T1(j - 1)) / ((std::abs(f) + j) * j) * (T1(1) - x);
          sum2 += fact * psi_term;

          if (std::abs(fact * psi_term) < epsilon * std::abs(sum2)) {
            break;
          }
        }
        if (j == 2000)
          throw std::runtime_error("hyperg_reflect: sum F2 failed to converge");

        if (sum2 == T1(0))
          F2 = T1(0);
        else
          F2 = sgn_ad2 * sgn_bd2 * std::exp(ln_pre2) * sum2;
      } else {
        F2 = T1(0);
      }

      if (is_integer(f, T2(1000))() % 2 == 1) {
        return F1 + -T1(1) * F2;
      } else {
        return F1 + T1(1) * F2;
      }
    } else {
      // These gamma functions appear in the denominator, so we
      // catch their harmless domain errors and set the terms to zero.
      bool ok1 = true;
      auto sgn_g1ca = T1(0), ln_g1ca = T1(0);
      auto sgn_g1cb = T1(0), ln_g1cb = T1(0);

      try {
        sgn_g1ca = log_gamma_sign_d;
        ln_g1ca = log_gamma_d;
        sgn_g1cb = log_gamma_sign_e;
        ln_g1cb = log_gamma_e;
      } catch (...) {
        ok1 = false;
      }

      bool ok2 = true;
      auto sgn_g2a = T1(0), ln_g2a = T1(0);
      auto sgn_g2b = T1(0), ln_g2b = T1(0);
      try {
        sgn_g2a = log_gamma_sign(a);
        ln_g2a = ln_gamma(a);
        sgn_g2b = log_gamma_sign(b);
        ln_g2b = ln_gamma(b);
      } catch (...) {
        ok2 = false;
      }

      const auto sgn1 = log_gamma_sign_c * log_gamma_sign(f) * sgn_g1ca * sgn_g1cb;
      const auto sgn2 = log_gamma_sign_c * log_gamma_sign(-f) * sgn_g2a * sgn_g2b;

      T1 pre1, pre2;

      if (ok1 && ok2) {
        auto ln_pre1 = log_gamma_c + log_gamma_f - ln_g1ca - ln_g1cb;
        auto ln_pre2 =
            log_gamma_c + ln_gamma(-f) - ln_g2a - ln_g2b + f * std::log(T1(1) - x);

        if (std::abs(ln_pre1) < std::log(t2_maximum)
            && std::abs(ln_pre2) < std::log(t2_maximum)) {
          pre1 = sgn1 * std::exp(ln_pre1);
          pre2 = sgn2 * std::exp(ln_pre2);
        } else {
          throw std::runtime_error("hyperg_reflect: overflow of gamma functions");
        }
      } else if (ok1 && !ok2) {
        auto ln_pre1 = log_gamma_c + log_gamma_f - ln_g1ca - ln_g1cb;

        if (std::abs(ln_pre1) < std::log(t2_maximum)) {
          pre1 = sgn1 * std::exp(ln_pre1);
          pre2 = T1(0);
        } else {
          throw std::runtime_error("hyperg_reflect: overflow of gamma functions");
        }
      } else if (!ok1 && ok2) {
        auto ln_pre2 =
            log_gamma_c + ln_gamma(-f) - ln_g2a - ln_g2b + f * std::log(T1(1) - x);

        if (std::abs(ln_pre2) < std::log(t2_maximum)) {
          pre1 = T1(0);
          pre2 = sgn2 * std::exp(ln_pre2);
        } else {
          throw std::runtime_error("hyperg_reflect: overflow of gamma functions");
        }
      } else {
        throw std::runtime_error("hyperg_reflect: underflow of gamma functions");
      }

      auto p = T1(1);
      auto q = T1(1);

      for (unsigned int j = 0; j < 100000; j++) {
        p = p * ((a + T1(j)) * (b + T1(j)) * (T1(1) - x) / ((T1(1) - f + T1(j)) * T1(1 + j)));
        q = q + p;

        if (std::abs(p) < epsilon) {
          break;
        }
      }

      auto r = T1(1);
      auto s = T1(1);

      for (unsigned int j = 0; j < 100000; j++) {
        r = r * ((d + T1(j)) * (e + T1(j)) * (T1(1) - x) / ((T1(1) + f + T1(j)) * T1(1 + j)));
        s = s + r;

        if (std::abs(r) < epsilon) {
          break;
        }
      }

      return pre1 * q + pre2 * s;
    }
  } else {
    auto f = T1(1);

    auto p = T1(1);
    auto q = T1(1) + (a + T1(1)) * (b + T1(1)) / (T1(2) * c) * -x;
    auto r = T1(1) + (a + T1(2)) * (b + T1(2)) / (T1(2) * (c + T1(1))) * -x
        * (T1(1) + (a + T1(1)) * (b + T1(1)) / (T1(2) * c) / T1(3) * -x);
    auto s = T1(1);
    auto t = q - a * b / c * -x;
    auto u = r - a * b / c * (T1(1) + (a + T1(2)) * (b + T1(2)) / (T1(2) * (c + T1(1))) * -x) * -x
        + a * b / c * ((a + T1(1)) * (b + T1(1)) / (T1(2) * c)) * (c / (c + T1(1))) * -x * -x;

    int n = 3;

    while (true) {
      auto v = (T2(1) + (T2(3 * (n * n)) + (a + b - T2(6)) * T2(n) + T2(2) - a * b - T2(2) * (a + b))
          / (T2(2 * (T2(2 * n - 1) - T2(2))) * (T2(n - 1) + c)) * -x) * u
          + (-(T2(n - 1) + a) * (T2(n - 1) + b) * (T2(n - 1) - c)
              / (T2(2 * (T2(2 * n - 1) - T2(2))) * (T2(n - 1) + c - T2(1)) * (T2(n - 1) + c))
              + -(T2(3 * (n * n)) - (a + b + T2(6)) * T2(n) + T2(2) - a * b) * (T2(n - 1) + a) * (T2(n - 1) + b)
                  / (T2(4 * T2(2 * n - 1) * (T2(2 * n - 1) - T2(2))) * (T2(n - 1) + c - T2(1)) * (T2(n - 1) + c)) * -x)
              * -x * t
          + ((T2(n - 1) + a - T2(1)) * (T2(n - 1) + a) * (T2(n - 1) + b - T2(1)) * (T2(n - 1) + b) * (T2(n - 2) - a)
              * (T2(n - 2) - b))
              / (T2(8 * (T2(2 * n - 1) - T2(2)) * (T2(2 * n - 1) - T2(2)) * (T2(2 * n - 1) - T2(2) - T2(2)))
                  * (T2(n - 3) + c) * (T2(n - 1) + c - T2(1)) * (T2(n - 1) + c)) * (-x * -x * -x) * s;
      auto w = (T2(1) + (T2(3 * (n * n)) + (a + b - T2(6)) * T2(n) + T2(2) - a * b - T2(2) * (a + b))
          / (T2(2 * (T2(2 * n - 1) - T2(2))) * (T2(n - 1) + c)) * -x) * r
          + (-(T2(n - 1) + a) * (T2(n - 1) + b) * (T2(n - 1) - c)
              / (T2(2 * (T2(2 * n - 1) - T2(2))) * (T2(n - 1) + c - T2(1)) * (T2(n - 1) + c))
              + -(T2(3 * (n * n)) - (a + b + T2(6)) * T2(n) + T2(2) - a * b) * (T2(n - 1) + a) * (T2(n - 1) + b)
                  / (T2(4 * T2(2 * n - 1) * (T2(2 * n - 1) - T2(2))) * (T2(n - 1) + c - T2(1)) * (T2(n - 1) + c)) * -x)
              * -x * q
          + ((T2(n - 1) + a - T2(1)) * (T2(n - 1) + a) * (T2(n - 1) + b - T2(1)) * (T2(n - 1) + b) * (T2(n - 2) - a)
              * (T2(n - 2) - b))
              / (T2(8 * (T2(2 * n - 1) - T2(2)) * (T2(2 * n - 1) - T2(2)) * (T2(2 * n - 1) - T2(2) - T2(2)))
                  * (T2(n - 3) + c) * (T2(n - 1) + c - T2(1)) * (T2(n - 1) + c)) * (-x * -x * -x) * p;

      f = v / w;

      if (std::abs((f - v / w) / f) < epsilon || n > 20000) {
        break;
      }

      const auto abs_v = std::abs(v);
      const auto abs_w = std::abs(w);

      if (abs_v > std::pow(t1_maximum, 1 / T2(6)) || abs_w > std::pow(t1_maximum, 1 / T2(6))) {
        v /= std::pow(t1_maximum, 1 / T2(6));
        w /= std::pow(t1_maximum, 1 / T2(6));
        u /= std::pow(t1_maximum, 1 / T2(6));
        r /= std::pow(t1_maximum, 1 / T2(6));
        t /= std::pow(t1_maximum, 1 / T2(6));
        q /= std::pow(t1_maximum, 1 / T2(6));
        s /= std::pow(t1_maximum, 1 / T2(6));
        p /= std::pow(t1_maximum, 1 / T2(6));
      } else if (abs_v < T2(1) / std::pow(t1_maximum, 1 / T2(6)) || abs_w < T2(1) / std::pow(t1_maximum, 1 / T2(6))) {
        v *= std::pow(t1_maximum, 1 / T2(6));
        w *= std::pow(t1_maximum, 1 / T2(6));
        u *= std::pow(t1_maximum, 1 / T2(6));
        r *= std::pow(t1_maximum, 1 / T2(6));
        t *= std::pow(t1_maximum, 1 / T2(6));
        q *= std::pow(t1_maximum, 1 / T2(6));
        s *= std::pow(t1_maximum, 1 / T2(6));
        p *= std::pow(t1_maximum, 1 / T2(6));
      }

      n++;
      p = q;
      q = r;
      r = w;
      s = t;
      t = u;
      u = v;
    }

    if (n >= 20000) {
      throw std::runtime_error("iteration failed to converge");
    }

    return f;
  }
}
}
}
}
}
