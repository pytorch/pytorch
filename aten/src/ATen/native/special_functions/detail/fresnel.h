#pragma once

#include <complex>

#include <c10/util/numbers.h>

namespace at::native::special_functions::detail {
template<typename T1>
void
fresnel_series(const T1 ax, T1 &c, T1 &s) {
  // Evaluate S and C by series expansion.
  auto summation = T1(0);

  auto s_summation = T1(0);
  auto c_summation = ax;

  auto sign = T1(1);
  auto fact = c10::numbers::pi_v<T1> / T1(2) * ax * ax;
  auto odd = true;
  auto term = ax;
  auto n = 3;
  auto k = 0;

  for (k = 1; k <= 100; ++k) {
    term *= fact / k;

    summation = summation + (sign * term / n);

    T1 test = std::abs(summation) * (T1(5) * std::numeric_limits<T1>::epsilon());

    if (odd) {
      sign = -sign;

      s_summation = summation;
      summation = c_summation;
    } else {
      c_summation = summation;
      summation = s_summation;
    }

    if (term < test) {
      break;
    }

    odd = !odd;

    n = n + 2;
  }

  if (k > 100) {
    throw std::runtime_error("fresnel_series: series evaluation failed");
  }

  c = c_summation;
  s = s_summation;
}

template<typename T1>
void
fresnel_cont_frac(const T1 ax, T1 &Cf, T1 &Sf) {
  std::complex<T1> b(T1(1), -(c10::numbers::pi_v<T1> * ax * ax));
  std::complex<T1> cc(T1(1) / std::numeric_limits<T1>::min(), T1(0));
  auto h = T1(1) / b;
  auto d = h;
  auto n = -1;
  auto k = 0;

  for (k = 2; k <= 100; ++k) {
    n += 2;
    b += T1(4);
    d = T1(1) / (-T1(n * (n + 1)) * d + b);
    cc = b + -T1(n * (n + 1)) / cc;
    h *= cc * d;

    if (std::abs((cc * d).real() - T1(1)) + std::abs((cc * d).imag()) < T1(5) * std::numeric_limits<T1>::epsilon()) {
      break;
    }
  }

  if (k > 100) {
    throw std::runtime_error("fresnel_cont_frac: continued fraction evaluation failed");
  }

  h = h * std::complex<T1>(ax, -ax);

  auto phase = std::polar(T1(1), c10::numbers::pi_v<T1> * ax * ax / T1(2));

  auto cs = std::complex<T1>(T1{0.5L}, T1{0.5L}) * (T1(1) - phase * h);

  Cf = cs.real();
  Sf = cs.imag();
}

template<typename T1>
std::complex<T1>
fresnel(const T1 x) {
  if (std::isnan(x)) {
    return std::complex<T1>{
        std::numeric_limits<T1>::quiet_NaN(),
        std::numeric_limits<T1>::quiet_NaN(),
    };
  }

  auto c = T1(0);
  auto s = T1(0);

  if (std::abs(x) < std::sqrt(std::numeric_limits<T1>::min())) {
    c = std::abs(x);
    s = T1(0);
  } else if (std::abs(x) < T1{1.5L}) {
    fresnel_series(std::abs(x), c, s);
  } else {
    fresnel_cont_frac(std::abs(x), c, s);
  }

  if (x < T1(0)) {
    c = -c;
    s = -s;
  }

  return std::complex<T1>(c, s);
}
}
