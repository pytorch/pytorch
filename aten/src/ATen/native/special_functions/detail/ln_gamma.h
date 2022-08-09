#pragma once

#include <ATen/native/special_functions/detail/cos_pi.h>
#include <ATen/native/special_functions/detail/is_complex_v.h>
#include <ATen/native/special_functions/detail/numeric_t.h>
#include <ATen/native/special_functions/detail/promote_t.h>
#include <ATen/native/special_functions/detail/sin_pi.h>
#include <ATen/native/special_functions/detail/tan_pi.h>
#include <c10/util/numbers.h>
#include <ATen/native/special_functions/sin_pi.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename Tp>
struct gamma_spouge_data {
};

template<>
struct gamma_spouge_data<float> {
  static constexpr std::array<float, 7>
      s_cheby
      {
          2.901419e+03F,
          -5.929168e+03F,
          4.148274e+03F,
          -1.164761e+03F,
          1.174135e+02F,
          -2.786588e+00F,
          3.775392e-03F,
      };
};

template<>
struct gamma_spouge_data<double> {
  static constexpr std::array<double, 18>
      s_cheby
      {
          2.785716565770350e+08,
          -1.693088166941517e+09,
          4.549688586500031e+09,
          -7.121728036151557e+09,
          7.202572947273274e+09,
          -4.935548868770376e+09,
          2.338187776097503e+09,
          -7.678102458920741e+08,
          1.727524819329867e+08,
          -2.595321377008346e+07,
          2.494811203993971e+06,
          -1.437252641338402e+05,
          4.490767356961276e+03,
          -6.505596924745029e+01,
          3.362323142416327e-01,
          -3.817361443986454e-04,
          3.273137866873352e-08,
          -7.642333165976788e-15,
      };
};

template<>
struct gamma_spouge_data<long double> {
  static constexpr std::array<long double, 22>
      s_cheby
      {
          1.681473171108908244e+10L,
          -1.269150315503303974e+11L,
          4.339449429013039995e+11L,
          -8.893680202692714895e+11L,
          1.218472425867950986e+12L,
          -1.178403473259353616e+12L,
          8.282455311246278274e+11L,
          -4.292112878930625978e+11L,
          1.646988347276488710e+11L,
          -4.661514921989111004e+10L,
          9.619972564515443397e+09L,
          -1.419382551781042824e+09L,
          1.454145470816386107e+08L,
          -9.923020719435758179e+06L,
          4.253557563919127284e+05L,
          -1.053371059784341875e+04L,
          1.332425479537961437e+02L,
          -7.118343974029489132e-01L,
          1.172051640057979518e-03L,
          -3.323940885824119041e-07L,
          4.503801674404338524e-12L,
          -5.320477002211632680e-20L,
      };
};

template<typename T1>
constexpr T1
spouge_binet1p(T1 z) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  T2 p = c10::numbers::sqrttau_v<T3>;

  for (auto j = 0ull; j < gamma_spouge_data<T3>::s_cheby.size(); j++) {
    p = p + (gamma_spouge_data<T3>::s_cheby[j] / (z + T3(j + 1)));
  }

  return p;
}

template<typename T1>
constexpr T1
spouge_log_gamma1p(T1 z) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  auto a = T3{gamma_spouge_data<T3>::s_cheby.size() + 1};

  if (std::real(z) < T3(-1)) {
    if (!is_complex_v<T2>) {
      return c10::numbers::lnpi_v<T3> - std::log(std::abs(at::native::special_functions::sin_pi(z)))
          - spouge_log_gamma1p(-T3(1) - z);
    } else {
      return c10::numbers::lnpi_v<T3> - std::log(at::native::special_functions::sin_pi(z))
          - spouge_log_gamma1p(-T3(1) - z);
    }
  } else {
    auto sum = spouge_binet1p(z);

    if (!is_complex_v<T2>) {
      sum = std::abs(sum);
    }

    return std::log(sum) + (z + T3(0.5L)) * std::log(z + a) - (z + a);
  }
}

template<typename T1>
T1
ln_gamma(T1 a) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  if (std::real(a) < T3{0.5L}) {
    if (std::abs(at::native::special_functions::sin_pi(a)) < std::numeric_limits<T3>::epsilon()) {
      return std::numeric_limits<T3>::infinity();
    } else {
      return c10::numbers::lnpi_v<T1> - std::log(std::abs(at::native::special_functions::sin_pi(a)))
          - ln_gamma(T2(1) - a);
    }
  } else if (std::real(a) > T3(1) && std::abs(a) < c10::numbers::factorials_size<T1>) {
    auto fact = T1(1);
    auto arg = a;

    while (std::real(arg) > T3(1)) {
      fact *= (arg -= T3(1));
    }

    return std::log(fact) + ln_gamma(arg);
  } else {
    return spouge_log_gamma1p(a - T3(1));
  }
}

template<typename T1>
std::complex<T1>
log_gamma(std::complex<T1> a) {
  using T2 = T1;
  using T3 = numeric_t<T2>;
  using T4 = std::complex<T3>;

  if (is_integer(a)) {
    if (is_integer(a)() <= 0)
      return std::numeric_limits<T3>::quiet_NaN();
    else if (is_integer(a)() < static_cast<int>(c10::numbers::factorials_size<T3>))
      return T3((c10::numbers::log_factorials_v[is_integer(a)() - 1]));
    else
      return ln_gamma(T3(is_integer(a)()));
  } else if (std::real(a) >= T3{0.5L}) {
    return spouge_log_gamma1p(a - T3(1));
  } else {
    if (std::abs(at::native::special_functions::sin_pi(a)) < std::numeric_limits<T3>::epsilon()) {
      return T4(std::numeric_limits<T3>::quiet_NaN(), T3(0));
    } else
      return c10::numbers::lnpi_v<T3> - std::log(at::native::special_functions::sin_pi(a)) - ln_gamma(T2(1) - a);
  }
}
}
}
}
}
